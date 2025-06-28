import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper
import argparse
import os
import re

# --- Helper Functions ---

def get_attribute(node, attr_name, default_value=None):
    """Helper to get a node's attribute. Returns default if not found."""
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                return tuple(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.FLOATS:
                return tuple(attr.floats)
            elif attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode('utf-8')
            elif attr.type == onnx.AttributeProto.TENSOR:
                return numpy_helper.to_array(attr.t)
    return default_value

def sanitize_name(name):
    """Converts a name to a valid Python variable name."""
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized or "unnamed"

# --- The Main Converter Class ---

class OnnxToPytorch:
    def __init__(self, onnx_model_path, output_py_path):
        self.onnx_model_path = onnx_model_path
        self.output_py_path = output_py_path
        self.model_name = sanitize_name(os.path.splitext(os.path.basename(output_py_path))[0])
        
        print("Loading ONNX model...")
        self.onnx_model = onnx.load(onnx_model_path)
        self.graph = self.onnx_model.graph
        
        self.tensor_map = {}
        self.init_lines = []
        self.forward_lines = []
        self.state_dict = {}

        print("Parsing initializers (weights and biases)...")
        self.initializers = {init.name: numpy_helper.to_array(init) for init in self.graph.initializer}

    def convert(self):
        """Main conversion function."""
        print("Starting conversion process...")
        
        for i, input_tensor in enumerate(self.graph.input):
            if input_tensor.name not in self.initializers:
                input_name = sanitize_name(input_tensor.name)
                self.tensor_map[input_tensor.name] = input_name

        for i, node in enumerate(self.graph.node):
            op_type = node.op_type
            handler = getattr(self, f"handle_{op_type}", self.handle_unknown)
            handler(node, i)

        self._generate_output_return()
        self._write_pytorch_file()
        self._save_state_dict()
        
        print("\nConversion complete!")
        print(f"PyTorch model saved to: {self.output_py_path}")
        print(f"Model weights saved to: {self._get_state_dict_path()}")

    def _write_pytorch_file(self):
        with open(self.output_py_path, "w") as f:
            f.write("import torch\n")
            f.write("import torch.nn as nn\n")
            f.write("import torch.nn.functional as F\n\n")
            f.write(f"class {self.model_name}(nn.Module):\n")
            f.write("    def __init__(self):\n")
            f.write(f"        super({self.model_name}, self).__init__()\n")
            for line in sorted(list(set(self.init_lines))):
                f.write(f"        {line}\n")
            
            f.write("\n")
            f.write(f"    def forward(self, {', '.join(self.get_model_inputs())}):\n")
            for line in self.forward_lines:
                f.write(f"        {line}\n")
            
            f.write("\n")
            f.write("# How to load the model and weights:\n")
            f.write(f"# 1. Create an instance of the model:\n")
            f.write(f"#    model = {self.model_name}()\n")
            f.write(f"# 2. Load the state dictionary:\n")
            f.write(f"#    state_dict_path = '{os.path.basename(self._get_state_dict_path())}'\n")
            f.write(f"#    model.load_state_dict(torch.load(state_dict_path))\n")
            f.write(f"# 3. Set the model to evaluation mode:\n")
            f.write(f"#    model.eval()\n")

    def _save_state_dict(self):
        torch_state_dict = {k: torch.from_numpy(v) for k, v in self.state_dict.items()}
        torch.save(torch_state_dict, self._get_state_dict_path())

    def _get_state_dict_path(self):
        return os.path.splitext(self.output_py_path)[0] + ".pth"

    def get_model_inputs(self):
        model_inputs = []
        for input_tensor in self.graph.input:
            if input_tensor.name not in self.initializers:
                model_inputs.append(sanitize_name(input_tensor.name))
        return model_inputs

    def _generate_output_return(self):
        output_names = [self.tensor_map.get(o.name, sanitize_name(o.name)) for o in self.graph.output]
        if len(output_names) == 1:
            self.forward_lines.append(f"return {output_names[0]}")
        else:
            self.forward_lines.append(f"return {', '.join(output_names)}")

    # --- Operator Handlers (Alphabetical Order) ---

    def handle_Add(self, node, layer_idx):
        input1_name = self.tensor_map.get(node.input[0])
        input2_name = self.tensor_map.get(node.input[1])
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        self.forward_lines.append(f"{output_name} = {input1_name} + {input2_name}")

    def handle_Clip(self, node, layer_idx):
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        
        min_val, max_val = None, None
        if len(node.input) > 1 and node.input[1] in self.initializers:
            min_val = self.initializers[node.input[1]].item()
        if len(node.input) > 2 and node.input[2] in self.initializers:
            max_val = self.initializers[node.input[2]].item()
        
        if min_val is None: min_val = get_attribute(node, 'min')
        if max_val is None: max_val = get_attribute(node, 'max')
            
        if min_val is not None and max_val is not None:
            self.forward_lines.append(f"{output_name} = torch.clamp({input_name}, min={min_val}, max={max_val})")
        elif min_val is not None:
            self.forward_lines.append(f"{output_name} = torch.clamp({input_name}, min={min_val})")
        elif max_val is not None:
            self.forward_lines.append(f"{output_name} = torch.clamp({input_name}, max={max_val})")
        else:
            self.forward_lines.append(f"{output_name} = {input_name}")
    
    def handle_Concat(self, node, layer_idx):
        input_names = [self.tensor_map[i] for i in node.input]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        
        axis = get_attribute(node, 'axis')
        
        self.forward_lines.append(f"{output_name} = torch.cat([{', '.join(input_names)}], dim={axis})")

    def handle_Conv(self, node, layer_idx):
        layer_name = f"conv_{layer_idx}"
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name

        kernel_shape = get_attribute(node, 'kernel_shape')
        strides = get_attribute(node, 'strides', (1, 1))
        pads = get_attribute(node, 'pads', (0, 0, 0, 0))
        group = get_attribute(node, 'group', 1)
        padding = (pads[0], pads[1])
        
        weight_name = node.input[1]
        weight = self.initializers[weight_name]
        in_channels = weight.shape[1] * group
        out_channels = weight.shape[0]
        has_bias = len(node.input) > 2
        
        self.init_lines.append(f"# Attributes: { {attr.name: get_attribute(node, attr.name) for attr in node.attribute} }")
        self.init_lines.append(f"self.{layer_name} = nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_shape}, stride={strides}, padding={padding}, groups={group}, bias={has_bias})")
        self.forward_lines.append(f"{output_name} = self.{layer_name}({input_name})")
        
        self.state_dict[f"{layer_name}.weight"] = weight
        if has_bias:
            bias_name = node.input[2]
            bias = self.initializers[bias_name]
            self.state_dict[f"{layer_name}.bias"] = bias
            
    def handle_Flatten(self, node, layer_idx):
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        axis = get_attribute(node, 'axis', 1)
        self.forward_lines.append(f"{output_name} = torch.flatten({input_name}, start_dim={axis})")

    def handle_Gemm(self, node, layer_idx):
        layer_name = f"linear_{layer_idx}"
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name

        weight_name = node.input[1]
        weight = self.initializers[weight_name]
        out_features, in_features = weight.shape
        has_bias = len(node.input) > 2
        
        self.init_lines.append(f"self.{layer_name} = nn.Linear(in_features={in_features}, out_features={out_features}, bias={has_bias})")
        self.forward_lines.append(f"{output_name} = self.{layer_name}({input_name})")
        
        self.state_dict[f"{layer_name}.weight"] = weight
        if has_bias:
            bias_name = node.input[2]
            bias = self.initializers[bias_name]
            self.state_dict[f"{layer_name}.bias"] = bias

    def handle_GlobalAveragePool(self, node, layer_idx):
        layer_name = f"gap_{layer_idx}"
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        
        self.init_lines.append(f"self.{layer_name} = nn.AdaptiveAvgPool2d((1, 1))")
        self.forward_lines.append(f"{output_name} = self.{layer_name}({input_name})")

    def handle_MaxPool(self, node, layer_idx):
        layer_name = f"maxpool_{layer_idx}"
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        
        kernel_shape = get_attribute(node, 'kernel_shape')
        strides = get_attribute(node, 'strides')
        pads = get_attribute(node, 'pads', (0,0))
        
        self.init_lines.append(f"self.{layer_name} = nn.MaxPool2d(kernel_size={kernel_shape}, stride={strides}, padding={pads[0]})")
        self.forward_lines.append(f"{output_name} = self.{layer_name}({input_name})")

    def handle_Pad(self, node, layer_idx):
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        
        pads_name = node.input[1]
        pads = self.initializers[pads_name]
        torch_pads = (int(pads[3]), int(pads[7]), int(pads[2]), int(pads[6]))
        
        mode = get_attribute(node, 'mode', 'constant')
        value = 0.0
        if len(node.input) > 2:
            value_name = node.input[2]
            value = self.initializers[value_name].item()

        self.forward_lines.append(f"{output_name} = F.pad({input_name}, {torch_pads}, mode='{mode}', value={value})")

    def handle_PRelu(self, node, layer_idx):
        layer_name = f"prelu_{layer_idx}"
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        
        slope_name = node.input[1]
        slope = self.initializers[slope_name]
        num_parameters = 1 if slope.ndim == 0 or len(slope) == 1 else slope.shape[0]

        self.init_lines.append(f"self.{layer_name} = nn.PReLU(num_parameters={num_parameters})")
        self.forward_lines.append(f"{output_name} = self.{layer_name}({input_name})")
        self.state_dict[f"{layer_name}.weight"] = slope

    def handle_Relu(self, node, layer_idx):
        layer_name = f"relu_{layer_idx}"
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        self.init_lines.append(f"self.{layer_name} = nn.ReLU()")
        self.forward_lines.append(f"{output_name} = self.{layer_name}({input_name})")

    def handle_Reshape(self, node, layer_idx):
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name

        shape_name = node.input[1]
        shape = self.initializers[shape_name]
        shape_str = ', '.join(map(str, shape)).replace('1,', f'{input_name}.size(0),', 1)
        self.forward_lines.append(f"{output_name} = {input_name}.reshape({shape_str})")

    def handle_Resize(self, node, layer_idx):
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name

        mode = get_attribute(node, 'mode', 'nearest')
        if mode.lower() == 'linear': # ONNX 'linear' is 'bilinear' for 4D tensors in PyTorch
            mode = 'bilinear'

        align_corners = False
        coord_mode = get_attribute(node, 'coordinate_transformation_mode')
        if coord_mode == 'align_corners':
            align_corners = True
        
        # Opset 11+ uses 'sizes' or 'scales' as the 2nd/3rd input
        if len(node.input) > 2: # Check for scales input
            scales_name = node.input[2]
            if scales_name in self.initializers:
                scales = self.initializers[scales_name]
                # PyTorch uses scale_factor for H, W dims
                scale_factor_str = f"scale_factor=({scales[2]}, {scales[3]})"
                self.forward_lines.append(f"{output_name} = F.interpolate({input_name}, {scale_factor_str}, mode='{mode}', align_corners={align_corners or 'None'})")
                return

        if len(node.input) > 1: # Check for sizes input
            sizes_name = node.input[1]
            if sizes_name in self.initializers:
                sizes = self.initializers[sizes_name]
                size_str = f"size=({int(sizes[2])}, {int(sizes[3])})"
                self.forward_lines.append(f"{output_name} = F.interpolate({input_name}, {size_str}, mode='{mode}', align_corners={align_corners or 'None'})")
                return

        # Fallback for older opsets or dynamic sizes (not fully supported here)
        self.forward_lines.append(f"# ---- Operator 'Resize' (node {layer_idx}) with dynamic size not fully supported ----")
        self.forward_lines.append(f"{output_name} = {input_name} # Placeholder for Resize")


    def handle_Sigmoid(self, node, layer_idx):
        layer_name = f"sigmoid_{layer_idx}"
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        self.init_lines.append(f"self.{layer_name} = nn.Sigmoid()")
        self.forward_lines.append(f"{output_name} = self.{layer_name}({input_name})")

    def handle_Squeeze(self, node, layer_idx):
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        axes = get_attribute(node, 'axes')
        if axes:
            self.forward_lines.append(f"{output_name} = torch.squeeze({input_name}, dim={axes[0]})")
        else:
            self.forward_lines.append(f"{output_name} = torch.squeeze({input_name})")

    def handle_Transpose(self, node, layer_idx):
        input_name = self.tensor_map[node.input[0]]
        output_name = sanitize_name(node.output[0])
        self.tensor_map[node.output[0]] = output_name
        perm = get_attribute(node, 'perm')
        self.forward_lines.append(f"{output_name} = {input_name}.permute({', '.join(map(str, perm))})")
    
    def handle_unknown(self, node, layer_idx):
        op_type = node.op_type
        print(f"Warning: Unsupported ONNX operator '{op_type}' at node {layer_idx}. Skipping.")
        self.forward_lines.append(f"# ---- Operator '{op_type}' (node {layer_idx}) is not supported yet ----")
        if node.input:
            input_name = self.tensor_map.get(node.input[0], "unknown_input")
            for i, out in enumerate(node.output):
                output_name = sanitize_name(out)
                self.tensor_map[out] = output_name
                self.forward_lines.append(f"{output_name} = {input_name} # Placeholder for unsupported op '{op_type}'")

def main():
    import os
    onnx_folder = r'C:\Users\csasd_rk5agwe\Desktop\idk thing\random numberz get tweaked omggg\ASL-Translation\mediapipe_wrapper\models\onnx'
    python_folder = r'C:\Users\csasd_rk5agwe\Desktop\idk thing\random numberz get tweaked omggg\ASL-Translation\mediapipe_wrapper\models_code'
    for p in os.listdir(onnx_folder):
        full_onnx_p = os.path.join(onnx_folder, p)
        full_python_p = os.path.join(python_folder, p.split('.')[0]+'.py')
        converter = OnnxToPytorch(full_onnx_p, full_python_p)
        converter.convert()

if __name__ == "__main__":
    main()