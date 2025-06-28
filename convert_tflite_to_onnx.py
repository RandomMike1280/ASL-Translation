from mediapipe_wrapper.enum import ModelEnum

import os
# import tensorflow as tf
import tf2onnx
import onnx

def convert_tflite_to_onnx(tflite_model_path, onnx_model_path, opset=13):
    """
    Converts a TensorFlow Lite (.tflite) model to ONNX format using tf2onnx.

    Args:
        tflite_model_path (str): Path to the input .tflite model file.
        onnx_model_path (str): Path where the output .onnx model will be saved.
        opset (int): The ONNX opset version to use for conversion.
                     Default is 13, which is a widely supported version.
                     You might need to adjust this based on your model's operations
                     and the ONNX Runtime version you plan to use.
    """
    if not os.path.exists(tflite_model_path):
        print(f"Error: TFLite model not found at '{tflite_model_path}'")
        return

    print(f"Attempting to convert '{tflite_model_path}' to ONNX...")
    print(f"Output ONNX path: '{onnx_model_path}'")
    print(f"Using ONNX opset version: {opset}")

    try:
        # tf2onnx.convert.from_tflite is the recommended way to convert tflite models
        # It internally handles loading the tflite model and converting its graph.
        # The 'output_path' argument directly saves the ONNX model.
        # We explicitly set the opset here.
        model_proto, _ = tf2onnx.convert.from_tflite(
            tflite_model_path,
            output_path=onnx_model_path,
            opset=opset
        )

        # You can optionally load and check the ONNX model after saving
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model) # Check if the ONNX model is valid

        print(f"\nSuccessfully converted TFLite model to ONNX: '{onnx_model_path}'")

    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("Please ensure your .tflite model is valid and its operations are supported by tf2onnx.")
        print("You might also try a different --opset value (e.g., 11, 14, or higher) if conversion fails.")
        print("If your model uses float16, try to convert the original TF model to float32 before TFLite conversion if possible.")

if __name__ == "__main__":
    for enum in ModelEnum:
        input_tflite_file = enum.value
        output_onnx_file = r"C:\Users\csasd_rk5agwe\Desktop\idk thing\random numberz get tweaked omggg\ASL-Translation\mediapipe_wrapper\models\\" + input_tflite_file.split('/')[-1].split('\\')[-1].split('.')[0] + '.onnx'

    input_tflite_file = r"C:\Users\csasd_rk5agwe\AppData\Local\Programs\Python\Python311\Lib\site-packages\mediapipe\modules\pose_detection\pose_detection.tflite"
    output_onnx_file = r"C:\Users\csasd_rk5agwe\Desktop\idk thing\random numberz get tweaked omggg\ASL-Translation\mediapipe_wrapper\models\pose_detection.onnx"

    # Define the ONNX opset version.
    # tf2onnx supports opset-14 to opset-18, and opset-6 to opset-13 should also work.
    # Default is often 15 or 13. If you encounter issues, try different values. [5, 6]
    target_opset = 15 # Or 14, 15, etc., depending on your needs and TF ops.

    # --- Run the conversion ---
    convert_tflite_to_onnx(input_tflite_file, output_onnx_file, target_opset)