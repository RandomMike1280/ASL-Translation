import torch
import torch.nn as nn
import torch.nn.functional as F

class palm_detection_full(nn.Module):
    def __init__(self):
        super(palm_detection_full, self).__init__()
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (1, 1), 'group': 1}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (1, 1), 'pads': (0, 0, 0, 0), 'group': 1}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 128, 'pads': (2, 2, 2, 2)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 256, 'pads': (2, 2, 2, 2)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 32, 'pads': (2, 2, 2, 2)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 64, 'pads': (2, 2, 2, 2)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 128, 'pads': (1, 1, 2, 2)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 256, 'pads': (1, 1, 2, 2)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 32, 'pads': (1, 1, 2, 2)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 64, 'pads': (1, 1, 2, 2)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'pads': (1, 1, 2, 2), 'group': 1}
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), groups=1, bias=True)
        self.conv_102 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=True)
        self.conv_103 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_107 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=32, bias=True)
        self.conv_110 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=True)
        self.conv_111 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_114 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=True)
        self.conv_115 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_118 = nn.Conv2d(in_channels=256, out_channels=108, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_121 = nn.Conv2d(in_channels=256, out_channels=6, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_125 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=True)
        self.conv_129 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_132 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=True)
        self.conv_133 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_136 = nn.Conv2d(in_channels=128, out_channels=36, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_139 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_15 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=32, bias=True)
        self.conv_16 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_21 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), groups=32, bias=True)
        self.conv_22 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_25 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=64, bias=True)
        self.conv_26 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_29 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=64, bias=True)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=32, bias=True)
        self.conv_30 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_33 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=64, bias=True)
        self.conv_34 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_37 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=64, bias=True)
        self.conv_38 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_43 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), groups=64, bias=True)
        self.conv_44 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_47 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=True)
        self.conv_48 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_51 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=True)
        self.conv_52 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_55 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=True)
        self.conv_56 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_59 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=True)
        self.conv_60 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_65 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), groups=128, bias=True)
        self.conv_66 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_69 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=True)
        self.conv_7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=32, bias=True)
        self.conv_70 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_73 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=True)
        self.conv_74 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_77 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=True)
        self.conv_78 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_81 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=True)
        self.conv_82 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_86 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), groups=256, bias=True)
        self.conv_87 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_90 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=True)
        self.conv_91 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_94 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=True)
        self.conv_95 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_98 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=True)
        self.conv_99 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.maxpool_19 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.maxpool_41 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.maxpool_63 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.maxpool_85 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.prelu_10 = nn.PReLU(num_parameters=1)
        self.prelu_101 = nn.PReLU(num_parameters=1)
        self.prelu_105 = nn.PReLU(num_parameters=1)
        self.prelu_108 = nn.PReLU(num_parameters=1)
        self.prelu_113 = nn.PReLU(num_parameters=1)
        self.prelu_117 = nn.PReLU(num_parameters=1)
        self.prelu_126 = nn.PReLU(num_parameters=1)
        self.prelu_131 = nn.PReLU(num_parameters=1)
        self.prelu_135 = nn.PReLU(num_parameters=1)
        self.prelu_14 = nn.PReLU(num_parameters=1)
        self.prelu_18 = nn.PReLU(num_parameters=1)
        self.prelu_2 = nn.PReLU(num_parameters=1)
        self.prelu_24 = nn.PReLU(num_parameters=1)
        self.prelu_28 = nn.PReLU(num_parameters=1)
        self.prelu_32 = nn.PReLU(num_parameters=1)
        self.prelu_36 = nn.PReLU(num_parameters=1)
        self.prelu_40 = nn.PReLU(num_parameters=1)
        self.prelu_46 = nn.PReLU(num_parameters=1)
        self.prelu_50 = nn.PReLU(num_parameters=1)
        self.prelu_54 = nn.PReLU(num_parameters=1)
        self.prelu_58 = nn.PReLU(num_parameters=1)
        self.prelu_6 = nn.PReLU(num_parameters=1)
        self.prelu_62 = nn.PReLU(num_parameters=1)
        self.prelu_68 = nn.PReLU(num_parameters=1)
        self.prelu_72 = nn.PReLU(num_parameters=1)
        self.prelu_76 = nn.PReLU(num_parameters=1)
        self.prelu_80 = nn.PReLU(num_parameters=1)
        self.prelu_84 = nn.PReLU(num_parameters=1)
        self.prelu_89 = nn.PReLU(num_parameters=1)
        self.prelu_93 = nn.PReLU(num_parameters=1)
        self.prelu_97 = nn.PReLU(num_parameters=1)

    def forward(self, input_1):
        model_1_model_batch_normalization_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_Conv2D1__3267_0 = input_1.permute(0, 3, 1, 2)
        model_1_model_batch_normalization_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_Conv2D1 = self.conv_1(model_1_model_batch_normalization_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_Conv2D1__3267_0)
        model_1_model_p_re_lu_add_model_1_model_p_re_lu_Relu_model_1_model_p_re_lu_Neg_1_model_1_model_p_re_lu_Relu_1_model_1_model_p_re_lu_mul1 = self.prelu_2(model_1_model_batch_normalization_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_Conv2D1)
        model_1_model_depthwise_conv2d_depthwise1 = self.conv_3(model_1_model_p_re_lu_add_model_1_model_p_re_lu_Relu_model_1_model_p_re_lu_Neg_1_model_1_model_p_re_lu_Relu_1_model_1_model_p_re_lu_mul1)
        model_1_model_batch_normalization_1_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_1_Conv2D1 = self.conv_4(model_1_model_depthwise_conv2d_depthwise1)
        model_1_model_add_add = model_1_model_p_re_lu_add_model_1_model_p_re_lu_Relu_model_1_model_p_re_lu_Neg_1_model_1_model_p_re_lu_Relu_1_model_1_model_p_re_lu_mul1 + model_1_model_batch_normalization_1_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_1_Conv2D1
        model_1_model_p_re_lu_1_add_model_1_model_p_re_lu_1_Relu_model_1_model_p_re_lu_1_Neg_1_model_1_model_p_re_lu_1_Relu_1_model_1_model_p_re_lu_1_mul1 = self.prelu_6(model_1_model_add_add)
        model_1_model_depthwise_conv2d_1_depthwise1 = self.conv_7(model_1_model_p_re_lu_1_add_model_1_model_p_re_lu_1_Relu_model_1_model_p_re_lu_1_Neg_1_model_1_model_p_re_lu_1_Relu_1_model_1_model_p_re_lu_1_mul1)
        model_1_model_batch_normalization_2_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_2_Conv2D1 = self.conv_8(model_1_model_depthwise_conv2d_1_depthwise1)
        model_1_model_add_1_add = model_1_model_p_re_lu_1_add_model_1_model_p_re_lu_1_Relu_model_1_model_p_re_lu_1_Neg_1_model_1_model_p_re_lu_1_Relu_1_model_1_model_p_re_lu_1_mul1 + model_1_model_batch_normalization_2_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_2_Conv2D1
        model_1_model_p_re_lu_2_add_model_1_model_p_re_lu_2_Relu_model_1_model_p_re_lu_2_Neg_1_model_1_model_p_re_lu_2_Relu_1_model_1_model_p_re_lu_2_mul1 = self.prelu_10(model_1_model_add_1_add)
        model_1_model_depthwise_conv2d_2_depthwise1 = self.conv_11(model_1_model_p_re_lu_2_add_model_1_model_p_re_lu_2_Relu_model_1_model_p_re_lu_2_Neg_1_model_1_model_p_re_lu_2_Relu_1_model_1_model_p_re_lu_2_mul1)
        model_1_model_batch_normalization_3_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_3_Conv2D1 = self.conv_12(model_1_model_depthwise_conv2d_2_depthwise1)
        model_1_model_add_2_add = model_1_model_p_re_lu_2_add_model_1_model_p_re_lu_2_Relu_model_1_model_p_re_lu_2_Neg_1_model_1_model_p_re_lu_2_Relu_1_model_1_model_p_re_lu_2_mul1 + model_1_model_batch_normalization_3_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_3_Conv2D1
        model_1_model_p_re_lu_3_add_model_1_model_p_re_lu_3_Relu_model_1_model_p_re_lu_3_Neg_1_model_1_model_p_re_lu_3_Relu_1_model_1_model_p_re_lu_3_mul1 = self.prelu_14(model_1_model_add_2_add)
        model_1_model_depthwise_conv2d_3_depthwise1 = self.conv_15(model_1_model_p_re_lu_3_add_model_1_model_p_re_lu_3_Relu_model_1_model_p_re_lu_3_Neg_1_model_1_model_p_re_lu_3_Relu_1_model_1_model_p_re_lu_3_mul1)
        model_1_model_batch_normalization_4_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_4_Conv2D1 = self.conv_16(model_1_model_depthwise_conv2d_3_depthwise1)
        model_1_model_add_3_add = model_1_model_p_re_lu_3_add_model_1_model_p_re_lu_3_Relu_model_1_model_p_re_lu_3_Neg_1_model_1_model_p_re_lu_3_Relu_1_model_1_model_p_re_lu_3_mul1 + model_1_model_batch_normalization_4_FusedBatchNormV3_model_1_model_depthwise_conv2d_4_depthwise_model_1_model_conv2d_4_Conv2D1
        model_1_model_p_re_lu_4_add_model_1_model_p_re_lu_4_Relu_model_1_model_p_re_lu_4_Neg_1_model_1_model_p_re_lu_4_Relu_1_model_1_model_p_re_lu_4_mul1 = self.prelu_18(model_1_model_add_3_add)
        model_1_model_max_pooling2d_MaxPool = self.maxpool_19(model_1_model_p_re_lu_4_add_model_1_model_p_re_lu_4_Relu_model_1_model_p_re_lu_4_Neg_1_model_1_model_p_re_lu_4_Relu_1_model_1_model_p_re_lu_4_mul1)
        model_1_model_channel_padding_Pad = F.pad(model_1_model_max_pooling2d_MaxPool, (0, 0, 0, 0), mode='constant', value=0.0)
        model_1_model_depthwise_conv2d_4_depthwise2 = self.conv_21(model_1_model_p_re_lu_4_add_model_1_model_p_re_lu_4_Relu_model_1_model_p_re_lu_4_Neg_1_model_1_model_p_re_lu_4_Relu_1_model_1_model_p_re_lu_4_mul1)
        model_1_model_batch_normalization_5_FusedBatchNormV3_model_1_model_depthwise_conv2d_9_depthwise_model_1_model_conv2d_5_Conv2D1 = self.conv_22(model_1_model_depthwise_conv2d_4_depthwise2)
        model_1_model_add_4_add = model_1_model_channel_padding_Pad + model_1_model_batch_normalization_5_FusedBatchNormV3_model_1_model_depthwise_conv2d_9_depthwise_model_1_model_conv2d_5_Conv2D1
        model_1_model_p_re_lu_5_add_model_1_model_p_re_lu_5_Relu_model_1_model_p_re_lu_5_Neg_1_model_1_model_p_re_lu_5_Relu_1_model_1_model_p_re_lu_5_mul1 = self.prelu_24(model_1_model_add_4_add)
        model_1_model_depthwise_conv2d_5_depthwise1 = self.conv_25(model_1_model_p_re_lu_5_add_model_1_model_p_re_lu_5_Relu_model_1_model_p_re_lu_5_Neg_1_model_1_model_p_re_lu_5_Relu_1_model_1_model_p_re_lu_5_mul1)
        model_1_model_batch_normalization_6_FusedBatchNormV3_model_1_model_depthwise_conv2d_9_depthwise_model_1_model_conv2d_6_Conv2D1 = self.conv_26(model_1_model_depthwise_conv2d_5_depthwise1)
        model_1_model_add_5_add = model_1_model_p_re_lu_5_add_model_1_model_p_re_lu_5_Relu_model_1_model_p_re_lu_5_Neg_1_model_1_model_p_re_lu_5_Relu_1_model_1_model_p_re_lu_5_mul1 + model_1_model_batch_normalization_6_FusedBatchNormV3_model_1_model_depthwise_conv2d_9_depthwise_model_1_model_conv2d_6_Conv2D1
        model_1_model_p_re_lu_6_add_model_1_model_p_re_lu_6_Relu_model_1_model_p_re_lu_6_Neg_1_model_1_model_p_re_lu_6_Relu_1_model_1_model_p_re_lu_6_mul1 = self.prelu_28(model_1_model_add_5_add)
        model_1_model_depthwise_conv2d_6_depthwise1 = self.conv_29(model_1_model_p_re_lu_6_add_model_1_model_p_re_lu_6_Relu_model_1_model_p_re_lu_6_Neg_1_model_1_model_p_re_lu_6_Relu_1_model_1_model_p_re_lu_6_mul1)
        model_1_model_batch_normalization_7_FusedBatchNormV3_model_1_model_depthwise_conv2d_9_depthwise_model_1_model_conv2d_7_Conv2D1 = self.conv_30(model_1_model_depthwise_conv2d_6_depthwise1)
        model_1_model_add_6_add = model_1_model_p_re_lu_6_add_model_1_model_p_re_lu_6_Relu_model_1_model_p_re_lu_6_Neg_1_model_1_model_p_re_lu_6_Relu_1_model_1_model_p_re_lu_6_mul1 + model_1_model_batch_normalization_7_FusedBatchNormV3_model_1_model_depthwise_conv2d_9_depthwise_model_1_model_conv2d_7_Conv2D1
        model_1_model_p_re_lu_7_add_model_1_model_p_re_lu_7_Relu_model_1_model_p_re_lu_7_Neg_1_model_1_model_p_re_lu_7_Relu_1_model_1_model_p_re_lu_7_mul1 = self.prelu_32(model_1_model_add_6_add)
        model_1_model_depthwise_conv2d_7_depthwise1 = self.conv_33(model_1_model_p_re_lu_7_add_model_1_model_p_re_lu_7_Relu_model_1_model_p_re_lu_7_Neg_1_model_1_model_p_re_lu_7_Relu_1_model_1_model_p_re_lu_7_mul1)
        model_1_model_batch_normalization_8_FusedBatchNormV3_model_1_model_depthwise_conv2d_9_depthwise_model_1_model_conv2d_8_Conv2D1 = self.conv_34(model_1_model_depthwise_conv2d_7_depthwise1)
        model_1_model_add_7_add = model_1_model_p_re_lu_7_add_model_1_model_p_re_lu_7_Relu_model_1_model_p_re_lu_7_Neg_1_model_1_model_p_re_lu_7_Relu_1_model_1_model_p_re_lu_7_mul1 + model_1_model_batch_normalization_8_FusedBatchNormV3_model_1_model_depthwise_conv2d_9_depthwise_model_1_model_conv2d_8_Conv2D1
        model_1_model_p_re_lu_8_add_model_1_model_p_re_lu_8_Relu_model_1_model_p_re_lu_8_Neg_1_model_1_model_p_re_lu_8_Relu_1_model_1_model_p_re_lu_8_mul1 = self.prelu_36(model_1_model_add_7_add)
        model_1_model_depthwise_conv2d_8_depthwise1 = self.conv_37(model_1_model_p_re_lu_8_add_model_1_model_p_re_lu_8_Relu_model_1_model_p_re_lu_8_Neg_1_model_1_model_p_re_lu_8_Relu_1_model_1_model_p_re_lu_8_mul1)
        model_1_model_batch_normalization_9_FusedBatchNormV3_model_1_model_depthwise_conv2d_9_depthwise_model_1_model_conv2d_9_Conv2D1 = self.conv_38(model_1_model_depthwise_conv2d_8_depthwise1)
        model_1_model_add_8_add = model_1_model_p_re_lu_8_add_model_1_model_p_re_lu_8_Relu_model_1_model_p_re_lu_8_Neg_1_model_1_model_p_re_lu_8_Relu_1_model_1_model_p_re_lu_8_mul1 + model_1_model_batch_normalization_9_FusedBatchNormV3_model_1_model_depthwise_conv2d_9_depthwise_model_1_model_conv2d_9_Conv2D1
        model_1_model_p_re_lu_9_add_model_1_model_p_re_lu_9_Relu_model_1_model_p_re_lu_9_Neg_1_model_1_model_p_re_lu_9_Relu_1_model_1_model_p_re_lu_9_mul1 = self.prelu_40(model_1_model_add_8_add)
        model_1_model_max_pooling2d_1_MaxPool = self.maxpool_41(model_1_model_p_re_lu_9_add_model_1_model_p_re_lu_9_Relu_model_1_model_p_re_lu_9_Neg_1_model_1_model_p_re_lu_9_Relu_1_model_1_model_p_re_lu_9_mul1)
        model_1_model_channel_padding_1_Pad = F.pad(model_1_model_max_pooling2d_1_MaxPool, (0, 0, 0, 0), mode='constant', value=0.0)
        model_1_model_depthwise_conv2d_9_depthwise2 = self.conv_43(model_1_model_p_re_lu_9_add_model_1_model_p_re_lu_9_Relu_model_1_model_p_re_lu_9_Neg_1_model_1_model_p_re_lu_9_Relu_1_model_1_model_p_re_lu_9_mul1)
        model_1_model_batch_normalization_10_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_10_Conv2D1 = self.conv_44(model_1_model_depthwise_conv2d_9_depthwise2)
        model_1_model_add_9_add = model_1_model_channel_padding_1_Pad + model_1_model_batch_normalization_10_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_10_Conv2D1
        model_1_model_p_re_lu_10_add_model_1_model_p_re_lu_10_Relu_model_1_model_p_re_lu_10_Neg_1_model_1_model_p_re_lu_10_Relu_1_model_1_model_p_re_lu_10_mul1 = self.prelu_46(model_1_model_add_9_add)
        model_1_model_depthwise_conv2d_10_depthwise1 = self.conv_47(model_1_model_p_re_lu_10_add_model_1_model_p_re_lu_10_Relu_model_1_model_p_re_lu_10_Neg_1_model_1_model_p_re_lu_10_Relu_1_model_1_model_p_re_lu_10_mul1)
        model_1_model_batch_normalization_11_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_11_Conv2D1 = self.conv_48(model_1_model_depthwise_conv2d_10_depthwise1)
        model_1_model_add_10_add = model_1_model_p_re_lu_10_add_model_1_model_p_re_lu_10_Relu_model_1_model_p_re_lu_10_Neg_1_model_1_model_p_re_lu_10_Relu_1_model_1_model_p_re_lu_10_mul1 + model_1_model_batch_normalization_11_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_11_Conv2D1
        model_1_model_p_re_lu_11_add_model_1_model_p_re_lu_11_Relu_model_1_model_p_re_lu_11_Neg_1_model_1_model_p_re_lu_11_Relu_1_model_1_model_p_re_lu_11_mul1 = self.prelu_50(model_1_model_add_10_add)
        model_1_model_depthwise_conv2d_11_depthwise1 = self.conv_51(model_1_model_p_re_lu_11_add_model_1_model_p_re_lu_11_Relu_model_1_model_p_re_lu_11_Neg_1_model_1_model_p_re_lu_11_Relu_1_model_1_model_p_re_lu_11_mul1)
        model_1_model_batch_normalization_12_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_12_Conv2D1 = self.conv_52(model_1_model_depthwise_conv2d_11_depthwise1)
        model_1_model_add_11_add = model_1_model_p_re_lu_11_add_model_1_model_p_re_lu_11_Relu_model_1_model_p_re_lu_11_Neg_1_model_1_model_p_re_lu_11_Relu_1_model_1_model_p_re_lu_11_mul1 + model_1_model_batch_normalization_12_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_12_Conv2D1
        model_1_model_p_re_lu_12_add_model_1_model_p_re_lu_12_Relu_model_1_model_p_re_lu_12_Neg_1_model_1_model_p_re_lu_12_Relu_1_model_1_model_p_re_lu_12_mul1 = self.prelu_54(model_1_model_add_11_add)
        model_1_model_depthwise_conv2d_12_depthwise1 = self.conv_55(model_1_model_p_re_lu_12_add_model_1_model_p_re_lu_12_Relu_model_1_model_p_re_lu_12_Neg_1_model_1_model_p_re_lu_12_Relu_1_model_1_model_p_re_lu_12_mul1)
        model_1_model_batch_normalization_13_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_13_Conv2D1 = self.conv_56(model_1_model_depthwise_conv2d_12_depthwise1)
        model_1_model_add_12_add = model_1_model_p_re_lu_12_add_model_1_model_p_re_lu_12_Relu_model_1_model_p_re_lu_12_Neg_1_model_1_model_p_re_lu_12_Relu_1_model_1_model_p_re_lu_12_mul1 + model_1_model_batch_normalization_13_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_13_Conv2D1
        model_1_model_p_re_lu_13_add_model_1_model_p_re_lu_13_Relu_model_1_model_p_re_lu_13_Neg_1_model_1_model_p_re_lu_13_Relu_1_model_1_model_p_re_lu_13_mul1 = self.prelu_58(model_1_model_add_12_add)
        model_1_model_depthwise_conv2d_13_depthwise1 = self.conv_59(model_1_model_p_re_lu_13_add_model_1_model_p_re_lu_13_Relu_model_1_model_p_re_lu_13_Neg_1_model_1_model_p_re_lu_13_Relu_1_model_1_model_p_re_lu_13_mul1)
        model_1_model_batch_normalization_14_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_14_Conv2D1 = self.conv_60(model_1_model_depthwise_conv2d_13_depthwise1)
        model_1_model_add_13_add = model_1_model_p_re_lu_13_add_model_1_model_p_re_lu_13_Relu_model_1_model_p_re_lu_13_Neg_1_model_1_model_p_re_lu_13_Relu_1_model_1_model_p_re_lu_13_mul1 + model_1_model_batch_normalization_14_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_14_Conv2D1
        model_1_model_p_re_lu_14_add_model_1_model_p_re_lu_14_Relu_model_1_model_p_re_lu_14_Neg_1_model_1_model_p_re_lu_14_Relu_1_model_1_model_p_re_lu_14_mul1 = self.prelu_62(model_1_model_add_13_add)
        model_1_model_max_pooling2d_2_MaxPool = self.maxpool_63(model_1_model_p_re_lu_14_add_model_1_model_p_re_lu_14_Relu_model_1_model_p_re_lu_14_Neg_1_model_1_model_p_re_lu_14_Relu_1_model_1_model_p_re_lu_14_mul1)
        model_1_model_channel_padding_2_Pad = F.pad(model_1_model_max_pooling2d_2_MaxPool, (0, 0, 0, 0), mode='constant', value=0.0)
        model_1_model_depthwise_conv2d_14_depthwise1 = self.conv_65(model_1_model_p_re_lu_14_add_model_1_model_p_re_lu_14_Relu_model_1_model_p_re_lu_14_Neg_1_model_1_model_p_re_lu_14_Relu_1_model_1_model_p_re_lu_14_mul1)
        model_1_model_batch_normalization_15_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_15_Conv2D1 = self.conv_66(model_1_model_depthwise_conv2d_14_depthwise1)
        model_1_model_add_14_add = model_1_model_channel_padding_2_Pad + model_1_model_batch_normalization_15_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_15_Conv2D1
        model_1_model_p_re_lu_15_add_model_1_model_p_re_lu_15_Relu_model_1_model_p_re_lu_15_Neg_1_model_1_model_p_re_lu_15_Relu_1_model_1_model_p_re_lu_15_mul1 = self.prelu_68(model_1_model_add_14_add)
        model_1_model_depthwise_conv2d_15_depthwise1 = self.conv_69(model_1_model_p_re_lu_15_add_model_1_model_p_re_lu_15_Relu_model_1_model_p_re_lu_15_Neg_1_model_1_model_p_re_lu_15_Relu_1_model_1_model_p_re_lu_15_mul1)
        model_1_model_batch_normalization_16_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_16_Conv2D1 = self.conv_70(model_1_model_depthwise_conv2d_15_depthwise1)
        model_1_model_add_15_add = model_1_model_p_re_lu_15_add_model_1_model_p_re_lu_15_Relu_model_1_model_p_re_lu_15_Neg_1_model_1_model_p_re_lu_15_Relu_1_model_1_model_p_re_lu_15_mul1 + model_1_model_batch_normalization_16_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_16_Conv2D1
        model_1_model_p_re_lu_16_add_model_1_model_p_re_lu_16_Relu_model_1_model_p_re_lu_16_Neg_1_model_1_model_p_re_lu_16_Relu_1_model_1_model_p_re_lu_16_mul1 = self.prelu_72(model_1_model_add_15_add)
        model_1_model_depthwise_conv2d_16_depthwise1 = self.conv_73(model_1_model_p_re_lu_16_add_model_1_model_p_re_lu_16_Relu_model_1_model_p_re_lu_16_Neg_1_model_1_model_p_re_lu_16_Relu_1_model_1_model_p_re_lu_16_mul1)
        model_1_model_batch_normalization_17_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_17_Conv2D1 = self.conv_74(model_1_model_depthwise_conv2d_16_depthwise1)
        model_1_model_add_16_add = model_1_model_p_re_lu_16_add_model_1_model_p_re_lu_16_Relu_model_1_model_p_re_lu_16_Neg_1_model_1_model_p_re_lu_16_Relu_1_model_1_model_p_re_lu_16_mul1 + model_1_model_batch_normalization_17_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_17_Conv2D1
        model_1_model_p_re_lu_17_add_model_1_model_p_re_lu_17_Relu_model_1_model_p_re_lu_17_Neg_1_model_1_model_p_re_lu_17_Relu_1_model_1_model_p_re_lu_17_mul1 = self.prelu_76(model_1_model_add_16_add)
        model_1_model_depthwise_conv2d_17_depthwise1 = self.conv_77(model_1_model_p_re_lu_17_add_model_1_model_p_re_lu_17_Relu_model_1_model_p_re_lu_17_Neg_1_model_1_model_p_re_lu_17_Relu_1_model_1_model_p_re_lu_17_mul1)
        model_1_model_batch_normalization_18_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_18_Conv2D1 = self.conv_78(model_1_model_depthwise_conv2d_17_depthwise1)
        model_1_model_add_17_add = model_1_model_p_re_lu_17_add_model_1_model_p_re_lu_17_Relu_model_1_model_p_re_lu_17_Neg_1_model_1_model_p_re_lu_17_Relu_1_model_1_model_p_re_lu_17_mul1 + model_1_model_batch_normalization_18_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_18_Conv2D1
        model_1_model_p_re_lu_18_add_model_1_model_p_re_lu_18_Relu_model_1_model_p_re_lu_18_Neg_1_model_1_model_p_re_lu_18_Relu_1_model_1_model_p_re_lu_18_mul1 = self.prelu_80(model_1_model_add_17_add)
        model_1_model_depthwise_conv2d_18_depthwise1 = self.conv_81(model_1_model_p_re_lu_18_add_model_1_model_p_re_lu_18_Relu_model_1_model_p_re_lu_18_Neg_1_model_1_model_p_re_lu_18_Relu_1_model_1_model_p_re_lu_18_mul1)
        model_1_model_batch_normalization_19_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_19_Conv2D1 = self.conv_82(model_1_model_depthwise_conv2d_18_depthwise1)
        model_1_model_add_18_add = model_1_model_p_re_lu_18_add_model_1_model_p_re_lu_18_Relu_model_1_model_p_re_lu_18_Neg_1_model_1_model_p_re_lu_18_Relu_1_model_1_model_p_re_lu_18_mul1 + model_1_model_batch_normalization_19_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_19_Conv2D1
        model_1_model_p_re_lu_19_add_model_1_model_p_re_lu_19_Relu_model_1_model_p_re_lu_19_Neg_1_model_1_model_p_re_lu_19_Relu_1_model_1_model_p_re_lu_19_mul1 = self.prelu_84(model_1_model_add_18_add)
        model_1_model_max_pooling2d_3_MaxPool = self.maxpool_85(model_1_model_p_re_lu_19_add_model_1_model_p_re_lu_19_Relu_model_1_model_p_re_lu_19_Neg_1_model_1_model_p_re_lu_19_Relu_1_model_1_model_p_re_lu_19_mul1)
        model_1_model_depthwise_conv2d_19_depthwise1 = self.conv_86(model_1_model_p_re_lu_19_add_model_1_model_p_re_lu_19_Relu_model_1_model_p_re_lu_19_Neg_1_model_1_model_p_re_lu_19_Relu_1_model_1_model_p_re_lu_19_mul1)
        model_1_model_batch_normalization_20_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_20_Conv2D1 = self.conv_87(model_1_model_depthwise_conv2d_19_depthwise1)
        model_1_model_add_19_add = model_1_model_max_pooling2d_3_MaxPool + model_1_model_batch_normalization_20_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_20_Conv2D1
        model_1_model_p_re_lu_20_add_model_1_model_p_re_lu_20_Relu_model_1_model_p_re_lu_20_Neg_1_model_1_model_p_re_lu_20_Relu_1_model_1_model_p_re_lu_20_mul1 = self.prelu_89(model_1_model_add_19_add)
        model_1_model_depthwise_conv2d_20_depthwise1 = self.conv_90(model_1_model_p_re_lu_20_add_model_1_model_p_re_lu_20_Relu_model_1_model_p_re_lu_20_Neg_1_model_1_model_p_re_lu_20_Relu_1_model_1_model_p_re_lu_20_mul1)
        model_1_model_batch_normalization_21_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_21_Conv2D1 = self.conv_91(model_1_model_depthwise_conv2d_20_depthwise1)
        model_1_model_add_20_add = model_1_model_p_re_lu_20_add_model_1_model_p_re_lu_20_Relu_model_1_model_p_re_lu_20_Neg_1_model_1_model_p_re_lu_20_Relu_1_model_1_model_p_re_lu_20_mul1 + model_1_model_batch_normalization_21_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_21_Conv2D1
        model_1_model_p_re_lu_21_add_model_1_model_p_re_lu_21_Relu_model_1_model_p_re_lu_21_Neg_1_model_1_model_p_re_lu_21_Relu_1_model_1_model_p_re_lu_21_mul1 = self.prelu_93(model_1_model_add_20_add)
        model_1_model_depthwise_conv2d_21_depthwise1 = self.conv_94(model_1_model_p_re_lu_21_add_model_1_model_p_re_lu_21_Relu_model_1_model_p_re_lu_21_Neg_1_model_1_model_p_re_lu_21_Relu_1_model_1_model_p_re_lu_21_mul1)
        model_1_model_batch_normalization_22_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_22_Conv2D1 = self.conv_95(model_1_model_depthwise_conv2d_21_depthwise1)
        model_1_model_add_21_add = model_1_model_p_re_lu_21_add_model_1_model_p_re_lu_21_Relu_model_1_model_p_re_lu_21_Neg_1_model_1_model_p_re_lu_21_Relu_1_model_1_model_p_re_lu_21_mul1 + model_1_model_batch_normalization_22_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_22_Conv2D1
        model_1_model_p_re_lu_22_add_model_1_model_p_re_lu_22_Relu_model_1_model_p_re_lu_22_Neg_1_model_1_model_p_re_lu_22_Relu_1_model_1_model_p_re_lu_22_mul1 = self.prelu_97(model_1_model_add_21_add)
        model_1_model_depthwise_conv2d_22_depthwise1 = self.conv_98(model_1_model_p_re_lu_22_add_model_1_model_p_re_lu_22_Relu_model_1_model_p_re_lu_22_Neg_1_model_1_model_p_re_lu_22_Relu_1_model_1_model_p_re_lu_22_mul1)
        model_1_model_batch_normalization_23_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_23_Conv2D1 = self.conv_99(model_1_model_depthwise_conv2d_22_depthwise1)
        model_1_model_add_22_add = model_1_model_p_re_lu_22_add_model_1_model_p_re_lu_22_Relu_model_1_model_p_re_lu_22_Neg_1_model_1_model_p_re_lu_22_Relu_1_model_1_model_p_re_lu_22_mul1 + model_1_model_batch_normalization_23_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_23_Conv2D1
        model_1_model_p_re_lu_23_add_model_1_model_p_re_lu_23_Relu_model_1_model_p_re_lu_23_Neg_1_model_1_model_p_re_lu_23_Relu_1_model_1_model_p_re_lu_23_mul1 = self.prelu_101(model_1_model_add_22_add)
        model_1_model_depthwise_conv2d_23_depthwise1 = self.conv_102(model_1_model_p_re_lu_23_add_model_1_model_p_re_lu_23_Relu_model_1_model_p_re_lu_23_Neg_1_model_1_model_p_re_lu_23_Relu_1_model_1_model_p_re_lu_23_mul1)
        model_1_model_batch_normalization_24_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_24_Conv2D1 = self.conv_103(model_1_model_depthwise_conv2d_23_depthwise1)
        model_1_model_add_23_add = model_1_model_p_re_lu_23_add_model_1_model_p_re_lu_23_Relu_model_1_model_p_re_lu_23_Neg_1_model_1_model_p_re_lu_23_Relu_1_model_1_model_p_re_lu_23_mul1 + model_1_model_batch_normalization_24_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_24_Conv2D1
        model_1_model_p_re_lu_24_add_model_1_model_p_re_lu_24_Relu_model_1_model_p_re_lu_24_Neg_1_model_1_model_p_re_lu_24_Relu_1_model_1_model_p_re_lu_24_mul1 = self.prelu_105(model_1_model_add_23_add)
        Resize__3476_0 = F.interpolate(model_1_model_p_re_lu_24_add_model_1_model_p_re_lu_24_Relu_model_1_model_p_re_lu_24_Neg_1_model_1_model_p_re_lu_24_Relu_1_model_1_model_p_re_lu_24_mul1, scale_factor=(2.0, 2.0), mode='bilinear', align_corners=None)
        model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_conv2d_25_BiasAdd_ReadVariableOp_resource_model_1_model_conv2d_25_BiasAdd_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_25_Conv2D = self.conv_107(Resize__3476_0)
        model_1_model_p_re_lu_25_add_model_1_model_p_re_lu_25_Relu_model_1_model_p_re_lu_25_Neg_1_model_1_model_p_re_lu_25_Relu_1_model_1_model_p_re_lu_25_mul1 = self.prelu_108(model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_conv2d_25_BiasAdd_ReadVariableOp_resource_model_1_model_conv2d_25_BiasAdd_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_25_Conv2D)
        model_1_model_add_24_add = model_1_model_p_re_lu_19_add_model_1_model_p_re_lu_19_Relu_model_1_model_p_re_lu_19_Neg_1_model_1_model_p_re_lu_19_Relu_1_model_1_model_p_re_lu_19_mul1 + model_1_model_p_re_lu_25_add_model_1_model_p_re_lu_25_Relu_model_1_model_p_re_lu_25_Neg_1_model_1_model_p_re_lu_25_Relu_1_model_1_model_p_re_lu_25_mul1
        model_1_model_depthwise_conv2d_24_depthwise1 = self.conv_110(model_1_model_add_24_add)
        model_1_model_batch_normalization_26_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_26_Conv2D1 = self.conv_111(model_1_model_depthwise_conv2d_24_depthwise1)
        model_1_model_add_25_add = model_1_model_add_24_add + model_1_model_batch_normalization_26_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D_model_1_model_conv2d_26_Conv2D1
        model_1_model_p_re_lu_26_add_model_1_model_p_re_lu_26_Relu_model_1_model_p_re_lu_26_Neg_1_model_1_model_p_re_lu_26_Relu_1_model_1_model_p_re_lu_26_mul1 = self.prelu_113(model_1_model_add_25_add)
        model_1_model_depthwise_conv2d_25_depthwise1 = self.conv_114(model_1_model_p_re_lu_26_add_model_1_model_p_re_lu_26_Relu_model_1_model_p_re_lu_26_Neg_1_model_1_model_p_re_lu_26_Relu_1_model_1_model_p_re_lu_26_mul1)
        model_1_model_batch_normalization_27_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D1 = self.conv_115(model_1_model_depthwise_conv2d_25_depthwise1)
        model_1_model_add_26_add = model_1_model_p_re_lu_26_add_model_1_model_p_re_lu_26_Relu_model_1_model_p_re_lu_26_Neg_1_model_1_model_p_re_lu_26_Relu_1_model_1_model_p_re_lu_26_mul1 + model_1_model_batch_normalization_27_FusedBatchNormV3_model_1_model_conv2d_27_Conv2D1
        model_1_model_p_re_lu_27_add_model_1_model_p_re_lu_27_Relu_model_1_model_p_re_lu_27_Neg_1_model_1_model_p_re_lu_27_Relu_1_model_1_model_p_re_lu_27_mul1 = self.prelu_117(model_1_model_add_26_add)
        model_1_model_regressor_palm_16_NO_PRUNING_BiasAdd_model_1_model_regressor_palm_16_NO_PRUNING_Conv2D_model_1_model_regressor_palm_16_NO_PRUNING_BiasAdd_ReadVariableOp_resource1 = self.conv_118(model_1_model_p_re_lu_27_add_model_1_model_p_re_lu_27_Relu_model_1_model_p_re_lu_27_Neg_1_model_1_model_p_re_lu_27_Relu_1_model_1_model_p_re_lu_27_mul1)
        model_1_model_regressor_palm_16_NO_PRUNING_BiasAdd_model_1_model_regressor_palm_16_NO_PRUNING_Conv2D_model_1_model_regressor_palm_16_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3505_0 = model_1_model_regressor_palm_16_NO_PRUNING_BiasAdd_model_1_model_regressor_palm_16_NO_PRUNING_Conv2D_model_1_model_regressor_palm_16_NO_PRUNING_BiasAdd_ReadVariableOp_resource1.permute(0, 2, 3, 1)
        model_1_model_reshaped_regressor_palm_16_Reshape = model_1_model_regressor_palm_16_NO_PRUNING_BiasAdd_model_1_model_regressor_palm_16_NO_PRUNING_Conv2D_model_1_model_regressor_palm_16_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3505_0.reshape(model_1_model_regressor_palm_16_NO_PRUNING_BiasAdd_model_1_model_regressor_palm_16_NO_PRUNING_Conv2D_model_1_model_regressor_palm_16_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3505_0.size(0), -1, 18)
        model_1_model_classifier_palm_16_NO_PRUNING_BiasAdd_model_1_model_classifier_palm_16_NO_PRUNING_Conv2D_model_1_model_classifier_palm_16_NO_PRUNING_BiasAdd_ReadVariableOp_resource1 = self.conv_121(model_1_model_p_re_lu_27_add_model_1_model_p_re_lu_27_Relu_model_1_model_p_re_lu_27_Neg_1_model_1_model_p_re_lu_27_Relu_1_model_1_model_p_re_lu_27_mul1)
        model_1_model_classifier_palm_16_NO_PRUNING_BiasAdd_model_1_model_classifier_palm_16_NO_PRUNING_Conv2D_model_1_model_classifier_palm_16_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3509_0 = model_1_model_classifier_palm_16_NO_PRUNING_BiasAdd_model_1_model_classifier_palm_16_NO_PRUNING_Conv2D_model_1_model_classifier_palm_16_NO_PRUNING_BiasAdd_ReadVariableOp_resource1.permute(0, 2, 3, 1)
        model_1_model_reshaped_classifier_palm_16_Reshape = model_1_model_classifier_palm_16_NO_PRUNING_BiasAdd_model_1_model_classifier_palm_16_NO_PRUNING_Conv2D_model_1_model_classifier_palm_16_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3509_0.reshape(model_1_model_classifier_palm_16_NO_PRUNING_BiasAdd_model_1_model_classifier_palm_16_NO_PRUNING_Conv2D_model_1_model_classifier_palm_16_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3509_0.size(0), -1, 1)
        Resize__3499_0 = F.interpolate(model_1_model_p_re_lu_27_add_model_1_model_p_re_lu_27_Relu_model_1_model_p_re_lu_27_Neg_1_model_1_model_p_re_lu_27_Relu_1_model_1_model_p_re_lu_27_mul1, scale_factor=(2.0, 2.0), mode='bilinear', align_corners=None)
        model_1_model_batch_normalization_28_FusedBatchNormV3_model_1_model_conv2d_28_BiasAdd_ReadVariableOp_resource_model_1_model_conv2d_28_BiasAdd_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_28_Conv2D = self.conv_125(Resize__3499_0)
        model_1_model_p_re_lu_28_add_model_1_model_p_re_lu_28_Relu_model_1_model_p_re_lu_28_Neg_1_model_1_model_p_re_lu_28_Relu_1_model_1_model_p_re_lu_28_mul1 = self.prelu_126(model_1_model_batch_normalization_28_FusedBatchNormV3_model_1_model_conv2d_28_BiasAdd_ReadVariableOp_resource_model_1_model_conv2d_28_BiasAdd_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_28_Conv2D)
        model_1_model_add_27_add = model_1_model_p_re_lu_14_add_model_1_model_p_re_lu_14_Relu_model_1_model_p_re_lu_14_Neg_1_model_1_model_p_re_lu_14_Relu_1_model_1_model_p_re_lu_14_mul1 + model_1_model_p_re_lu_28_add_model_1_model_p_re_lu_28_Relu_model_1_model_p_re_lu_28_Neg_1_model_1_model_p_re_lu_28_Relu_1_model_1_model_p_re_lu_28_mul1
        model_1_model_depthwise_conv2d_26_depthwise1 = self.conv_128(model_1_model_add_27_add)
        model_1_model_batch_normalization_29_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_29_Conv2D1 = self.conv_129(model_1_model_depthwise_conv2d_26_depthwise1)
        model_1_model_add_28_add = model_1_model_add_27_add + model_1_model_batch_normalization_29_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D_model_1_model_conv2d_29_Conv2D1
        model_1_model_p_re_lu_29_add_model_1_model_p_re_lu_29_Relu_model_1_model_p_re_lu_29_Neg_1_model_1_model_p_re_lu_29_Relu_1_model_1_model_p_re_lu_29_mul1 = self.prelu_131(model_1_model_add_28_add)
        model_1_model_depthwise_conv2d_27_depthwise1 = self.conv_132(model_1_model_p_re_lu_29_add_model_1_model_p_re_lu_29_Relu_model_1_model_p_re_lu_29_Neg_1_model_1_model_p_re_lu_29_Relu_1_model_1_model_p_re_lu_29_mul1)
        model_1_model_batch_normalization_30_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D1 = self.conv_133(model_1_model_depthwise_conv2d_27_depthwise1)
        model_1_model_add_29_add = model_1_model_p_re_lu_29_add_model_1_model_p_re_lu_29_Relu_model_1_model_p_re_lu_29_Neg_1_model_1_model_p_re_lu_29_Relu_1_model_1_model_p_re_lu_29_mul1 + model_1_model_batch_normalization_30_FusedBatchNormV3_model_1_model_conv2d_30_Conv2D1
        model_1_model_p_re_lu_30_add_model_1_model_p_re_lu_30_Relu_model_1_model_p_re_lu_30_Neg_1_model_1_model_p_re_lu_30_Relu_1_model_1_model_p_re_lu_30_mul1 = self.prelu_135(model_1_model_add_29_add)
        model_1_model_regressor_palm_8_NO_PRUNING_BiasAdd_model_1_model_regressor_palm_8_NO_PRUNING_Conv2D_model_1_model_regressor_palm_8_NO_PRUNING_BiasAdd_ReadVariableOp_resource1 = self.conv_136(model_1_model_p_re_lu_30_add_model_1_model_p_re_lu_30_Relu_model_1_model_p_re_lu_30_Neg_1_model_1_model_p_re_lu_30_Relu_1_model_1_model_p_re_lu_30_mul1)
        model_1_model_regressor_palm_8_NO_PRUNING_BiasAdd_model_1_model_regressor_palm_8_NO_PRUNING_Conv2D_model_1_model_regressor_palm_8_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3529_0 = model_1_model_regressor_palm_8_NO_PRUNING_BiasAdd_model_1_model_regressor_palm_8_NO_PRUNING_Conv2D_model_1_model_regressor_palm_8_NO_PRUNING_BiasAdd_ReadVariableOp_resource1.permute(0, 2, 3, 1)
        model_1_model_reshaped_regressor_palm_8_Reshape = model_1_model_regressor_palm_8_NO_PRUNING_BiasAdd_model_1_model_regressor_palm_8_NO_PRUNING_Conv2D_model_1_model_regressor_palm_8_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3529_0.reshape(model_1_model_regressor_palm_8_NO_PRUNING_BiasAdd_model_1_model_regressor_palm_8_NO_PRUNING_Conv2D_model_1_model_regressor_palm_8_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3529_0.size(0), -1, 18)
        model_1_model_classifier_palm_8_NO_PRUNING_BiasAdd_model_1_model_classifier_palm_8_NO_PRUNING_Conv2D_model_1_model_classifier_palm_8_NO_PRUNING_BiasAdd_ReadVariableOp_resource1 = self.conv_139(model_1_model_p_re_lu_30_add_model_1_model_p_re_lu_30_Relu_model_1_model_p_re_lu_30_Neg_1_model_1_model_p_re_lu_30_Relu_1_model_1_model_p_re_lu_30_mul1)
        model_1_model_classifier_palm_8_NO_PRUNING_BiasAdd_model_1_model_classifier_palm_8_NO_PRUNING_Conv2D_model_1_model_classifier_palm_8_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3533_0 = model_1_model_classifier_palm_8_NO_PRUNING_BiasAdd_model_1_model_classifier_palm_8_NO_PRUNING_Conv2D_model_1_model_classifier_palm_8_NO_PRUNING_BiasAdd_ReadVariableOp_resource1.permute(0, 2, 3, 1)
        model_1_model_reshaped_classifier_palm_8_Reshape = model_1_model_classifier_palm_8_NO_PRUNING_BiasAdd_model_1_model_classifier_palm_8_NO_PRUNING_Conv2D_model_1_model_classifier_palm_8_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3533_0.reshape(model_1_model_classifier_palm_8_NO_PRUNING_BiasAdd_model_1_model_classifier_palm_8_NO_PRUNING_Conv2D_model_1_model_classifier_palm_8_NO_PRUNING_BiasAdd_ReadVariableOp_resource1__3533_0.size(0), -1, 1)
        Identity_1 = torch.cat([model_1_model_reshaped_classifier_palm_8_Reshape, model_1_model_reshaped_classifier_palm_16_Reshape], dim=1)
        Identity = torch.cat([model_1_model_reshaped_regressor_palm_8_Reshape, model_1_model_reshaped_regressor_palm_16_Reshape], dim=1)
        return Identity, Identity_1

# How to load the model and weights:
# 1. Create an instance of the model:
#    model = palm_detection_full()
# 2. Load the state dictionary:
#    state_dict_path = 'palm_detection_full.pth'
#    model.load_state_dict(torch.load(state_dict_path))
# 3. Set the model to evaluation mode:
#    model.eval()
