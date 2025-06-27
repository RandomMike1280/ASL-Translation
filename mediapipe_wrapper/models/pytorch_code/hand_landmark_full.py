import torch
import torch.nn as nn
import torch.nn.functional as F

class hand_landmark_full(nn.Module):
    def __init__(self):
        super(hand_landmark_full, self).__init__()
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (1, 1), 'pads': (0, 0, 0, 0), 'group': 1}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 1152, 'pads': (1, 1, 1, 1)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 144, 'pads': (1, 1, 1, 1)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 24, 'pads': (1, 1, 1, 1)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 480, 'pads': (1, 1, 1, 1)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 1152, 'pads': (2, 2, 2, 2)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 240, 'pads': (2, 2, 2, 2)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 480, 'pads': (2, 2, 2, 2)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 672, 'pads': (2, 2, 2, 2)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 240, 'pads': (0, 0, 1, 1)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 64, 'pads': (0, 0, 1, 1)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'pads': (0, 0, 1, 1), 'group': 1}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 144, 'pads': (1, 1, 2, 2)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (5, 5), 'group': 672, 'pads': (1, 1, 2, 2)}
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), groups=1, bias=True)
        self.conv_10 = nn.Conv2d(in_channels=64, out_channels=24, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_11 = nn.Conv2d(in_channels=24, out_channels=144, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_13 = nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=True)
        self.conv_15 = nn.Conv2d(in_channels=144, out_channels=24, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_17 = nn.Conv2d(in_channels=24, out_channels=144, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_19 = nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), groups=144, bias=True)
        self.conv_21 = nn.Conv2d(in_channels=144, out_channels=40, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_22 = nn.Conv2d(in_channels=40, out_channels=240, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_24 = nn.Conv2d(in_channels=240, out_channels=240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=True)
        self.conv_26 = nn.Conv2d(in_channels=240, out_channels=40, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_28 = nn.Conv2d(in_channels=40, out_channels=240, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=True)
        self.conv_30 = nn.Conv2d(in_channels=240, out_channels=240, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), groups=240, bias=True)
        self.conv_32 = nn.Conv2d(in_channels=240, out_channels=80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_33 = nn.Conv2d(in_channels=80, out_channels=480, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_35 = nn.Conv2d(in_channels=480, out_channels=480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=True)
        self.conv_37 = nn.Conv2d(in_channels=480, out_channels=80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_39 = nn.Conv2d(in_channels=80, out_channels=480, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_41 = nn.Conv2d(in_channels=480, out_channels=480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=True)
        self.conv_43 = nn.Conv2d(in_channels=480, out_channels=80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_45 = nn.Conv2d(in_channels=80, out_channels=480, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_47 = nn.Conv2d(in_channels=480, out_channels=480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=True)
        self.conv_49 = nn.Conv2d(in_channels=480, out_channels=112, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_5 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_50 = nn.Conv2d(in_channels=112, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_52 = nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=True)
        self.conv_54 = nn.Conv2d(in_channels=672, out_channels=112, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_56 = nn.Conv2d(in_channels=112, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_58 = nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=True)
        self.conv_6 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_60 = nn.Conv2d(in_channels=672, out_channels=112, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_62 = nn.Conv2d(in_channels=112, out_channels=672, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_64 = nn.Conv2d(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), groups=672, bias=True)
        self.conv_66 = nn.Conv2d(in_channels=672, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_67 = nn.Conv2d(in_channels=192, out_channels=1152, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_69 = nn.Conv2d(in_channels=1152, out_channels=1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=True)
        self.conv_71 = nn.Conv2d(in_channels=1152, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_73 = nn.Conv2d(in_channels=192, out_channels=1152, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_75 = nn.Conv2d(in_channels=1152, out_channels=1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=True)
        self.conv_77 = nn.Conv2d(in_channels=1152, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_79 = nn.Conv2d(in_channels=192, out_channels=1152, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), groups=64, bias=True)
        self.conv_81 = nn.Conv2d(in_channels=1152, out_channels=1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=True)
        self.conv_83 = nn.Conv2d(in_channels=1152, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_85 = nn.Conv2d(in_channels=192, out_channels=1152, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_87 = nn.Conv2d(in_channels=1152, out_channels=1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=True)
        self.gap_89 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_91 = nn.Linear(in_features=1, out_features=1152, bias=True)
        self.linear_92 = nn.Linear(in_features=1, out_features=1152, bias=True)
        self.linear_93 = nn.Linear(in_features=63, out_features=1152, bias=True)
        self.linear_94 = nn.Linear(in_features=63, out_features=1152, bias=True)
        self.sigmoid_95 = nn.Sigmoid()
        self.sigmoid_96 = nn.Sigmoid()

    def forward(self, input_1):
        model_1_model_re_lu_Relu6_model_1_model_batch_normalization_FusedBatchNormV3_model_1_model_batch_normalization_1_FusedBatchNormV3_model_1_model_depthwise_conv2d_depthwise_model_1_model_conv2d_5_Conv2D_model_1_model_conv2d_Conv2D__2115_0 = input_1.permute(0, 3, 1, 2)
        model_1_model_re_lu_Relu6_model_1_model_batch_normalization_FusedBatchNormV3_model_1_model_batch_normalization_1_FusedBatchNormV3_model_1_model_depthwise_conv2d_depthwise_model_1_model_conv2d_5_Conv2D_model_1_model_conv2d_Conv2D = self.conv_1(model_1_model_re_lu_Relu6_model_1_model_batch_normalization_FusedBatchNormV3_model_1_model_batch_normalization_1_FusedBatchNormV3_model_1_model_depthwise_conv2d_depthwise_model_1_model_conv2d_5_Conv2D_model_1_model_conv2d_Conv2D__2115_0)
        Relu6__2024_0 = torch.clamp(model_1_model_re_lu_Relu6_model_1_model_batch_normalization_FusedBatchNormV3_model_1_model_batch_normalization_1_FusedBatchNormV3_model_1_model_depthwise_conv2d_depthwise_model_1_model_conv2d_5_Conv2D_model_1_model_conv2d_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_1_Relu6_model_1_model_batch_normalization_1_FusedBatchNormV3_model_1_model_depthwise_conv2d_depthwise_model_1_model_conv2d_5_Conv2D = self.conv_3(Relu6__2024_0)
        Relu6__2026_0 = torch.clamp(model_1_model_re_lu_1_Relu6_model_1_model_batch_normalization_1_FusedBatchNormV3_model_1_model_depthwise_conv2d_depthwise_model_1_model_conv2d_5_Conv2D, min=0.0, max=6.0)
        model_1_model_batch_normalization_2_FusedBatchNormV3_model_1_model_conv2d_1_Conv2D1 = self.conv_5(Relu6__2026_0)
        model_1_model_re_lu_2_Relu6_model_1_model_batch_normalization_3_FusedBatchNormV3_model_1_model_batch_normalization_4_FusedBatchNormV3_model_1_model_depthwise_conv2d_1_depthwise_model_1_model_conv2d_2_Conv2D = self.conv_6(model_1_model_batch_normalization_2_FusedBatchNormV3_model_1_model_conv2d_1_Conv2D1)
        Relu6__2029_0 = torch.clamp(model_1_model_re_lu_2_Relu6_model_1_model_batch_normalization_3_FusedBatchNormV3_model_1_model_batch_normalization_4_FusedBatchNormV3_model_1_model_depthwise_conv2d_1_depthwise_model_1_model_conv2d_2_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_3_Relu6_model_1_model_batch_normalization_4_FusedBatchNormV3_model_1_model_depthwise_conv2d_1_depthwise = self.conv_8(Relu6__2029_0)
        Relu6__2031_0 = torch.clamp(model_1_model_re_lu_3_Relu6_model_1_model_batch_normalization_4_FusedBatchNormV3_model_1_model_depthwise_conv2d_1_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_5_FusedBatchNormV3_model_1_model_conv2d_5_Conv2D_model_1_model_conv2d_3_Conv2D1 = self.conv_10(Relu6__2031_0)
        model_1_model_re_lu_4_Relu6_model_1_model_batch_normalization_6_FusedBatchNormV3_model_1_model_batch_normalization_10_FusedBatchNormV3_model_1_model_depthwise_conv2d_3_depthwise_model_1_model_conv2d_4_Conv2D = self.conv_11(model_1_model_batch_normalization_5_FusedBatchNormV3_model_1_model_conv2d_5_Conv2D_model_1_model_conv2d_3_Conv2D1)
        Relu6__2034_0 = torch.clamp(model_1_model_re_lu_4_Relu6_model_1_model_batch_normalization_6_FusedBatchNormV3_model_1_model_batch_normalization_10_FusedBatchNormV3_model_1_model_depthwise_conv2d_3_depthwise_model_1_model_conv2d_4_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_5_Relu6_model_1_model_batch_normalization_7_FusedBatchNormV3_model_1_model_batch_normalization_10_FusedBatchNormV3_model_1_model_depthwise_conv2d_3_depthwise_model_1_model_depthwise_conv2d_2_depthwise = self.conv_13(Relu6__2034_0)
        Relu6__2036_0 = torch.clamp(model_1_model_re_lu_5_Relu6_model_1_model_batch_normalization_7_FusedBatchNormV3_model_1_model_batch_normalization_10_FusedBatchNormV3_model_1_model_depthwise_conv2d_3_depthwise_model_1_model_depthwise_conv2d_2_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_8_FusedBatchNormV3_model_1_model_conv2d_5_Conv2D1 = self.conv_15(Relu6__2036_0)
        model_1_model_add_add = model_1_model_batch_normalization_8_FusedBatchNormV3_model_1_model_conv2d_5_Conv2D1 + model_1_model_batch_normalization_5_FusedBatchNormV3_model_1_model_conv2d_5_Conv2D_model_1_model_conv2d_3_Conv2D1
        model_1_model_re_lu_6_Relu6_model_1_model_batch_normalization_9_FusedBatchNormV3_model_1_model_batch_normalization_10_FusedBatchNormV3_model_1_model_depthwise_conv2d_3_depthwise_model_1_model_conv2d_6_Conv2D = self.conv_17(model_1_model_add_add)
        Relu6__2039_0 = torch.clamp(model_1_model_re_lu_6_Relu6_model_1_model_batch_normalization_9_FusedBatchNormV3_model_1_model_batch_normalization_10_FusedBatchNormV3_model_1_model_depthwise_conv2d_3_depthwise_model_1_model_conv2d_6_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_7_Relu6_model_1_model_batch_normalization_10_FusedBatchNormV3_model_1_model_depthwise_conv2d_3_depthwise = self.conv_19(Relu6__2039_0)
        Relu6__2041_0 = torch.clamp(model_1_model_re_lu_7_Relu6_model_1_model_batch_normalization_10_FusedBatchNormV3_model_1_model_depthwise_conv2d_3_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_11_FusedBatchNormV3_model_1_model_conv2d_9_Conv2D_model_1_model_conv2d_7_Conv2D1 = self.conv_21(Relu6__2041_0)
        model_1_model_re_lu_8_Relu6_model_1_model_batch_normalization_12_FusedBatchNormV3_model_1_model_batch_normalization_16_FusedBatchNormV3_model_1_model_depthwise_conv2d_5_depthwise_model_1_model_conv2d_8_Conv2D = self.conv_22(model_1_model_batch_normalization_11_FusedBatchNormV3_model_1_model_conv2d_9_Conv2D_model_1_model_conv2d_7_Conv2D1)
        Relu6__2044_0 = torch.clamp(model_1_model_re_lu_8_Relu6_model_1_model_batch_normalization_12_FusedBatchNormV3_model_1_model_batch_normalization_16_FusedBatchNormV3_model_1_model_depthwise_conv2d_5_depthwise_model_1_model_conv2d_8_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_9_Relu6_model_1_model_batch_normalization_13_FusedBatchNormV3_model_1_model_batch_normalization_16_FusedBatchNormV3_model_1_model_depthwise_conv2d_5_depthwise_model_1_model_depthwise_conv2d_4_depthwise = self.conv_24(Relu6__2044_0)
        Relu6__2046_0 = torch.clamp(model_1_model_re_lu_9_Relu6_model_1_model_batch_normalization_13_FusedBatchNormV3_model_1_model_batch_normalization_16_FusedBatchNormV3_model_1_model_depthwise_conv2d_5_depthwise_model_1_model_depthwise_conv2d_4_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_14_FusedBatchNormV3_model_1_model_conv2d_9_Conv2D1 = self.conv_26(Relu6__2046_0)
        model_1_model_add_1_add = model_1_model_batch_normalization_14_FusedBatchNormV3_model_1_model_conv2d_9_Conv2D1 + model_1_model_batch_normalization_11_FusedBatchNormV3_model_1_model_conv2d_9_Conv2D_model_1_model_conv2d_7_Conv2D1
        model_1_model_re_lu_10_Relu6_model_1_model_batch_normalization_15_FusedBatchNormV3_model_1_model_batch_normalization_16_FusedBatchNormV3_model_1_model_depthwise_conv2d_5_depthwise_model_1_model_conv2d_10_Conv2D = self.conv_28(model_1_model_add_1_add)
        Relu6__2049_0 = torch.clamp(model_1_model_re_lu_10_Relu6_model_1_model_batch_normalization_15_FusedBatchNormV3_model_1_model_batch_normalization_16_FusedBatchNormV3_model_1_model_depthwise_conv2d_5_depthwise_model_1_model_conv2d_10_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_11_Relu6_model_1_model_batch_normalization_16_FusedBatchNormV3_model_1_model_depthwise_conv2d_5_depthwise = self.conv_30(Relu6__2049_0)
        Relu6__2051_0 = torch.clamp(model_1_model_re_lu_11_Relu6_model_1_model_batch_normalization_16_FusedBatchNormV3_model_1_model_depthwise_conv2d_5_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_17_FusedBatchNormV3_model_1_model_conv2d_15_Conv2D_model_1_model_conv2d_11_Conv2D1 = self.conv_32(Relu6__2051_0)
        model_1_model_re_lu_12_Relu6_model_1_model_batch_normalization_18_FusedBatchNormV3_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise_model_1_model_conv2d_12_Conv2D = self.conv_33(model_1_model_batch_normalization_17_FusedBatchNormV3_model_1_model_conv2d_15_Conv2D_model_1_model_conv2d_11_Conv2D1)
        Relu6__2054_0 = torch.clamp(model_1_model_re_lu_12_Relu6_model_1_model_batch_normalization_18_FusedBatchNormV3_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise_model_1_model_conv2d_12_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_13_Relu6_model_1_model_batch_normalization_19_FusedBatchNormV3_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise_model_1_model_depthwise_conv2d_6_depthwise = self.conv_35(Relu6__2054_0)
        Relu6__2056_0 = torch.clamp(model_1_model_re_lu_13_Relu6_model_1_model_batch_normalization_19_FusedBatchNormV3_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise_model_1_model_depthwise_conv2d_6_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_20_FusedBatchNormV3_model_1_model_conv2d_15_Conv2D_model_1_model_conv2d_13_Conv2D1 = self.conv_37(Relu6__2056_0)
        model_1_model_add_2_add = model_1_model_batch_normalization_20_FusedBatchNormV3_model_1_model_conv2d_15_Conv2D_model_1_model_conv2d_13_Conv2D1 + model_1_model_batch_normalization_17_FusedBatchNormV3_model_1_model_conv2d_15_Conv2D_model_1_model_conv2d_11_Conv2D1
        model_1_model_re_lu_14_Relu6_model_1_model_batch_normalization_21_FusedBatchNormV3_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise_model_1_model_conv2d_14_Conv2D = self.conv_39(model_1_model_add_2_add)
        Relu6__2059_0 = torch.clamp(model_1_model_re_lu_14_Relu6_model_1_model_batch_normalization_21_FusedBatchNormV3_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise_model_1_model_conv2d_14_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_15_Relu6_model_1_model_batch_normalization_22_FusedBatchNormV3_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise_model_1_model_depthwise_conv2d_7_depthwise = self.conv_41(Relu6__2059_0)
        Relu6__2061_0 = torch.clamp(model_1_model_re_lu_15_Relu6_model_1_model_batch_normalization_22_FusedBatchNormV3_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise_model_1_model_depthwise_conv2d_7_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_23_FusedBatchNormV3_model_1_model_conv2d_15_Conv2D1 = self.conv_43(Relu6__2061_0)
        model_1_model_add_3_add = model_1_model_batch_normalization_23_FusedBatchNormV3_model_1_model_conv2d_15_Conv2D1 + model_1_model_add_2_add
        model_1_model_re_lu_16_Relu6_model_1_model_batch_normalization_24_FusedBatchNormV3_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise_model_1_model_conv2d_16_Conv2D = self.conv_45(model_1_model_add_3_add)
        Relu6__2064_0 = torch.clamp(model_1_model_re_lu_16_Relu6_model_1_model_batch_normalization_24_FusedBatchNormV3_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise_model_1_model_conv2d_16_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_17_Relu6_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise = self.conv_47(Relu6__2064_0)
        Relu6__2066_0 = torch.clamp(model_1_model_re_lu_17_Relu6_model_1_model_batch_normalization_25_FusedBatchNormV3_model_1_model_depthwise_conv2d_8_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_26_FusedBatchNormV3_model_1_model_conv2d_21_Conv2D_model_1_model_conv2d_17_Conv2D1 = self.conv_49(Relu6__2066_0)
        model_1_model_re_lu_18_Relu6_model_1_model_batch_normalization_27_FusedBatchNormV3_model_1_model_batch_normalization_34_FusedBatchNormV3_model_1_model_depthwise_conv2d_11_depthwise_model_1_model_conv2d_18_Conv2D = self.conv_50(model_1_model_batch_normalization_26_FusedBatchNormV3_model_1_model_conv2d_21_Conv2D_model_1_model_conv2d_17_Conv2D1)
        Relu6__2069_0 = torch.clamp(model_1_model_re_lu_18_Relu6_model_1_model_batch_normalization_27_FusedBatchNormV3_model_1_model_batch_normalization_34_FusedBatchNormV3_model_1_model_depthwise_conv2d_11_depthwise_model_1_model_conv2d_18_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_19_Relu6_model_1_model_batch_normalization_28_FusedBatchNormV3_model_1_model_batch_normalization_34_FusedBatchNormV3_model_1_model_depthwise_conv2d_11_depthwise_model_1_model_depthwise_conv2d_9_depthwise = self.conv_52(Relu6__2069_0)
        Relu6__2071_0 = torch.clamp(model_1_model_re_lu_19_Relu6_model_1_model_batch_normalization_28_FusedBatchNormV3_model_1_model_batch_normalization_34_FusedBatchNormV3_model_1_model_depthwise_conv2d_11_depthwise_model_1_model_depthwise_conv2d_9_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_29_FusedBatchNormV3_model_1_model_conv2d_21_Conv2D_model_1_model_conv2d_19_Conv2D1 = self.conv_54(Relu6__2071_0)
        model_1_model_add_4_add = model_1_model_batch_normalization_29_FusedBatchNormV3_model_1_model_conv2d_21_Conv2D_model_1_model_conv2d_19_Conv2D1 + model_1_model_batch_normalization_26_FusedBatchNormV3_model_1_model_conv2d_21_Conv2D_model_1_model_conv2d_17_Conv2D1
        model_1_model_re_lu_20_Relu6_model_1_model_batch_normalization_30_FusedBatchNormV3_model_1_model_batch_normalization_34_FusedBatchNormV3_model_1_model_depthwise_conv2d_11_depthwise_model_1_model_conv2d_20_Conv2D = self.conv_56(model_1_model_add_4_add)
        Relu6__2074_0 = torch.clamp(model_1_model_re_lu_20_Relu6_model_1_model_batch_normalization_30_FusedBatchNormV3_model_1_model_batch_normalization_34_FusedBatchNormV3_model_1_model_depthwise_conv2d_11_depthwise_model_1_model_conv2d_20_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_21_Relu6_model_1_model_batch_normalization_31_FusedBatchNormV3_model_1_model_depthwise_conv2d_10_depthwise_model_1_model_depthwise_conv2d_11_depthwise = self.conv_58(Relu6__2074_0)
        Relu6__2076_0 = torch.clamp(model_1_model_re_lu_21_Relu6_model_1_model_batch_normalization_31_FusedBatchNormV3_model_1_model_depthwise_conv2d_10_depthwise_model_1_model_depthwise_conv2d_11_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_32_FusedBatchNormV3_model_1_model_conv2d_21_Conv2D1 = self.conv_60(Relu6__2076_0)
        model_1_model_add_5_add = model_1_model_batch_normalization_32_FusedBatchNormV3_model_1_model_conv2d_21_Conv2D1 + model_1_model_add_4_add
        model_1_model_re_lu_22_Relu6_model_1_model_batch_normalization_33_FusedBatchNormV3_model_1_model_batch_normalization_34_FusedBatchNormV3_model_1_model_depthwise_conv2d_11_depthwise_model_1_model_conv2d_22_Conv2D = self.conv_62(model_1_model_add_5_add)
        Relu6__2079_0 = torch.clamp(model_1_model_re_lu_22_Relu6_model_1_model_batch_normalization_33_FusedBatchNormV3_model_1_model_batch_normalization_34_FusedBatchNormV3_model_1_model_depthwise_conv2d_11_depthwise_model_1_model_conv2d_22_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_23_Relu6_model_1_model_batch_normalization_34_FusedBatchNormV3_model_1_model_depthwise_conv2d_11_depthwise = self.conv_64(Relu6__2079_0)
        Relu6__2081_0 = torch.clamp(model_1_model_re_lu_23_Relu6_model_1_model_batch_normalization_34_FusedBatchNormV3_model_1_model_depthwise_conv2d_11_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_35_FusedBatchNormV3_model_1_model_conv2d_29_Conv2D_model_1_model_conv2d_23_Conv2D1 = self.conv_66(Relu6__2081_0)
        model_1_model_re_lu_24_Relu6_model_1_model_batch_normalization_36_FusedBatchNormV3_model_1_model_batch_normalization_46_FusedBatchNormV3_model_1_model_depthwise_conv2d_15_depthwise_model_1_model_conv2d_24_Conv2D = self.conv_67(model_1_model_batch_normalization_35_FusedBatchNormV3_model_1_model_conv2d_29_Conv2D_model_1_model_conv2d_23_Conv2D1)
        Relu6__2084_0 = torch.clamp(model_1_model_re_lu_24_Relu6_model_1_model_batch_normalization_36_FusedBatchNormV3_model_1_model_batch_normalization_46_FusedBatchNormV3_model_1_model_depthwise_conv2d_15_depthwise_model_1_model_conv2d_24_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_25_Relu6_model_1_model_batch_normalization_37_FusedBatchNormV3_model_1_model_depthwise_conv2d_12_depthwise_model_1_model_depthwise_conv2d_15_depthwise = self.conv_69(Relu6__2084_0)
        Relu6__2086_0 = torch.clamp(model_1_model_re_lu_25_Relu6_model_1_model_batch_normalization_37_FusedBatchNormV3_model_1_model_depthwise_conv2d_12_depthwise_model_1_model_depthwise_conv2d_15_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_38_FusedBatchNormV3_model_1_model_conv2d_29_Conv2D_model_1_model_conv2d_25_Conv2D1 = self.conv_71(Relu6__2086_0)
        model_1_model_add_6_add = model_1_model_batch_normalization_38_FusedBatchNormV3_model_1_model_conv2d_29_Conv2D_model_1_model_conv2d_25_Conv2D1 + model_1_model_batch_normalization_35_FusedBatchNormV3_model_1_model_conv2d_29_Conv2D_model_1_model_conv2d_23_Conv2D1
        model_1_model_re_lu_26_Relu6_model_1_model_batch_normalization_39_FusedBatchNormV3_model_1_model_batch_normalization_46_FusedBatchNormV3_model_1_model_depthwise_conv2d_15_depthwise_model_1_model_conv2d_26_Conv2D = self.conv_73(model_1_model_add_6_add)
        Relu6__2089_0 = torch.clamp(model_1_model_re_lu_26_Relu6_model_1_model_batch_normalization_39_FusedBatchNormV3_model_1_model_batch_normalization_46_FusedBatchNormV3_model_1_model_depthwise_conv2d_15_depthwise_model_1_model_conv2d_26_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_27_Relu6_model_1_model_batch_normalization_40_FusedBatchNormV3_model_1_model_depthwise_conv2d_13_depthwise_model_1_model_depthwise_conv2d_15_depthwise = self.conv_75(Relu6__2089_0)
        Relu6__2091_0 = torch.clamp(model_1_model_re_lu_27_Relu6_model_1_model_batch_normalization_40_FusedBatchNormV3_model_1_model_depthwise_conv2d_13_depthwise_model_1_model_depthwise_conv2d_15_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_41_FusedBatchNormV3_model_1_model_conv2d_29_Conv2D_model_1_model_conv2d_27_Conv2D1 = self.conv_77(Relu6__2091_0)
        model_1_model_add_7_add = model_1_model_batch_normalization_41_FusedBatchNormV3_model_1_model_conv2d_29_Conv2D_model_1_model_conv2d_27_Conv2D1 + model_1_model_add_6_add
        model_1_model_re_lu_28_Relu6_model_1_model_batch_normalization_42_FusedBatchNormV3_model_1_model_batch_normalization_46_FusedBatchNormV3_model_1_model_depthwise_conv2d_15_depthwise_model_1_model_conv2d_28_Conv2D = self.conv_79(model_1_model_add_7_add)
        Relu6__2094_0 = torch.clamp(model_1_model_re_lu_28_Relu6_model_1_model_batch_normalization_42_FusedBatchNormV3_model_1_model_batch_normalization_46_FusedBatchNormV3_model_1_model_depthwise_conv2d_15_depthwise_model_1_model_conv2d_28_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_29_Relu6_model_1_model_batch_normalization_43_FusedBatchNormV3_model_1_model_depthwise_conv2d_14_depthwise_model_1_model_depthwise_conv2d_15_depthwise = self.conv_81(Relu6__2094_0)
        Relu6__2096_0 = torch.clamp(model_1_model_re_lu_29_Relu6_model_1_model_batch_normalization_43_FusedBatchNormV3_model_1_model_depthwise_conv2d_14_depthwise_model_1_model_depthwise_conv2d_15_depthwise, min=0.0, max=6.0)
        model_1_model_batch_normalization_44_FusedBatchNormV3_model_1_model_conv2d_29_Conv2D1 = self.conv_83(Relu6__2096_0)
        model_1_model_add_8_add = model_1_model_batch_normalization_44_FusedBatchNormV3_model_1_model_conv2d_29_Conv2D1 + model_1_model_add_7_add
        model_1_model_re_lu_30_Relu6_model_1_model_batch_normalization_45_FusedBatchNormV3_model_1_model_batch_normalization_46_FusedBatchNormV3_model_1_model_depthwise_conv2d_15_depthwise_model_1_model_conv2d_30_Conv2D = self.conv_85(model_1_model_add_8_add)
        Relu6__2099_0 = torch.clamp(model_1_model_re_lu_30_Relu6_model_1_model_batch_normalization_45_FusedBatchNormV3_model_1_model_batch_normalization_46_FusedBatchNormV3_model_1_model_depthwise_conv2d_15_depthwise_model_1_model_conv2d_30_Conv2D, min=0.0, max=6.0)
        model_1_model_re_lu_31_Relu6_model_1_model_batch_normalization_46_FusedBatchNormV3_model_1_model_depthwise_conv2d_15_depthwise = self.conv_87(Relu6__2099_0)
        Relu6__2101_0 = torch.clamp(model_1_model_re_lu_31_Relu6_model_1_model_batch_normalization_46_FusedBatchNormV3_model_1_model_depthwise_conv2d_15_depthwise, min=0.0, max=6.0)
        model_1_model_global_average_pooling2d_Mean = self.gap_89(Relu6__2101_0)
        model_1_model_global_average_pooling2d_Mean_Squeeze__2605_0 = torch.squeeze(model_1_model_global_average_pooling2d_Mean)
        model_1_model_conv_handflag_MatMul_model_1_model_conv_handflag_BiasAdd_Gemm__2113_0 = self.linear_91(model_1_model_global_average_pooling2d_Mean_Squeeze__2605_0)
        model_1_model_conv_handedness_MatMul_model_1_model_conv_handedness_BiasAdd_Gemm__2114_0 = self.linear_92(model_1_model_global_average_pooling2d_Mean_Squeeze__2605_0)
        Identity = self.linear_93(model_1_model_global_average_pooling2d_Mean_Squeeze__2605_0)
        Identity_3 = self.linear_94(model_1_model_global_average_pooling2d_Mean_Squeeze__2605_0)
        Identity_2 = self.sigmoid_95(model_1_model_conv_handedness_MatMul_model_1_model_conv_handedness_BiasAdd_Gemm__2114_0)
        Identity_1 = self.sigmoid_96(model_1_model_conv_handflag_MatMul_model_1_model_conv_handflag_BiasAdd_Gemm__2113_0)
        return Identity, Identity_1, Identity_2, Identity_3

# How to load the model and weights:
# 1. Create an instance of the model:
#    model = hand_landmark_full()
# 2. Load the state dictionary:
#    state_dict_path = 'hand_landmark_full.pth'
#    model.load_state_dict(torch.load(state_dict_path))
# 3. Set the model to evaluation mode:
#    model.eval()
