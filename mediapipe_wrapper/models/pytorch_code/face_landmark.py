import torch
import torch.nn as nn
import torch.nn.functional as F

class face_landmark(nn.Module):
    def __init__(self):
        super(face_landmark, self).__init__()
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (1, 1), 'group': 1}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 128, 'pads': (1, 1, 1, 1)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 16, 'pads': (1, 1, 1, 1)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 32, 'pads': (1, 1, 1, 1)}
        # Attributes: {'strides': (1, 1), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 64, 'pads': (1, 1, 1, 1)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 128, 'pads': (0, 0, 1, 1)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 16, 'pads': (0, 0, 1, 1)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 32, 'pads': (0, 0, 1, 1)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 64, 'pads': (0, 0, 1, 1)}
        # Attributes: {'strides': (2, 2), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'pads': (0, 0, 1, 1), 'group': 1}
        # Attributes: {'strides': (3, 3), 'dilations': (1, 1), 'kernel_shape': (3, 3), 'group': 1}
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), groups=1, bias=True)
        self.conv_13 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), groups=16, bias=True)
        self.conv_14 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_17 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=True)
        self.conv_18 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_21 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=True)
        self.conv_22 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_27 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), groups=32, bias=True)
        self.conv_28 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=True)
        self.conv_31 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=True)
        self.conv_32 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_35 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=True)
        self.conv_36 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_41 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), groups=64, bias=True)
        self.conv_42 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_45 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=True)
        self.conv_46 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_49 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=True)
        self.conv_50 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_54 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), groups=128, bias=True)
        self.conv_55 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_58 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=True)
        self.conv_59 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_62 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=True)
        self.conv_63 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_67 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), groups=128, bias=True)
        self.conv_68 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=True)
        self.conv_71 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_73 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=True)
        self.conv_74 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_77 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(3, 3), padding=(0, 0), groups=1, bias=True)
        self.conv_79 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), groups=128, bias=True)
        self.conv_8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_80 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_83 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=True)
        self.conv_84 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_87 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=True)
        self.conv_88 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_91 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_93 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=True)
        self.conv_94 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=True)
        self.conv_97 = nn.Conv2d(in_channels=32, out_channels=1404, kernel_size=(3, 3), stride=(3, 3), padding=(0, 0), groups=1, bias=True)
        self.maxpool_11 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.maxpool_25 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.maxpool_39 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.maxpool_53 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.maxpool_66 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.prelu_10 = nn.PReLU(num_parameters=1)
        self.prelu_16 = nn.PReLU(num_parameters=1)
        self.prelu_2 = nn.PReLU(num_parameters=1)
        self.prelu_20 = nn.PReLU(num_parameters=1)
        self.prelu_24 = nn.PReLU(num_parameters=1)
        self.prelu_30 = nn.PReLU(num_parameters=1)
        self.prelu_34 = nn.PReLU(num_parameters=1)
        self.prelu_38 = nn.PReLU(num_parameters=1)
        self.prelu_44 = nn.PReLU(num_parameters=1)
        self.prelu_48 = nn.PReLU(num_parameters=1)
        self.prelu_52 = nn.PReLU(num_parameters=1)
        self.prelu_57 = nn.PReLU(num_parameters=1)
        self.prelu_6 = nn.PReLU(num_parameters=1)
        self.prelu_61 = nn.PReLU(num_parameters=1)
        self.prelu_65 = nn.PReLU(num_parameters=1)
        self.prelu_70 = nn.PReLU(num_parameters=1)
        self.prelu_72 = nn.PReLU(num_parameters=1)
        self.prelu_76 = nn.PReLU(num_parameters=1)
        self.prelu_82 = nn.PReLU(num_parameters=1)
        self.prelu_86 = nn.PReLU(num_parameters=1)
        self.prelu_90 = nn.PReLU(num_parameters=1)
        self.prelu_92 = nn.PReLU(num_parameters=1)
        self.prelu_96 = nn.PReLU(num_parameters=1)

    def forward(self, input_1):
        conv2d_1__1356_0 = input_1.permute(0, 3, 1, 2)
        conv2d_1 = self.conv_1(conv2d_1__1356_0)
        p_re_lu_1 = self.prelu_2(conv2d_1)
        depthwise_conv2d_1 = self.conv_3(p_re_lu_1)
        conv2d_2 = self.conv_4(depthwise_conv2d_1)
        add_1 = p_re_lu_1 + conv2d_2
        p_re_lu_2 = self.prelu_6(add_1)
        depthwise_conv2d_2 = self.conv_7(p_re_lu_2)
        conv2d_3 = self.conv_8(depthwise_conv2d_2)
        add_2 = p_re_lu_2 + conv2d_3
        p_re_lu_3 = self.prelu_10(add_2)
        max_pooling2d_1 = self.maxpool_11(p_re_lu_3)
        channel_padding_1 = F.pad(max_pooling2d_1, (0, 0, 0, 0), mode='constant', value=0.0)
        depthwise_conv2d_3 = self.conv_13(p_re_lu_3)
        conv2d_4 = self.conv_14(depthwise_conv2d_3)
        add_3 = channel_padding_1 + conv2d_4
        p_re_lu_4 = self.prelu_16(add_3)
        depthwise_conv2d_4 = self.conv_17(p_re_lu_4)
        conv2d_5 = self.conv_18(depthwise_conv2d_4)
        add_4 = p_re_lu_4 + conv2d_5
        p_re_lu_5 = self.prelu_20(add_4)
        depthwise_conv2d_5 = self.conv_21(p_re_lu_5)
        conv2d_6 = self.conv_22(depthwise_conv2d_5)
        add_5 = p_re_lu_5 + conv2d_6
        p_re_lu_6 = self.prelu_24(add_5)
        max_pooling2d_2 = self.maxpool_25(p_re_lu_6)
        channel_padding_2 = F.pad(max_pooling2d_2, (0, 0, 0, 0), mode='constant', value=0.0)
        depthwise_conv2d_6 = self.conv_27(p_re_lu_6)
        conv2d_7 = self.conv_28(depthwise_conv2d_6)
        add_6 = channel_padding_2 + conv2d_7
        p_re_lu_7 = self.prelu_30(add_6)
        depthwise_conv2d_7 = self.conv_31(p_re_lu_7)
        conv2d_8 = self.conv_32(depthwise_conv2d_7)
        add_7 = p_re_lu_7 + conv2d_8
        p_re_lu_8 = self.prelu_34(add_7)
        depthwise_conv2d_8 = self.conv_35(p_re_lu_8)
        conv2d_9 = self.conv_36(depthwise_conv2d_8)
        add_8 = p_re_lu_8 + conv2d_9
        p_re_lu_9 = self.prelu_38(add_8)
        max_pooling2d_3 = self.maxpool_39(p_re_lu_9)
        channel_padding_3 = F.pad(max_pooling2d_3, (0, 0, 0, 0), mode='constant', value=0.0)
        depthwise_conv2d_9 = self.conv_41(p_re_lu_9)
        conv2d_10 = self.conv_42(depthwise_conv2d_9)
        add_9 = channel_padding_3 + conv2d_10
        p_re_lu_10 = self.prelu_44(add_9)
        depthwise_conv2d_10 = self.conv_45(p_re_lu_10)
        conv2d_11 = self.conv_46(depthwise_conv2d_10)
        add_10 = p_re_lu_10 + conv2d_11
        p_re_lu_11 = self.prelu_48(add_10)
        depthwise_conv2d_11 = self.conv_49(p_re_lu_11)
        conv2d_12 = self.conv_50(depthwise_conv2d_11)
        add_11 = p_re_lu_11 + conv2d_12
        p_re_lu_12 = self.prelu_52(add_11)
        max_pooling2d_4 = self.maxpool_53(p_re_lu_12)
        depthwise_conv2d_12 = self.conv_54(p_re_lu_12)
        conv2d_13 = self.conv_55(depthwise_conv2d_12)
        add_12 = max_pooling2d_4 + conv2d_13
        p_re_lu_13 = self.prelu_57(add_12)
        depthwise_conv2d_13 = self.conv_58(p_re_lu_13)
        conv2d_14 = self.conv_59(depthwise_conv2d_13)
        add_13 = p_re_lu_13 + conv2d_14
        p_re_lu_14 = self.prelu_61(add_13)
        depthwise_conv2d_14 = self.conv_62(p_re_lu_14)
        conv2d_15 = self.conv_63(depthwise_conv2d_14)
        add_14 = p_re_lu_14 + conv2d_15
        p_re_lu_15 = self.prelu_65(add_14)
        max_pooling2d_5 = self.maxpool_66(p_re_lu_15)
        depthwise_conv2d_23 = self.conv_67(p_re_lu_15)
        conv2d_28 = self.conv_68(depthwise_conv2d_23)
        add_23 = max_pooling2d_5 + conv2d_28
        p_re_lu_26 = self.prelu_70(add_23)
        conv2d_29 = self.conv_71(p_re_lu_26)
        p_re_lu_27 = self.prelu_72(conv2d_29)
        depthwise_conv2d_24 = self.conv_73(p_re_lu_27)
        conv2d_30 = self.conv_74(depthwise_conv2d_24)
        add_24 = p_re_lu_27 + conv2d_30
        p_re_lu_28 = self.prelu_76(add_24)
        conv2d_31_raw_output___1309_0 = self.conv_77(p_re_lu_28)
        conv2d_31 = conv2d_31_raw_output___1309_0.reshape(conv2d_31_raw_output___1309_0.size(0), 1, 1, 1)
        depthwise_conv2d_15 = self.conv_79(p_re_lu_15)
        conv2d_16 = self.conv_80(depthwise_conv2d_15)
        add_15 = max_pooling2d_5 + conv2d_16
        p_re_lu_16 = self.prelu_82(add_15)
        depthwise_conv2d_16 = self.conv_83(p_re_lu_16)
        conv2d_17 = self.conv_84(depthwise_conv2d_16)
        add_16 = p_re_lu_16 + conv2d_17
        p_re_lu_17 = self.prelu_86(add_16)
        depthwise_conv2d_17 = self.conv_87(p_re_lu_17)
        conv2d_18 = self.conv_88(depthwise_conv2d_17)
        add_17 = p_re_lu_17 + conv2d_18
        p_re_lu_18 = self.prelu_90(add_17)
        conv2d_19 = self.conv_91(p_re_lu_18)
        p_re_lu_19 = self.prelu_92(conv2d_19)
        depthwise_conv2d_18 = self.conv_93(p_re_lu_19)
        conv2d_20 = self.conv_94(depthwise_conv2d_18)
        add_18 = p_re_lu_19 + conv2d_20
        p_re_lu_20 = self.prelu_96(add_18)
        conv2d_21_raw_output___1307_0 = self.conv_97(p_re_lu_20)
        conv2d_21 = conv2d_21_raw_output___1307_0.reshape(conv2d_21_raw_output___1307_0.size(0), 1, 1, 1404)
        return conv2d_21, conv2d_31

# How to load the model and weights:
# 1. Create an instance of the model:
#    model = face_landmark()
# 2. Load the state dictionary:
#    state_dict_path = 'face_landmark.pth'
#    model.load_state_dict(torch.load(state_dict_path))
# 3. Set the model to evaluation mode:
#    model.eval()
