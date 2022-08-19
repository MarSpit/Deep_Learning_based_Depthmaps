import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import namedtuple

# Depthwise Separable Convolution 
# from: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch
class Depthwise_Separable_Conv(nn.Module):
 def __init__(self, in_planes, out_planes, kernel_size=3, stride = 1, padding = 1): 
    super(Depthwise_Separable_Conv, self).__init__() 
    self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size = kernel_size, stride = stride, padding = padding, groups = in_planes) 
    self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1) 
  
 def forward(self, x): 
    out = self.depthwise(x) 
    out = self.pointwise(out) 
    return out
# based on: https://github.com/dwofk/fast-depth/blob/master/models.py
class Depthwise_Separable_Conv_Transposed(nn.Module):
 def __init__(self, in_planes, out_planes, kernel_size=2, stride = 2): 
    super(Depthwise_Separable_Conv_Transposed, self).__init__() 
    self.depthwise = nn.ConvTranspose2d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size, stride=stride, groups=in_planes)
    self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1) 
  
 def forward(self, x): 
    out = self.depthwise(x) 
    out = self.pointwise(out) 
    return out

class PyD_Net2_Netz_1(nn.Module):
    def __init__(self, conv_dim=64):
        super(PyD_Net2_Netz_1, self).__init__()

        # Features extractor
        self.conv_ext_1 = self.conv_down_block(4,32)
        self.conv_ext_2 = self.conv_down_block(32, 64)
        self.conv_ext_3 = self.conv_down_block(64, 128)
        self.conv_ext_4 = self.conv_down_block(128, 192)

        # Depth Decoder
        self.conv_dec_1 = self.conv_disp_block(40)
        self.conv_dec_2 = self.conv_disp_block(72)
        self.conv_dec_3 = self.conv_disp_block(136)
        self.conv_dec_4 = self.conv_disp_block(192)
        self.disp = nn.Sigmoid()

        # Upsampling
        self.deconv = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2)
        self.deconv_final = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=2, stride=2)

    def conv_down_block(self, in_channels, out_channels):
        conv_down_block = []
        conv_down_block += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        conv_down_block += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()]
        return nn.Sequential(*conv_down_block)


    def conv_disp_block(self, in_channels):
        conv_disp_block = []
        conv_disp_block += [nn.Conv2d(in_channels = in_channels, out_channels= 96 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU()]
        conv_disp_block += [nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding = 1), nn.LeakyReLU()]
        conv_disp_block += [nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()]
        conv_disp_block += [nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*conv_disp_block)

    def forward(self, input):
        image = input['image']
        depth = input['depth']
        x = torch.concat((image, depth), 1)
        conv1 = self.conv_ext_1(x)     # 4,4,352,1216 --> 4,32,176,608
        conv2 = self.conv_ext_2(conv1) # 4,32,176,608 --> 4,64,88,304
        conv3 = self.conv_ext_3(conv2) # 4,64, 88,304 --> 4,128,44,152
        conv4 = self.conv_ext_4(conv3) # 4,128,44,152 --> 4,192,22,76

        conv4b = self.conv_dec_4(conv4)
        conv4b = self.deconv(conv4b) 
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.conv_dec_3(concat3)
        conv3b = self.deconv(conv3b)
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.conv_dec_2(concat2)
        conv2b = self.deconv(conv2b)
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.conv_dec_1(concat1)
        final_depth = self.deconv_final(conv1b)

        return final_depth

class PyD_Net2_Netz_2(nn.Module):
    def __init__(self, conv_dim=64):
        super(PyD_Net2_Netz_2, self).__init__()

        # Features extractor
        self.conv_ext_1 = self.conv_down_block(4,32)
        self.conv_ext_2 = self.conv_down_block(32, 64)
        self.conv_ext_3 = self.conv_down_block(64, 128)
        self.conv_ext_4 = self.conv_down_block(128, 192)

        # Depth Decoder
        self.conv_dec_1 = self.conv_disp_block(40)
        self.conv_dec_2 = self.conv_disp_block(72)
        self.conv_dec_3 = self.conv_disp_block(136)
        self.conv_dec_4 = self.conv_disp_block(192)

        self.disp = nn.Sigmoid()

        # Upsampling
        self.deconv = Depthwise_Separable_Conv_Transposed(in_planes = 8, out_planes = 8, kernel_size=2, stride = 2)
        self.deconv_final = Depthwise_Separable_Conv_Transposed(in_planes = 8, out_planes = 1, kernel_size=2, stride = 2)

    def conv_down_block(self, in_channels, out_channels):
        conv_down_block = []
        conv_down_block += [Depthwise_Separable_Conv(in_planes = in_channels, out_planes= out_channels, kernel_size=3, stride = 2, padding = 1), nn.LeakyReLU()]
        conv_down_block += [Depthwise_Separable_Conv(in_planes = out_channels, out_planes= out_channels, kernel_size=3, stride = 1, padding = 1), nn.LeakyReLU()]
        return nn.Sequential(*conv_down_block)

    def conv_disp_block(self, in_channels):
        conv_disp_block = []
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = in_channels, out_planes= 96), nn.LeakyReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 96, out_planes= 64), nn.LeakyReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 64, out_planes= 32), nn.LeakyReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 32, out_planes= 8), nn.LeakyReLU()]
        return nn.Sequential(*conv_disp_block)

    def forward(self, input):
        image = input['image']
        depth = input['depth']
        x = torch.concat((image, depth), 1)
        conv1 = self.conv_ext_1(x)     # 4,4,352,1216 --> 4,32,176,608
        conv2 = self.conv_ext_2(conv1) # 4,32,176,608 --> 4,64,88,304
        conv3 = self.conv_ext_3(conv2) # 4,64, 88,304 --> 4,128,44,152
        conv4 = self.conv_ext_4(conv3) # 4,128,44,152 --> 4,192,22,76
        
        conv4b = self.conv_dec_4(conv4)
        conv4b = self.deconv(conv4b)
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.conv_dec_3(concat3)
        conv3b = self.deconv(conv3b)
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.conv_dec_2(concat2)
        conv2b = self.deconv(conv2b)
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.conv_dec_1(concat1)
        disp1 = self.disp(conv1b)
        final_depth = self.deconv_final(disp1)

        return final_depth

class PyD_Net2_Netz_3(nn.Module):
    def __init__(self, conv_dim=64):
        super(PyD_Net2_Netz_3, self).__init__()

        # Features extractor
        self.conv_ext_1 = self.conv_down_block(4,16)
        self.conv_ext_2 = self.conv_down_block(16, 32)
        self.conv_ext_3 = self.conv_down_block(32, 64)
        self.conv_ext_4 = self.conv_down_block(64, 92)

        # Depth Decoder
        self.conv_dec_1 = self.conv_disp_block(24)
        self.conv_dec_2 = self.conv_disp_block(40)
        self.conv_dec_3 = self.conv_disp_block(72)
        self.conv_dec_4 = self.conv_disp_block(92)

        self.disp = nn.Sigmoid()

        # Upsampling
        self.deconv = Depthwise_Separable_Conv_Transposed(in_planes = 8, out_planes = 8, kernel_size=2, stride = 2)
        self.deconv_final = Depthwise_Separable_Conv_Transposed(in_planes = 8, out_planes = 1, kernel_size=2, stride = 2)

    def conv_down_block(self, in_channels, out_channels):
        conv_down_block = []
        conv_down_block += [Depthwise_Separable_Conv(in_planes = in_channels, out_planes= out_channels, kernel_size=3, stride = 2, padding = 1), nn.LeakyReLU()]
        conv_down_block += [Depthwise_Separable_Conv(in_planes = out_channels, out_planes= out_channels, kernel_size=3, stride = 1, padding = 1), nn.LeakyReLU()]
        return nn.Sequential(*conv_down_block)

    def conv_disp_block(self, in_channels):
        conv_disp_block = []
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = in_channels, out_planes= 96), nn.LeakyReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 96, out_planes= 64), nn.LeakyReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 64, out_planes= 32), nn.LeakyReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 32, out_planes= 8), nn.LeakyReLU()]
        return nn.Sequential(*conv_disp_block)

    def forward(self, input):
        image = input['image']
        depth = input['depth']
        x = torch.concat((image, depth), 1)
        conv1 = self.conv_ext_1(x)     # 4,4,352,1216 --> 4,32,176,608
        conv2 = self.conv_ext_2(conv1) # 4,32,176,608 --> 4,64,88,304
        conv3 = self.conv_ext_3(conv2) # 4,64, 88,304 --> 4,128,44,152
        conv4 = self.conv_ext_4(conv3) # 4,128,44,152 --> 4,192,22,76
       
        conv4b = self.conv_dec_4(conv4)
        conv4b = self.deconv(conv4b) 
        concat3 = torch.cat((conv3, conv4b), 1)
        conv3b = self.conv_dec_3(concat3)
        conv3b = self.deconv(conv3b)
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.conv_dec_2(concat2)
        conv2b = self.deconv(conv2b)
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.conv_dec_1(concat1)
        disp1 = self.disp(conv1b)
        final_depth = self.deconv_final(disp1)

        return final_depth
    
class PyD_Net2_Netz_4(nn.Module):
    def __init__(self, conv_dim=64):
        super(PyD_Net2_Netz_4, self).__init__()
        # Features extractor
        self.conv_ext_1 = self.conv_down_block(4,16)
        self.conv_ext_2 = self.conv_down_block(16, 32)
        self.conv_ext_3 = self.conv_down_block(32, 64)

        # Depth Decoder
        self.conv_dec_1 = self.conv_disp_block(24)
        self.conv_dec_2 = self.conv_disp_block(40)
        self.conv_dec_3 = self.conv_disp_block(64)
        self.disp = nn.Sigmoid()

        # Upsampling
        self.deconv = Depthwise_Separable_Conv_Transposed(in_planes = 8, out_planes = 8, kernel_size=2, stride = 2)
        self.deconv_final = Depthwise_Separable_Conv_Transposed(in_planes = 8, out_planes = 1, kernel_size=2, stride = 2)

    def conv_down_block(self, in_channels, out_channels):
        conv_down_block = []
        conv_down_block += [Depthwise_Separable_Conv(in_planes = in_channels, out_planes= out_channels, kernel_size=3, stride = 2, padding = 1), nn.LeakyReLU()]
        conv_down_block += [Depthwise_Separable_Conv(in_planes = out_channels, out_planes= out_channels, kernel_size=3, stride = 1, padding = 1), nn.LeakyReLU()]
        return nn.Sequential(*conv_down_block)

    def conv_disp_block(self, in_channels):
        conv_disp_block = []
        
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = in_channels, out_planes= 96), nn.LeakyReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 96, out_planes= 64), nn.LeakyReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 64, out_planes= 32), nn.LeakyReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 32, out_planes= 8), nn.LeakyReLU()]
        return nn.Sequential(*conv_disp_block)

    def forward(self, input):
        image = input['image']
        depth = input['depth']
        x = torch.concat((image, depth), 1)
        conv1 = self.conv_ext_1(x)     # 4,4,352,1216 --> 4,32,176,608
        conv2 = self.conv_ext_2(conv1) # 4,32,176,608 --> 4,64,88,304
        conv3 = self.conv_ext_3(conv2) # 4,64, 88,304 --> 4,128,44,152
        
        conv3b = self.conv_dec_3(conv3)
        conv3b = self.deconv(conv3b)
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.conv_dec_2(concat2)
        conv2b = self.deconv(conv2b)
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.conv_dec_1(concat1)
        disp1 = self.disp(conv1b)
        final_depth = self.deconv_final(disp1)

        return final_depth

class PyD_Net2_Netz_5(nn.Module):
    def __init__(self, conv_dim=64):
        super(PyD_Net2_Netz_5, self).__init__()
        # Features extractor
        self.conv_ext_1 = self.conv_down_block(4,16)
        self.conv_ext_2 = self.conv_down_block(16, 32)
        self.conv_ext_3 = self.conv_down_block(32, 64)

        # Depth Decoder
        self.conv_dec_1 = self.conv_disp_block(24)
        self.conv_dec_2 = self.conv_disp_block(40)
        self.conv_dec_3 = self.conv_disp_block(64)
        self.disp = nn.Sigmoid()

        # Upsampling
        self.deconv = Depthwise_Separable_Conv_Transposed(in_planes = 8, out_planes = 8, kernel_size=2, stride = 2)
        self.deconv_final = Depthwise_Separable_Conv_Transposed(in_planes = 8, out_planes = 1, kernel_size=2, stride = 2)

    def conv_down_block(self, in_channels, out_channels):
        conv_down_block = []
        conv_down_block += [Depthwise_Separable_Conv(in_planes = in_channels, out_planes= out_channels, kernel_size=3, stride = 2, padding = 1), nn.ReLU()]
        conv_down_block += [Depthwise_Separable_Conv(in_planes = out_channels, out_planes= out_channels, kernel_size=3, stride = 1, padding = 1), nn.ReLU()]
        return nn.Sequential(*conv_down_block)

    def conv_disp_block(self, in_channels):
        conv_disp_block = []
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = in_channels, out_planes= 96), nn.ReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 96, out_planes= 64), nn.ReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 64, out_planes= 32), nn.ReLU()]
        conv_disp_block += [Depthwise_Separable_Conv(in_planes = 32, out_planes= 8), nn.ReLU()]
        return nn.Sequential(*conv_disp_block)

    def forward(self, input):
        image = input['image']
        depth = input['depth']
        x = torch.concat((image, depth), 1)
        conv1 = self.conv_ext_1(x)     # 4,4,352,1216 --> 4,32,176,608
        conv2 = self.conv_ext_2(conv1) # 4,32,176,608 --> 4,64,88,304
        conv3 = self.conv_ext_3(conv2) # 4,64, 88,304 --> 4,128,44,152
      
        conv3b = self.conv_dec_3(conv3)
        conv3b = self.deconv(conv3b)
        concat2 = torch.cat((conv2, conv3b), 1)
        conv2b = self.conv_dec_2(concat2)
        conv2b = self.deconv(conv2b)
        concat1 = torch.cat((conv1, conv2b), 1)
        conv1b = self.conv_dec_1(concat1)
        disp1 = self.disp(conv1b)
        final_depth = self.deconv_final(disp1)

        return final_depth