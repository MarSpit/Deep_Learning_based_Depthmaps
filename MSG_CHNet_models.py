import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import namedtuple

# Network Architecture from https://github.com/anglixjtu/msg_chn_wacv20

class DepthEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(DepthEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),

                                  )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),

                                  )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x2=None, pre_x3=None, pre_x4=None):
        ### input

        x0 = self.init(input)
        if pre_x4 is not None:
            x0 = x0 + F.interpolate(pre_x4, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        if pre_x3 is not None:  # newly added skip connection
            x1 = x1 + F.interpolate(pre_x3, scale_factor=scale, mode='bilinear', align_corners=True)

        x2 = self.enc2(x1)  # 1/4 input size
        if pre_x2 is not None:  # newly added skip connection
            x2 = x2 + F.interpolate(pre_x2, scale_factor=scale, mode='bilinear', align_corners=True)

        return x0, x1, x2


class RGBEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(RGBEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc3 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc4 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x=None):
        ### input

        x0 = self.init(input)
        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        x2 = self.enc2(x1)  # 1/4 input size
        x3 = self.enc3(x2)  # 1/8 input size
        x4 = self.enc4(x3)  # 1/16 input size

        return x0, x1, x2, x3, x4


class DepthDecoder(nn.Module):
    def __init__(self, layers, filter_size):
        super(DepthDecoder, self).__init__()
        padding = int((filter_size - 1) / 2)

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  )

        self.prdct = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                   nn.ReLU(),
                                   nn.Conv2d(layers // 2, 1, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_dx, pre_cx):

        x2 = pre_dx[2] + pre_cx[2]  # torch.cat((pre_dx[2], pre_cx[2]), 1)
        x1 = pre_dx[1] + pre_cx[1]  # torch.cat((pre_dx[1], pre_cx[1]), 1) #
        x0 = pre_dx[0] + pre_cx[0]

        x3 = self.dec2(x2)  # 1/2 input size
        x4 = self.dec1(x1 + x3)  # 1/1 input size

        ### prediction
        output_d = self.prdct(x4 + x0)

        return x2, x3, x4, output_d

##########################################
# Implemenations with Depthwise separable convolutional Layers (DSCL)

# Depthwise Separable Convolution 
# from: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch

class Depthwise_Separable_Conv(nn.Module):
 def __init__(self, in_planes, out_planes, kernel_size=3, stride = 1, padding = 1): 
    super(Depthwise_Separable_Conv, self).__init__() 
    self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride = stride, padding=padding, groups=in_planes) 
    self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1) 
  
 def forward(self, x): 
    out = self.depthwise(x) 
    out = self.pointwise(out) 
    return out

# based on: https://github.com/dwofk/fast-depth/blob/master/models.py
class Depthwise_Separable_Conv_Transposed(nn.Module):
 def __init__(self, in_planes, out_planes, kernel_size=2, stride = 2, padding = 1, output_padding = 1): 
    super(Depthwise_Separable_Conv_Transposed, self).__init__() 
    #self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=2, stride = 2, padding=padding, groups=in_planes) 
    self.depthwise = nn.ConvTranspose2d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size, stride=stride, padding = padding, output_padding = output_padding)
    self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1) 
  
 def forward(self, x): 
    out = self.depthwise(x) 
    out = self.pointwise(out) 
    return out

class DepthEncoder_DSCL(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(DepthEncoder_DSCL, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(Depthwise_Separable_Conv(in_planes=in_layers, out_planes=layers, kernel_size=filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=1, padding=padding)
                                  
                                  )

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=1, padding=padding),

                                  )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=1, padding=padding),

                                  )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x2=None, pre_x3=None, pre_x4=None):
        ### input

        x0 = self.init(input)
        if pre_x4 is not None:
            x0 = x0 + F.interpolate(pre_x4, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        if pre_x3 is not None:  # newly added skip connection
            x1 = x1 + F.interpolate(pre_x3, scale_factor=scale, mode='bilinear', align_corners=True)

        x2 = self.enc2(x1)  # 1/4 input size
        if pre_x2 is not None:  # newly added skip connection
            x2 = x2 + F.interpolate(pre_x2, scale_factor=scale, mode='bilinear', align_corners=True)

        return x0, x1, x2


class RGBEncoder_DSCL(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(RGBEncoder_DSCL, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(Depthwise_Separable_Conv(in_planes=in_layers, out_planes=layers, kernel_size=filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=1, padding=padding), )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=1, padding=padding), )

        self.enc3 = nn.Sequential(nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=1, padding=padding), )

        self.enc4 = nn.Sequential(nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers, out_planes=layers, kernel_size=filter_size, stride=1, padding=padding), )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x=None):
        ### input

        x0 = self.init(input)
        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        x2 = self.enc2(x1)  # 1/4 input size
        x3 = self.enc3(x2)  # 1/8 input size
        x4 = self.enc4(x3)  # 1/16 input size

        return x0, x1, x2, x3, x4


class DepthDecoder_DSCL(nn.Module):
    def __init__(self, layers, filter_size):
        super(DepthDecoder_DSCL, self).__init__()
        padding = int((filter_size - 1) / 2)

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  Depthwise_Separable_Conv_Transposed(in_planes = layers // 2, out_planes = layers // 2, kernel_size=filter_size, stride = 2, padding=padding, output_padding=padding),
                                  nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers // 2, out_planes=layers // 2, kernel_size=filter_size, stride=1, padding=padding),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  Depthwise_Separable_Conv_Transposed(in_planes = layers // 2, out_planes = layers // 2, kernel_size=filter_size, stride = 2, padding=padding, output_padding=padding),
                                  nn.ReLU(),
                                  Depthwise_Separable_Conv(in_planes=layers // 2, out_planes=layers // 2, kernel_size=filter_size, stride=1, padding=padding),
                                  )

        self.prdct = nn.Sequential(nn.ReLU(),
                                   Depthwise_Separable_Conv(in_planes=layers // 2, out_planes=layers // 2, kernel_size=filter_size, stride=1, padding=padding),
                                   nn.ReLU(),
                                   Depthwise_Separable_Conv(in_planes=layers // 2, out_planes=1, kernel_size=filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_dx, pre_cx):

        x2 = pre_dx[2] + pre_cx[2]  # torch.cat((pre_dx[2], pre_cx[2]), 1)
        x1 = pre_dx[1] + pre_cx[1]  # torch.cat((pre_dx[1], pre_cx[1]), 1) #
        x0 = pre_dx[0] + pre_cx[0]

        x3 = self.dec2(x2)  # 1/2 input size
        x4 = self.dec1(x1 + x3)  # 1/1 input size

        ### prediction
        output_d = self.prdct(x4 + x0)

        return x2, x3, x4, output_d


##########################################
##########################################
# Network implementations

class MSG_CHNet_64(nn.Module):
    def __init__(self):
        super(MSG_CHNet_64, self).__init__()

        denc_layers = 64
        cenc_layers = 64
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)

    def forward(self, input):
        input_rgb = input['image']
        input_d = input['depth']
        
        C = (input_d > 0).float()

        enc_c = self.rgb_encoder(input_rgb)

        ## for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        return output_d11, output_d12, output_d14

class MSG_CHNet_32(nn.Module):
    def __init__(self):
        super(MSG_CHNet_32, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)

    def forward(self, input):
        input_rgb = input['image']
        input_d = input['depth']
        
        C = (input_d > 0).float()

        enc_c = self.rgb_encoder(input_rgb)

        ## for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        return output_d11, output_d12, output_d14

class MSG_CHNet_Netz_1(nn.Module):
    def __init__(self):
        super(MSG_CHNet_Netz_1, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder_DSCL(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder_DSCL(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder_DSCL(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder_DSCL(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder_DSCL(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder_DSCL(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder_DSCL(ddcd_layers, 3)

    def forward(self, input):
        input_rgb = input['image']
        input_d = input['depth']
        
        C = (input_d > 0).float()

        enc_c = self.rgb_encoder(input_rgb)

        ## for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        return output_d11, output_d12, output_d14

class MSG_CHNet_Netz_2(nn.Module):
    def __init__(self):
        super(MSG_CHNet_Netz_2, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder_DSCL(3, cenc_layers, 3)

        #self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        #self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder_DSCL(1, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder_DSCL(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder_DSCL(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder_DSCL(ddcd_layers, 3)

    def forward(self, input):
        input_rgb = input['image']
        input_d = input['depth']
        
        C = (input_d > 0).float()

        enc_c = self.rgb_encoder(input_rgb)

        ## for the 1/4 res
        #input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        #enc_d14 = self.depth_encoder1(input_d14)
        #dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        #predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        #input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_d12)#input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = predict_d11
        #output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        return output_d11, output_d12#, output_d14,

class MSG_CHNet_Netz_3(nn.Module):
    def __init__(self):
        super(MSG_CHNet_Netz_3, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder_DSCL(3, cenc_layers, 3)

        #self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        #self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        #self.depth_encoder2 = DepthEncoder(1, denc_layers, 3)
        #self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder_DSCL(1, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder_DSCL(ddcd_layers, 3)

    def forward(self, input):
        input_rgb = input['image']
        input_d = input['depth']
        
        C = (input_d > 0).float()

        enc_c = self.rgb_encoder(input_rgb)

        ## for the 1/4 res
        #input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        #enc_d14 = self.depth_encoder1(input_d14)
        #dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        #input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        #predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        #input_12 = torch.cat((input_d12, predict_d12), 1)

        #enc_d12 = self.depth_encoder2(input_d12)#input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        #dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        #predict_d11 = F.interpolate(dcd_d12[3], scale_factor=2, mode='bilinear', align_corners=True)
        #input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_d)#, 1)#, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] #+ predict_d11
        #output_d12 = predict_d11
        #output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        return output_d11#, output_d12, output_d14,

class MSG_CHNet_Netz_4(nn.Module):
    def __init__(self):
        super(MSG_CHNet_Netz_4, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        #self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        #self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        #self.depth_encoder2 = DepthEncoder(1, denc_layers, 3)
        #self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder_DSCL(ddcd_layers, 3)

    def forward(self, input):
        input_rgb = input['image']
        input_d = input['depth']
        
        C = (input_d > 0).float()

        enc_c = self.rgb_encoder(input_rgb)

        ## for the 1/4 res
        #input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        #enc_d14 = self.depth_encoder1(input_d14)
        #dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        #input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        #predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        #input_12 = torch.cat((input_d12, predict_d12), 1)

        #enc_d12 = self.depth_encoder2(input_d12)#input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        #dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        #predict_d11 = F.interpolate(dcd_d12[3], scale_factor=2, mode='bilinear', align_corners=True)
        #input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_d)#, 1)#, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] #+ predict_d11
        #output_d12 = predict_d11
        #output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        return output_d11#, output_d12, output_d14,
