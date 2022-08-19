import torch
import torch.nn.functional as F
import torch.nn as nn
import params


##############################################
# Concolutional Block Attention Module (CBAM) implementation
# Combining channel and spatial attention
# Code from: https://github.com/Jongchan/attention-module

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


##############################################
# Depthwise Separable Convolution
# from: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch

class Depthwise_Separable_Conv(nn.Module):
 def __init__(self, in_planes, out_planes, stride = 1, kernel_size=3, padding = 1): 
   super(Depthwise_Separable_Conv, self).__init__() 
   self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride = stride, padding = padding, groups = in_planes) 
   self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1) 
  
 def forward(self, x): 
   out = self.depthwise(x)        
   out = self.pointwise(out)   
   return out

# based on: https://github.com/dwofk/fast-depth/blob/master/models.py
class Depthwise_Separable_Conv_Transposed(nn.Module):
 def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding = 1, output_padding = 1): 
    super(Depthwise_Separable_Conv_Transposed, self).__init__() 
    self.depthwise = nn.ConvTranspose2d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size, stride=stride, padding = padding, output_padding = output_padding, groups=in_planes)
    self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1) 
  
 def forward(self, x): 
    out = self.depthwise(x) 
    out = self.pointwise(out) 
    return out

#########################################

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

#########################################
# Basic Block based on: https://github.com/Jongchan/attention-module/blob/master/MODELS/model_resnet.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False, geoplanes=3, cbam_reduction_ratio=2, use_depthwise_separable_conv='False', res_reduction = 'True'):
        super(BasicBlock, self).__init__()
        if res_reduction == 'True' and use_depthwise_separable_conv == 'False':
            self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        elif res_reduction == 'False' and use_depthwise_separable_conv == 'False':
            self.conv1 = conv3x3(inplanes + geoplanes, planes, stride=1)
        elif res_reduction == 'True' and use_depthwise_separable_conv == 'True': 
            #self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
            self.conv1 = Depthwise_Separable_Conv(in_planes = inplanes + geoplanes, out_planes = planes, stride = stride, padding = 1, kernel_size=3)
        elif res_reduction == 'False' and use_depthwise_separable_conv == 'True': # Depthwise_Separable_Conv layer can only be apllied if the resolution stays constant
            self.conv1 = Depthwise_Separable_Conv(in_planes = inplanes+geoplanes, out_planes = planes, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if use_depthwise_separable_conv == 'False':
            self.conv2 = conv3x3(planes+geoplanes, planes)
        elif use_depthwise_separable_conv == 'True':
            self.conv2 = Depthwise_Separable_Conv(in_planes = planes+geoplanes, out_planes = planes, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes + geoplanes, planes, stride),
                nn.BatchNorm2d(planes)
            )
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(gate_channels=planes, reduction_ratio=cbam_reduction_ratio, no_spatial=False) # reduction ratio: at channel attention it is reduced to planes /reduction ratio at the bottleneck
                
        else:
            self.cbam = None

    def forward(self, x, g1=None, g2=None):
        residual = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2, out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

# Class to implement geometry features to network
# From https://github.com/JUGGHM/PENet_ICRA2021/
class GeometryFeature(nn.Module):
    def __init__(self):
        super(GeometryFeature, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z*(0.5*h*(vnorm+1)-ch)/fh  # X = ((u-u0)*Z)/fx
        y = z*(0.5*w*(unorm+1)-cw)/fw  # Y = ((v-v0)*Z)/fy
        return torch.cat((x, y, z),1)

# From https://github.com/JUGGHM/PENet_ICRA2021/
class SparseDownSampleClose(nn.Module):
    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600
    def forward(self, d, mask):
        encode_d = - (1-mask)*self.large_number - d

        d = - self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1-mask_result)*self.large_number

        return d_result, mask_result

def deconvbnrelu(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
    return nn.Sequential(
		nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
        nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)

def DSCLdeconvbnrelu(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
    return nn.Sequential(
		#nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
		Depthwise_Separable_Conv_Transposed(in_planes = in_channels, out_planes = out_channels, kernel_size = kernel_size, padding = padding, output_padding=output_padding, stride=stride),
        nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)

def convbnrelu(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)

##########################################
##########################################
# Network implementations

class ENet(nn.Module):
    def __init__(self):
        super(ENet, self).__init__()

        # Geofeatures
        self.geofeature = GeometryFeature()
        
        # Encoder
        self.img_encoder_1_init = torch.nn.Sequential(
            nn.Conv2d(4, 32, 5, stride = 1, padding = 2, bias=False), #--> 32,352,1216  #
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        
        self.img_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_3  = BasicBlock(inplanes=64, planes=64, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_4  = BasicBlock(inplanes=64, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_5  = BasicBlock(inplanes=128, planes=128, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_6  = BasicBlock(inplanes=128, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_7  = BasicBlock(inplanes=256, planes=256, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_8  = BasicBlock(inplanes=256, planes=512, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_9  = BasicBlock(inplanes=512, planes=512, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_10 = BasicBlock(inplanes=512, planes=1024, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=16, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_11 = BasicBlock(inplanes=1024, planes=1024, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        
        # Decoder 1
        self.img_decoder_1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_6 = deconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)


        ### Depth branch
        # Encoder 2
        self.depth_encoder_1_init = torch.nn.Sequential(
            nn.Conv2d(2, 32, 5, stride = 1, padding = 2), #--> 32,352,1216  #
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.depth_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_3  = BasicBlock(inplanes=64, planes=64, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_4  = BasicBlock(inplanes=128, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_5  = BasicBlock(inplanes=128, planes=128, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_6  = BasicBlock(inplanes=256, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_7  = BasicBlock(inplanes=256, planes=256, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_8  = BasicBlock(inplanes=512, planes=512, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_9  = BasicBlock(inplanes=512, planes=512, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_10 = BasicBlock(inplanes=1024, planes=1024, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=16, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_11 = BasicBlock(inplanes=1024, planes=1024, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')

        # Decoder 2
        self.depth_decoder_1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)


    def forward(self, input):
        # Geometry features
        image = input['image']
        depth = input['depth']
        x = torch.concat((image, depth), 1)
        #d = input['input'][:,3,:,:]
        position = input['position'] # hier sind die normierten Koordinaten für u und v des Kamerakoordinatensystems abgespeichert
        K = input['K']
        unorm = position[:, 0:1, :, :] 
        vnorm = position[:, 1:2, :, :]

        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)

        valid_mask = torch.where(depth>0, torch.full_like(depth, 1.0), torch.full_like(depth, 0.0))
        d_s2, vm_s2 = self.sparsepooling(depth, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None
       
        geo_s1 = self.geofeature(depth, vnorm, unorm, params.image_dimension[0], params.image_dimension[1], c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, params.image_dimension[0] / 2, params.image_dimension[1] / 2, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, params.image_dimension[0] / 4, params.image_dimension[1] / 4, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, params.image_dimension[0] / 8, params.image_dimension[1] / 8, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, params.image_dimension[0] / 16, params.image_dimension[1] / 16, c352, c1216, f352, f1216)
        geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, params.image_dimension[0] / 32, params.image_dimension[1] / 32, c352, c1216, f352, f1216)

        ### Image branch
        # Encoder 1
        x1 = self.img_encoder_1_init(torch.concat((image, depth), dim = 1))
        x2 = self.img_encoder_2(x1, geo_s1, geo_s2)
        x3 = self.img_encoder_3(x2, geo_s2, geo_s2)
        x4 = self.img_encoder_4(x3, geo_s2, geo_s3)
        x5 = self.img_encoder_5(x4, geo_s3, geo_s3)
        x6 = self.img_encoder_6(x5, geo_s3, geo_s4)
        x7 = self.img_encoder_7(x6, geo_s4, geo_s4)
        x8 = self.img_encoder_8(x7, geo_s4, geo_s5)
        x9 = self.img_encoder_9(x8, geo_s5, geo_s5)
        x10 = self.img_encoder_10(x9, geo_s5, geo_s6)
        x11 = self.img_encoder_11(x10, geo_s6, geo_s6)

        # Decoder 1
        x1_dec = self.img_decoder_1(x11)
        x1_dec_plus = x1_dec + x9
        x2_dec = self.img_decoder_2(x1_dec_plus)
        x2_dec_plus = x2_dec + x7
        x3_dec = self.img_decoder_3(x2_dec_plus) 
        x3_dec_plus = x3_dec + x5
        x4_dec = self.img_decoder_4(x3_dec_plus)
        x4_dec_plus = x4_dec + x3
        x5_dec = self.img_decoder_5(x4_dec_plus)
        x5_dec_plus = x5_dec + x1
        x6_dec = self.img_decoder_6(x5_dec_plus) 

        rgb_depth, rgb_conf = torch.chunk(x6_dec, 2, dim=1)
        
        ### Depth branch
        # Encoder 2
        x2_1_con = torch.concat((depth, rgb_depth),1) 
        x2_2 = self.depth_encoder_1_init(x2_1_con)
        
        x2_3 = self.depth_encoder_2(x2_2, geo_s1, geo_s2)
        x2_4 = self.depth_encoder_3(x2_3, geo_s2, geo_s2)

        x2_4_con = torch.concat((x2_4, x4_dec_plus),1) # Skip Connections
        x2_5 = self.depth_encoder_4(x2_4_con, geo_s2, geo_s3)
        x2_6 = self.depth_encoder_5(x2_5, geo_s3, geo_s3)

        x2_6_con = torch.concat((x2_6, x3_dec_plus), 1) # Skip Connections
        x2_7 = self.depth_encoder_6(x2_6_con, geo_s3, geo_s4)
        x2_8 = self.depth_encoder_7(x2_7, geo_s4, geo_s4)

        x2_8_con = torch.concat((x2_8, x2_dec_plus),1) # Skip Connections
        x2_9 = self.depth_encoder_8(x2_8_con, geo_s4, geo_s5)
        x2_10 = self.depth_encoder_9(x2_9, geo_s5, geo_s5)
        x2_10_con = torch.concat((x2_10, x1_dec_plus),1) # Skip Connections
        x2_11 = self.depth_encoder_10 (x2_10_con, geo_s5, geo_s6)
        x2_12 = self.depth_encoder_11 (x2_11, geo_s6, geo_s6)
        
        # Decoder 2
        x2_1_dec = self.depth_decoder_1(x2_12 + x11)#(torch.concat((x6_1, y6),1)))
        x2_2_dec = self.depth_decoder_2(x2_1_dec + x1_dec_plus)
        x2_3_dec = self.depth_decoder_3(x2_2_dec + x2_dec_plus) 
        x2_4_dec = self.depth_decoder_4(x2_3_dec + x3_dec_plus)
        x2_5_dec = self.depth_decoder_5(x2_4_dec + x4_dec_plus)
        x2_6_dec = self.depth_decoder_6(x2_5_dec)

        d_depth, d_conf = torch.chunk(x2_6_dec, 2, dim = 1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf,d_conf), dim = 1)), 2, dim = 1)

        output = rgb_conf*rgb_depth + d_conf*d_depth

        return output, rgb_depth, d_depth 

class ENet_late_Fusion(nn.Module):
    def __init__(self):
        super(ENet_late_Fusion, self).__init__()

        # Geofeatures
        self.geofeature = GeometryFeature()
        
        # Image encoder
        self.img_encoder_1_init = torch.nn.Sequential(
            nn.Conv2d(3, 16, 5, stride = 1, padding = 2, bias=False), #--> 32,352,1216  #
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.img_encoder_2  = BasicBlock(inplanes=16, planes=32, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_3  = BasicBlock(inplanes=32, planes=32, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_4  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_5  = BasicBlock(inplanes=64, planes=64, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_6  = BasicBlock(inplanes=64, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_7  = BasicBlock(inplanes=128, planes=128, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_8  = BasicBlock(inplanes=128, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_9  = BasicBlock(inplanes=256, planes=256, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_10 = BasicBlock(inplanes=256, planes=512, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=16, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_11 = BasicBlock(inplanes=512, planes=512, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        
        # Depth encoder
        self.depth_first_encoder_1_init = torch.nn.Sequential(
            nn.Conv2d(1, 16, 5, stride = 1, padding = 2, bias=False), #--> 32,352,1216  #
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.depth_first_encoder_2  = BasicBlock(inplanes=16, planes=32, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_first_encoder_3  = BasicBlock(inplanes=32, planes=32, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_first_encoder_4  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_first_encoder_5  = BasicBlock(inplanes=64, planes=64, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_first_encoder_6  = BasicBlock(inplanes=64, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_first_encoder_7  = BasicBlock(inplanes=128, planes=128, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_first_encoder_8  = BasicBlock(inplanes=128, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_first_encoder_9  = BasicBlock(inplanes=256, planes=256, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_first_encoder_10 = BasicBlock(inplanes=256, planes=512, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_first_encoder_11 = BasicBlock(inplanes=512, planes=512, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')

         # Decoder 1
        self.img_decoder_1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_6 = deconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)
       
        # Encoder
        self.depth_encoder_1_init = torch.nn.Sequential(
            nn.Conv2d(2, 32, 5, stride = 1, padding = 2), #--> 32,352,1216  #
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.depth_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_3  = BasicBlock(inplanes=64, planes=64, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_4  = BasicBlock(inplanes=128, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_5  = BasicBlock(inplanes=128, planes=128, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_6  = BasicBlock(inplanes=256, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_7  = BasicBlock(inplanes=256, planes=256, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_8  = BasicBlock(inplanes=512, planes=512, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_9  = BasicBlock(inplanes=512, planes=512, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_10 = BasicBlock(inplanes=1024, planes=1024, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_11 = BasicBlock(inplanes=1024, planes=1024, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')

        # Decoder
        self.depth_decoder_1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

    def forward(self, input):
        image = input['image']
        depth = input['depth']
        position = input['position'] 
        K = input['K']
        unorm = position[:, 0:1, :, :] 
        vnorm = position[:, 1:2, :, :]

        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)

        valid_mask = torch.where(depth>0, torch.full_like(depth, 1.0), torch.full_like(depth, 0.0))
        d_s2, vm_s2 = self.sparsepooling(depth, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None
       
        geo_s1 = self.geofeature(depth, vnorm, unorm, params.image_dimension[0], params.image_dimension[1], c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, params.image_dimension[0] / 2, params.image_dimension[1] / 2, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, params.image_dimension[0] / 4, params.image_dimension[1] / 4, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, params.image_dimension[0] / 8, params.image_dimension[1] / 8, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, params.image_dimension[0] / 16, params.image_dimension[1] / 16, c352, c1216, f352, f1216)
        geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, params.image_dimension[0] / 32, params.image_dimension[1] / 32, c352, c1216, f352, f1216)

        # Image encoder
        x1 = self.img_encoder_1_init(image)
        x2 = self.img_encoder_2(x1, geo_s1, geo_s2)
        x3 = self.img_encoder_3(x2, geo_s2, geo_s2)
        x4 = self.img_encoder_4(x3, geo_s2, geo_s3)
        x5 = self.img_encoder_5(x4, geo_s3, geo_s3)
        x6 = self.img_encoder_6(x5, geo_s3, geo_s4)
        x7 = self.img_encoder_7(x6, geo_s4, geo_s4)
        x8 = self.img_encoder_8(x7, geo_s4, geo_s5)
        x9 = self.img_encoder_9(x8, geo_s5, geo_s5)
        x10 = self.img_encoder_10(x9, geo_s5, geo_s6)
        x11 = self.img_encoder_11(x10, geo_s6, geo_s6)
        
        # Depth encoder
        y1  = self.depth_first_encoder_1_init(depth)
        y2  = self.depth_first_encoder_2(y1, geo_s1, geo_s2)
        y3  = self.depth_first_encoder_3(y2, geo_s2, geo_s2)
        y4  = self.depth_first_encoder_4(y3, geo_s2, geo_s3)
        y5  = self.depth_first_encoder_5(y4, geo_s3, geo_s3)
        y6  = self.depth_first_encoder_6(y5, geo_s3, geo_s4)
        y7  = self.depth_first_encoder_7(y6, geo_s4, geo_s4)
        y8  = self.depth_first_encoder_8(y7, geo_s4, geo_s5)
        y9  = self.depth_first_encoder_9(y8, geo_s5, geo_s5)
        y10 = self.depth_first_encoder_10(y9, geo_s5, geo_s6)
        y11 = self.depth_first_encoder_11(y10, geo_s6, geo_s6)

        # Decoder 
        x11_con = torch.concat((x11,y11), 1)
        x1_dec = self.img_decoder_1(x11_con)
        x1_dec_plus = x1_dec + torch.concat((x9,y9), 1)
        x2_dec = self.img_decoder_2(x1_dec_plus)
        x2_dec_plus = x2_dec +  torch.concat((x7,y7), 1)
        x3_dec = self.img_decoder_3(x2_dec_plus) 
        x3_dec_plus = x3_dec +  torch.concat((x5,y5), 1)
        x4_dec = self.img_decoder_4(x3_dec_plus)
        x4_dec_plus = x4_dec +  torch.concat((x3,y3), 1)
        x5_dec = self.img_decoder_5(x4_dec_plus)
        x5_dec_plus = x5_dec +  torch.concat((x1,y1), 1)
        x6_dec = self.img_decoder_6(x5_dec_plus) 

        rgb_depth, rgb_conf = torch.chunk(x6_dec, 2, dim=1)
        
        # Encoder
        x2_1_con = torch.concat((depth, rgb_depth),1) 
        x2_2 = self.depth_encoder_1_init(x2_1_con)
        x2_3 = self.depth_encoder_2(x2_2, geo_s1, geo_s2)
        x2_4 = self.depth_encoder_3(x2_3, geo_s2, geo_s2)
        x2_4_con = torch.concat((x2_4, x4_dec_plus),1) 
        x2_5 = self.depth_encoder_4(x2_4_con, geo_s2, geo_s3)
        x2_6 = self.depth_encoder_5(x2_5, geo_s3, geo_s3)
        x2_6_con = torch.concat((x2_6, x3_dec_plus), 1) 
        x2_7 = self.depth_encoder_6(x2_6_con, geo_s3, geo_s4)
        x2_8 = self.depth_encoder_7(x2_7, geo_s4, geo_s4)
        x2_8_con = torch.concat((x2_8, x2_dec_plus),1) 
        x2_9 = self.depth_encoder_8(x2_8_con, geo_s4, geo_s5)
        x2_10 = self.depth_encoder_9(x2_9, geo_s5, geo_s5)
        x2_10_con = torch.concat((x2_10, x1_dec_plus),1)
        x2_11 = self.depth_encoder_10 (x2_10_con, geo_s5, geo_s6)
        x2_12 = self.depth_encoder_11 (x2_11, geo_s6, geo_s6)

        # Decoder
        x2_1_dec = self.depth_decoder_1(x2_12 + torch.concat((x11, y11), 1))#(torch.concat((x6_1, y6),1)))
        x2_2_dec = self.depth_decoder_2(x2_1_dec + x1_dec_plus)
        x2_3_dec = self.depth_decoder_3(x2_2_dec + x2_dec_plus) 
        x2_4_dec = self.depth_decoder_4(x2_3_dec + x3_dec_plus)
        x2_5_dec = self.depth_decoder_5(x2_4_dec + x4_dec_plus)
        x2_6_dec = self.depth_decoder_6(x2_5_dec)

        d_depth, d_conf = torch.chunk(x2_6_dec, 2, dim = 1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf,d_conf), dim = 1)), 2, dim = 1)

        output = rgb_conf*rgb_depth + d_conf*d_depth

        return output, rgb_depth, d_depth

class ENet_CBAM(nn.Module):
    def __init__(self):
        super(ENet_CBAM, self).__init__()

        # Geofeatures
        self.geofeature = GeometryFeature()
        
        # Image branch
        # Encoder 1
        self.img_encoder_1_init = torch.nn.Sequential(
            nn.Conv2d(4, 32, 5, stride = 1, padding = 2, bias=False), #--> 32,352,1216  #
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.img_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=True, cbam_reduction_ratio=2, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_3  = BasicBlock(inplanes=64, planes=64, stride=1, geoplanes=3, use_cbam=True, cbam_reduction_ratio=2, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_4  = BasicBlock(inplanes=64, planes=128, stride=2, geoplanes=3, use_cbam=True, cbam_reduction_ratio=4, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_5  = BasicBlock(inplanes=128, planes=128, stride=1, geoplanes=3, use_cbam=True, cbam_reduction_ratio=4, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_6  = BasicBlock(inplanes=128, planes=256, stride=2, geoplanes=3, use_cbam=True, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_7  = BasicBlock(inplanes=256, planes=256, stride=1, geoplanes=3, use_cbam=True, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_8  = BasicBlock(inplanes=256, planes=512, stride=2, geoplanes=3, use_cbam=True, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_9  = BasicBlock(inplanes=512, planes=512, stride=1, geoplanes=3, use_cbam=True, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_10 = BasicBlock(inplanes=512, planes=1024, stride=2, geoplanes=3, use_cbam=True, cbam_reduction_ratio=16, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_11 = BasicBlock(inplanes=1024, planes=1024, stride=1, geoplanes=3, use_cbam=True, cbam_reduction_ratio=16, use_depthwise_separable_conv='False', res_reduction='False')
        
        # Decoder 1
        self.img_decoder_1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_6 = deconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)


        # Depth branch
        # Encoder
        self.depth_encoder_1_init = torch.nn.Sequential(
            nn.Conv2d(2, 32, 5, stride = 1, padding = 2), #--> 32,352,1216  #
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.depth_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=True, cbam_reduction_ratio=2, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_3  = BasicBlock(inplanes=64, planes=64, stride=1, geoplanes=3, use_cbam=True, cbam_reduction_ratio=2, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_4  = BasicBlock(inplanes=128, planes=128, stride=2, geoplanes=3, use_cbam=True, cbam_reduction_ratio=4, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_5  = BasicBlock(inplanes=128, planes=128, stride=1, geoplanes=3, use_cbam=True, cbam_reduction_ratio=4, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_6  = BasicBlock(inplanes=256, planes=256, stride=2, geoplanes=3, use_cbam=True, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_7  = BasicBlock(inplanes=256, planes=256, stride=1, geoplanes=3, use_cbam=True, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_8  = BasicBlock(inplanes=512, planes=512, stride=2, geoplanes=3, use_cbam=True, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_9  = BasicBlock(inplanes=512, planes=512, stride=1, geoplanes=3, use_cbam=True, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_10 = BasicBlock(inplanes=1024, planes=1024, stride=2, geoplanes=3, use_cbam=True, cbam_reduction_ratio=16, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_11 = BasicBlock(inplanes=1024, planes=1024, stride=1, geoplanes=3, use_cbam=True, cbam_reduction_ratio=16, use_depthwise_separable_conv='False', res_reduction='True')

        # Decoder 1
        self.depth_decoder_1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)


    def forward(self, input):

        image = input['image']
        depth = input['depth']
        position = input['position'] 
        K = input['K']
        unorm = position[:, 0:1, :, :] 
        vnorm = position[:, 1:2, :, :]

        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)

        valid_mask = torch.where(depth>0, torch.full_like(depth, 1.0), torch.full_like(depth, 0.0))
        d_s2, vm_s2 = self.sparsepooling(depth, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None
       
        geo_s1 = self.geofeature(depth, vnorm, unorm, params.image_dimension[0], params.image_dimension[1], c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, params.image_dimension[0] / 2, params.image_dimension[1] / 2, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, params.image_dimension[0] / 4, params.image_dimension[1] / 4, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, params.image_dimension[0] / 8, params.image_dimension[1] / 8, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, params.image_dimension[0] / 16, params.image_dimension[1] / 16, c352, c1216, f352, f1216)
        geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, params.image_dimension[0] / 32, params.image_dimension[1] / 32, c352, c1216, f352, f1216)

        # Image branch
        # Encoder 1
        x1 = self.img_encoder_1_init(torch.concat((image, depth), dim = 1))
        x2 = self.img_encoder_2(x1, geo_s1, geo_s2)
        x3 = self.img_encoder_3(x2, geo_s2, geo_s2)
        x4 = self.img_encoder_4(x3, geo_s2, geo_s3)
        x5 = self.img_encoder_5(x4, geo_s3, geo_s3)
        x6 = self.img_encoder_6(x5, geo_s3, geo_s4)
        x7 = self.img_encoder_7(x6, geo_s4, geo_s4)
        x8 = self.img_encoder_8(x7, geo_s4, geo_s5)
        x9 = self.img_encoder_9(x8, geo_s5, geo_s5)
        x10 = self.img_encoder_10(x9, geo_s5, geo_s6)
        x11 = self.img_encoder_11(x10, geo_s6, geo_s6)
       
        # Decoder 1
        x1_dec = self.img_decoder_1(x11)
        x1_dec_plus = x1_dec + x9
        x2_dec = self.img_decoder_2(x1_dec_plus)
        x2_dec_plus = x2_dec + x7
        x3_dec = self.img_decoder_3(x2_dec_plus) 
        x3_dec_plus = x3_dec + x5
        x4_dec = self.img_decoder_4(x3_dec_plus)
        x4_dec_plus = x4_dec + x3
        x5_dec = self.img_decoder_5(x4_dec_plus)
        x5_dec_plus = x5_dec + x1
        x6_dec = self.img_decoder_6(x5_dec_plus) 

        rgb_depth, rgb_conf = torch.chunk(x6_dec, 2, dim=1)
        
        ### Depth branch
        # Encoder 2
        x2_1_con = torch.concat((depth, rgb_depth),1) 
        x2_2 = self.depth_encoder_1_init(x2_1_con)
        x2_3 = self.depth_encoder_2(x2_2, geo_s1, geo_s2)
        x2_4 = self.depth_encoder_3(x2_3, geo_s2, geo_s2)
        x2_4_con = torch.concat((x2_4, x4_dec_plus),1) 
        x2_5 = self.depth_encoder_4(x2_4_con, geo_s2, geo_s3)
        x2_6 = self.depth_encoder_5(x2_5, geo_s3, geo_s3)
        x2_6_con = torch.concat((x2_6, x3_dec_plus), 1) 
        x2_7 = self.depth_encoder_6(x2_6_con, geo_s3, geo_s4)
        x2_8 = self.depth_encoder_7(x2_7, geo_s4, geo_s4)
        x2_8_con = torch.concat((x2_8, x2_dec_plus),1) 
        x2_9 = self.depth_encoder_8(x2_8_con, geo_s4, geo_s5)
        x2_10 = self.depth_encoder_9(x2_9, geo_s5, geo_s5)
        x2_10_con = torch.concat((x2_10, x1_dec_plus),1) 
        x2_11 = self.depth_encoder_10 (x2_10_con, geo_s5, geo_s6)
        x2_12 = self.depth_encoder_11 (x2_11, geo_s6, geo_s6)

        # Decoder 2
        x2_1_dec = self.depth_decoder_1(x2_12 + x11)
        x2_2_dec = self.depth_decoder_2(x2_1_dec + x1_dec_plus)
        x2_3_dec = self.depth_decoder_3(x2_2_dec + x2_dec_plus) 
        x2_4_dec = self.depth_decoder_4(x2_3_dec + x3_dec_plus)
        x2_5_dec = self.depth_decoder_5(x2_4_dec + x4_dec_plus)
        x2_6_dec = self.depth_decoder_6(x2_5_dec)

        d_depth, d_conf = torch.chunk(x2_6_dec, 2, dim = 1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf,d_conf), dim = 1)), 2, dim = 1)

        output = rgb_conf*rgb_depth + d_conf*d_depth

        return output, rgb_depth, d_depth

class ENet_no_Geofeatures(nn.Module):
    def __init__(self):
        super(ENet_no_Geofeatures, self).__init__()

        # Image branch
        # Encoder 1
        self.img_encoder_1_init = torch.nn.Sequential(
            nn.Conv2d(4, 32, 5, stride = 1, padding = 2, bias=False), #--> 32,352,1216  #
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        
        self.img_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=0, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_3  = BasicBlock(inplanes=64, planes=64, stride=1, geoplanes=0, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_4  = BasicBlock(inplanes=64, planes=128, stride=2, geoplanes=0, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_5  = BasicBlock(inplanes=128, planes=128, stride=1, geoplanes=0, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_6  = BasicBlock(inplanes=128, planes=256, stride=2, geoplanes=0, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_7  = BasicBlock(inplanes=256, planes=256, stride=1, geoplanes=0, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_8  = BasicBlock(inplanes=256, planes=512, stride=2, geoplanes=0, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_9  = BasicBlock(inplanes=512, planes=512, stride=1, geoplanes=0, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.img_encoder_10 = BasicBlock(inplanes=512, planes=1024, stride=2, geoplanes=0, use_cbam=False, cbam_reduction_ratio=16, use_depthwise_separable_conv='False', res_reduction='True')
        self.img_encoder_11 = BasicBlock(inplanes=1024, planes=1024, stride=1, geoplanes=0, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        

        # Decoder 1
        self.img_decoder_1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_6 = deconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)

        # Depth branch
        # Encoder 2
        self.depth_encoder_1_init = torch.nn.Sequential(
            nn.Conv2d(2, 32, 5, stride = 1, padding = 2), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.depth_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=0, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_3  = BasicBlock(inplanes=64, planes=64, stride=1, geoplanes=0, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_4  = BasicBlock(inplanes=128, planes=128, stride=2, geoplanes=0, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_5  = BasicBlock(inplanes=128, planes=128, stride=1, geoplanes=0, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_6  = BasicBlock(inplanes=256, planes=256, stride=2, geoplanes=0, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_7  = BasicBlock(inplanes=256, planes=256, stride=1, geoplanes=0, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_8  = BasicBlock(inplanes=512, planes=512, stride=2, geoplanes=0, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_9  = BasicBlock(inplanes=512, planes=512, stride=1, geoplanes=0, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')
        self.depth_encoder_10 = BasicBlock(inplanes=1024, planes=1024, stride=2, geoplanes=0, use_cbam=False, cbam_reduction_ratio=16, use_depthwise_separable_conv='False', res_reduction='True')
        self.depth_encoder_11 = BasicBlock(inplanes=1024, planes=1024, stride=1, geoplanes=0, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='False', res_reduction='False')


        # Decoder 2
        self.depth_decoder_1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

    def forward(self, input):
        image = input['image']
        depth = input['depth']
       
        # Image branch
        # Encoder 1
        x1 = self.img_encoder_1_init(torch.concat((image, depth), dim = 1))
        x2 = self.img_encoder_2(x1)
        x3 = self.img_encoder_3(x2)
        x4 = self.img_encoder_4(x3)
        x5 = self.img_encoder_5(x4)
        x6 = self.img_encoder_6(x5)
        x7 = self.img_encoder_7(x6)
        x8 = self.img_encoder_8(x7)
        x9 = self.img_encoder_9(x8)
        x10 = self.img_encoder_10(x9)
        x11 = self.img_encoder_11(x10)

        # Decoder 1 
        x1_dec = self.img_decoder_1(x11)
        x1_dec_plus = x1_dec + x9
        x2_dec = self.img_decoder_2(x1_dec_plus)
        x2_dec_plus = x2_dec + x7
        x3_dec = self.img_decoder_3(x2_dec_plus) 
        x3_dec_plus = x3_dec + x5
        x4_dec = self.img_decoder_4(x3_dec_plus)
        x4_dec_plus = x4_dec + x3
        x5_dec = self.img_decoder_5(x4_dec_plus)
        x5_dec_plus = x5_dec + x1
        x6_dec = self.img_decoder_6(x5_dec_plus) 

        rgb_depth, rgb_conf = torch.chunk(x6_dec, 2, dim=1)
        
        # Depth branch
        # Encoder 2
        x2_1_con = torch.concat((depth, rgb_depth),1)
        x2_2 = self.depth_encoder_1_init(x2_1_con)
        x2_3 = self.depth_encoder_2(x2_2)
        x2_4 = self.depth_encoder_3(x2_3)
        x2_4_con = torch.concat((x2_4, x4_dec_plus),1) 
        x2_5 = self.depth_encoder_4(x2_4_con)
        x2_6 = self.depth_encoder_5(x2_5)
        x2_6_con = torch.concat((x2_6, x3_dec_plus), 1) 
        x2_7 = self.depth_encoder_6(x2_6_con)
        x2_8 = self.depth_encoder_7(x2_7)
        x2_8_con = torch.concat((x2_8, x2_dec_plus),1) 
        x2_9 = self.depth_encoder_8(x2_8_con)
        x2_10 = self.depth_encoder_9(x2_9)
        x2_10_con = torch.concat((x2_10, x1_dec_plus),1) 
        x2_11 = self.depth_encoder_10 (x2_10_con)
        x2_12 = self.depth_encoder_11 (x2_11)

        # Depth branch
        # Decoder 2
        x2_1_dec = self.depth_decoder_1(x2_12 + x11)
        x2_2_dec = self.depth_decoder_2(x2_1_dec + x1_dec_plus)
        x2_3_dec = self.depth_decoder_3(x2_2_dec + x2_dec_plus) 
        x2_4_dec = self.depth_decoder_4(x2_3_dec + x3_dec_plus)
        x2_5_dec = self.depth_decoder_5(x2_4_dec + x4_dec_plus)
        x2_6_dec = self.depth_decoder_6(x2_5_dec)

        d_depth, d_conf = torch.chunk(x2_6_dec, 2, dim = 1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf,d_conf), dim = 1)), 2, dim = 1)

        output = rgb_conf*rgb_depth + d_conf*d_depth
        return output, rgb_depth, d_depth

class ENet_Netz_1(nn.Module):
    def __init__(self):
        super(ENet_Netz_1, self).__init__()

        # Geofeatures
        self.geofeature = GeometryFeature()
        
        # Encoder 1
        self.img_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=4, out_planes=32, stride = 1, padding = 2, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        # Image encoder
        self.img_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_3  = BasicBlock(inplanes=64, planes=64, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='True', res_reduction='False')
        self.img_encoder_4  = BasicBlock(inplanes=64, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_5  = BasicBlock(inplanes=128, planes=128, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='True', res_reduction='False')
        self.img_encoder_6  = BasicBlock(inplanes=128, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_7  = BasicBlock(inplanes=256, planes=256, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='True', res_reduction='False')
        self.img_encoder_8  = BasicBlock(inplanes=256, planes=512, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_9  = BasicBlock(inplanes=512, planes=512, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='True', res_reduction='False')
        self.img_encoder_10 = BasicBlock(inplanes=512, planes=1024, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=16, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_11 = BasicBlock(inplanes=1024, planes=1024, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='True', res_reduction='False')
        
        # Decoder
        self.img_decoder_1 = DSCLdeconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_2 = DSCLdeconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_3 = DSCLdeconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_4 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_5 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_6 = DSCLdeconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)

        ### Depth branch
        # Encoder
        self.depth_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=2, out_planes=32, stride = 1, kernel_size=5, padding = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.depth_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_3  = BasicBlock(inplanes=64, planes=64, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='True', res_reduction='False')
        self.depth_encoder_4  = BasicBlock(inplanes=128, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_5  = BasicBlock(inplanes=128, planes=128, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='True', res_reduction='False')
        self.depth_encoder_6  = BasicBlock(inplanes=256, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_7  = BasicBlock(inplanes=256, planes=256, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='True', res_reduction='False')
        self.depth_encoder_8  = BasicBlock(inplanes=512, planes=512, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_9  = BasicBlock(inplanes=512, planes=512, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='True', res_reduction='False')
        self.depth_encoder_10 = BasicBlock(inplanes=1024, planes=1024, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=16, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_11 = BasicBlock(inplanes=1024, planes=1024, stride=1, geoplanes=3, use_cbam=False, cbam_reduction_ratio=0, use_depthwise_separable_conv='True', res_reduction='False')

        # Decoder
        self.depth_decoder_1 = DSCLdeconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_2 = DSCLdeconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_3 = DSCLdeconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_4 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_5 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)


        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)


    def forward(self, input):
        # Geometry features
        # from: https://github.com/JUGGHM/PENet_ICRA2021 
        image = input['image']
        depth = input['depth']
        position = input['position'] 
        K = input['K']
        unorm = position[:, 0:1, :, :] 
        vnorm = position[:, 1:2, :, :]

        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)

        valid_mask = torch.where(depth>0, torch.full_like(depth, 1.0), torch.full_like(depth, 0.0))
        d_s2, vm_s2 = self.sparsepooling(depth, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None
       
        geo_s1 = self.geofeature(depth, vnorm, unorm, params.image_dimension[0], params.image_dimension[1], c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, params.image_dimension[0] / 2, params.image_dimension[1] / 2, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, params.image_dimension[0] / 4, params.image_dimension[1] / 4, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, params.image_dimension[0] / 8, params.image_dimension[1] / 8, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, params.image_dimension[0] / 16, params.image_dimension[1] / 16, c352, c1216, f352, f1216)
        geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, params.image_dimension[0] / 32, params.image_dimension[1] / 32, c352, c1216, f352, f1216)

        ### Image branch
        # Encoder 1
        x1 = self.img_encoder_1_init(torch.concat((image, depth), dim = 1))
        x2 = self.img_encoder_2(x1, geo_s1, geo_s2)
        x3 = self.img_encoder_3(x2, geo_s2, geo_s2)
        x4 = self.img_encoder_4(x3, geo_s2, geo_s3)
        x5 = self.img_encoder_5(x4, geo_s3, geo_s3)
        x6 = self.img_encoder_6(x5, geo_s3, geo_s4)
        x7 = self.img_encoder_7(x6, geo_s4, geo_s4)
        x8 = self.img_encoder_8(x7, geo_s4, geo_s5)
        x9 = self.img_encoder_9(x8, geo_s5, geo_s5)
        x10 = self.img_encoder_10(x9, geo_s5, geo_s6)
        x11 = self.img_encoder_11(x10, geo_s6, geo_s6)
        
        # Decoder 1
        x1_dec = self.img_decoder_1(x11)
        x1_dec_plus = x1_dec + x9
        x2_dec = self.img_decoder_2(x1_dec_plus)
        x2_dec_plus = x2_dec + x7
        x3_dec = self.img_decoder_3(x2_dec_plus) 
        x3_dec_plus = x3_dec + x5
        x4_dec = self.img_decoder_4(x3_dec_plus)
        x4_dec_plus = x4_dec + x3
        x5_dec = self.img_decoder_5(x4_dec_plus)
        x5_dec_plus = x5_dec + x1
        x6_dec = self.img_decoder_6(x5_dec_plus) 

        rgb_depth, rgb_conf = torch.chunk(x6_dec, 2, dim=1)
        
        ### Depth branch
        # Encoder 2
        x2_1_con = torch.concat((depth, rgb_depth),1) # Concatenating the input sparse depth and the output dense depth image from the image dominante autoencoder
        x2_2 = self.depth_encoder_1_init(x2_1_con)
        x2_3 = self.depth_encoder_2(x2_2, geo_s1, geo_s2)
        x2_4 = self.depth_encoder_3(x2_3, geo_s2, geo_s2)
        x2_4_con = torch.concat((x2_4, x4_dec_plus),1) # Skip Connections
        x2_5 = self.depth_encoder_4(x2_4_con, geo_s2, geo_s3)
        x2_6 = self.depth_encoder_5(x2_5, geo_s3, geo_s3)
        x2_6_con = torch.concat((x2_6, x3_dec_plus), 1) # Skip Connections
        x2_7 = self.depth_encoder_6(x2_6_con, geo_s3, geo_s4)
        x2_8 = self.depth_encoder_7(x2_7, geo_s4, geo_s4)
        x2_8_con = torch.concat((x2_8, x2_dec_plus),1) # Skip Connections
        x2_9 = self.depth_encoder_8(x2_8_con, geo_s4, geo_s5)
        x2_10 = self.depth_encoder_9(x2_9, geo_s5, geo_s5)
        x2_10_con = torch.concat((x2_10, x1_dec_plus),1) # Skip Connections
        x2_11 = self.depth_encoder_10 (x2_10_con, geo_s5, geo_s6)
        x2_12 = self.depth_encoder_11 (x2_11, geo_s6, geo_s6)

        # Decoder 2
        x2_1_dec = self.depth_decoder_1(x2_12 + x11)#(torch.concat((x6_1, y6),1)))
        x2_2_dec = self.depth_decoder_2(x2_1_dec + x1_dec_plus)
        x2_3_dec = self.depth_decoder_3(x2_2_dec + x2_dec_plus) 
        x2_4_dec = self.depth_decoder_4(x2_3_dec + x3_dec_plus)
        x2_5_dec = self.depth_decoder_5(x2_4_dec + x4_dec_plus)
        x2_6_dec = self.depth_decoder_6(x2_5_dec)

        d_depth, d_conf = torch.chunk(x2_6_dec, 2, dim = 1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf,d_conf), dim = 1)), 2, dim = 1)

        output = rgb_conf*rgb_depth + d_conf*d_depth

        return output, rgb_depth, d_depth

class ENet_Netz_2(nn.Module):
    def __init__(self):
        super(ENet_Netz_2, self).__init__()
        # Geofeatures
        self.geofeature = GeometryFeature()
        
        # Encoder
        self.img_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=4, out_planes=32, stride = 1, padding = 2, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        # Image encoder
        self.img_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_4  = BasicBlock(inplanes=64, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_6  = BasicBlock(inplanes=128, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_8  = BasicBlock(inplanes=256, planes=512, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_10 = BasicBlock(inplanes=512, planes=1024, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=16, use_depthwise_separable_conv='True', res_reduction='True')
               
        # Decoder
        self.img_decoder_1 = DSCLdeconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_2 = DSCLdeconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_3 = DSCLdeconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_4 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_5 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_6 = DSCLdeconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)

        ### Depth branch
        # Encoder 2
        self.depth_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=2, out_planes=32, stride = 1, kernel_size=5, padding = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.depth_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_4  = BasicBlock(inplanes=128, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_6  = BasicBlock(inplanes=256, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_8  = BasicBlock(inplanes=512, planes=512, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_10 = BasicBlock(inplanes=1024, planes=1024, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=16, use_depthwise_separable_conv='True', res_reduction='True')
               
        # Decoder
        self.depth_decoder_1 = DSCLdeconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_2 = DSCLdeconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_3 = DSCLdeconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_4 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_5 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)


    def forward(self, input):
        image = input['image']
        depth = input['depth']
        position = input['position'] 
        K = input['K']
        unorm = position[:, 0:1, :, :] 
        vnorm = position[:, 1:2, :, :]

        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)

        valid_mask = torch.where(depth>0, torch.full_like(depth, 1.0), torch.full_like(depth, 0.0))
        d_s2, vm_s2 = self.sparsepooling(depth, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None
       
        geo_s1 = self.geofeature(depth, vnorm, unorm, params.image_dimension[0], params.image_dimension[1], c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, params.image_dimension[0] / 2, params.image_dimension[1] / 2, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, params.image_dimension[0] / 4, params.image_dimension[1] / 4, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, params.image_dimension[0] / 8, params.image_dimension[1] / 8, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, params.image_dimension[0] / 16, params.image_dimension[1] / 16, c352, c1216, f352, f1216)
        geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, params.image_dimension[0] / 32, params.image_dimension[1] / 32, c352, c1216, f352, f1216)

        ### Image branch
        # Encoder 1
        x1 = self.img_encoder_1_init(torch.concat((image, depth), dim = 1))
        x2 = self.img_encoder_2(x1, geo_s1, geo_s2)
        x4 = self.img_encoder_4(x2, geo_s2, geo_s3)
        x6 = self.img_encoder_6(x4, geo_s3, geo_s4)
        x8 = self.img_encoder_8(x6, geo_s4, geo_s5)
        x10 = self.img_encoder_10(x8, geo_s5, geo_s6)

        # Decoder 1
        x1_dec = self.img_decoder_1(x10)
        x1_dec_plus = x1_dec + x8
        x2_dec = self.img_decoder_2(x1_dec_plus)
        x2_dec_plus = x2_dec + x6
        x3_dec = self.img_decoder_3(x2_dec_plus) 
        x3_dec_plus = x3_dec + x4
        x4_dec = self.img_decoder_4(x3_dec_plus)
        x4_dec_plus = x4_dec + x2
        x5_dec = self.img_decoder_5(x4_dec_plus)
        x5_dec_plus = x5_dec + x1
        x6_dec = self.img_decoder_6(x5_dec_plus) 

        rgb_depth, rgb_conf = torch.chunk(x6_dec, 2, dim=1)
        
        ### Depth branch
        # Encoder 2
        x2_1_con = torch.concat((depth, rgb_depth),1) # Concatenating the input sparse depth and the output dense depth image from the image dominante autoencoder
        x2_2 = self.depth_encoder_1_init(x2_1_con)
        x2_3 = self.depth_encoder_2(x2_2, geo_s1, geo_s2)
        x2_4_con = torch.concat((x2_3, x4_dec_plus),1) # Skip Connections
        x2_5 = self.depth_encoder_4(x2_4_con, geo_s2, geo_s3)
        x2_6_con = torch.concat((x2_5, x3_dec_plus), 1) # Skip Connections
        x2_7 = self.depth_encoder_6(x2_6_con, geo_s3, geo_s4)
        x2_8_con = torch.concat((x2_7, x2_dec_plus),1) # Skip Connections
        x2_9 = self.depth_encoder_8(x2_8_con, geo_s4, geo_s5)
        x2_10_con = torch.concat((x2_9, x1_dec_plus),1) # Skip Connections
        x2_11 = self.depth_encoder_10 (x2_10_con, geo_s5, geo_s6)
        
        # Decoder 2
        x2_1_dec = self.depth_decoder_1(x2_11 + x10)#orch.concat((x6_1, y6),1)))
        x2_2_dec = self.depth_decoder_2(x2_1_dec + x1_dec_plus)
        x2_3_dec = self.depth_decoder_3(x2_2_dec + x2_dec_plus) 
        x2_4_dec = self.depth_decoder_4(x2_3_dec + x3_dec_plus)
        x2_5_dec = self.depth_decoder_5(x2_4_dec + x4_dec_plus)
        x2_6_dec = self.depth_decoder_6(x2_5_dec)

        d_depth, d_conf = torch.chunk(x2_6_dec, 2, dim = 1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf,d_conf), dim = 1)), 2, dim = 1)

        output = rgb_conf*rgb_depth + d_conf*d_depth

        return output, rgb_depth, d_depth


class ENet_Netz_3(nn.Module):
    def __init__(self):
        super(ENet_Netz_3, self).__init__()

        # Geofeatures
        self.geofeature = GeometryFeature()
        
        # Image Branch
        # Encoder 1
        self.img_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=4, out_planes=32, stride = 1, padding = 2, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.img_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_4  = BasicBlock(inplanes=64, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_6  = BasicBlock(inplanes=128, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_8  = BasicBlock(inplanes=256, planes=512, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
       

        # Decoder 2
        self.img_decoder_2 = DSCLdeconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_3 = DSCLdeconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_4 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_5 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_6 = DSCLdeconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)
        
        ### Depth branch
        # Encoder 1
        self.depth_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=2, out_planes=32, stride = 1, kernel_size=5, padding = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.depth_encoder_2  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_4  = BasicBlock(inplanes=128, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_6  = BasicBlock(inplanes=256, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_8  = BasicBlock(inplanes=512, planes=512, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
  
        # Decoder 2
        self.depth_decoder_2 = DSCLdeconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_3 = DSCLdeconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_4 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_5 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        
        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)


    def forward(self, input):
        image = input['image']
        depth = input['depth']
        position = input['position'] 
        K = input['K']
        unorm = position[:, 0:1, :, :] 
        vnorm = position[:, 1:2, :, :]

        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        
        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        
        valid_mask = torch.where(depth>0, torch.full_like(depth, 1.0), torch.full_like(depth, 0.0))
        d_s2, vm_s2 = self.sparsepooling(depth, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        
        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
               
        geo_s1 = self.geofeature(depth, vnorm, unorm, params.image_dimension[0], params.image_dimension[1], c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, params.image_dimension[0] / 2, params.image_dimension[1] / 2, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, params.image_dimension[0] / 4, params.image_dimension[1] / 4, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, params.image_dimension[0] / 8, params.image_dimension[1] / 8, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, params.image_dimension[0] / 16, params.image_dimension[1] / 16, c352, c1216, f352, f1216)
        
        # Image branch
        # Encoder
        x1 = self.img_encoder_1_init(torch.concat((image, depth), dim = 1))
        x2 = self.img_encoder_2(x1, geo_s1, geo_s2)
        x4 = self.img_encoder_4(x2, geo_s2, geo_s3)
        x6 = self.img_encoder_6(x4, geo_s3, geo_s4)
        x8 = self.img_encoder_8(x6, geo_s4, geo_s5)
                
        # Depth Encoder
        x2_dec = self.img_decoder_2(x8)
        x2_dec_plus = x2_dec + x6
        x3_dec = self.img_decoder_3(x2_dec_plus) 
        x3_dec_plus = x3_dec + x4
        x4_dec = self.img_decoder_4(x3_dec_plus)
        x4_dec_plus = x4_dec + x2
        x5_dec = self.img_decoder_5(x4_dec_plus)
        x5_dec_plus = x5_dec + x1
        x6_dec = self.img_decoder_6(x5_dec_plus) 

        rgb_depth, rgb_conf = torch.chunk(x6_dec, 2, dim=1)
        
        # Depth branch
        # Encoder
        x2_1_con = torch.concat((depth, rgb_depth),1) 
        x2_2 = self.depth_encoder_1_init(x2_1_con)
        x2_3 = self.depth_encoder_2(x2_2, geo_s1, geo_s2)
        x2_4_con = torch.concat((x2_3, x4_dec_plus),1) 
        x2_5 = self.depth_encoder_4(x2_4_con, geo_s2, geo_s3)
        x2_6_con = torch.concat((x2_5, x3_dec_plus), 1) 
        x2_7 = self.depth_encoder_6(x2_6_con, geo_s3, geo_s4)
        x2_8_con = torch.concat((x2_7, x2_dec_plus),1) 
        x2_9 = self.depth_encoder_8(x2_8_con, geo_s4, geo_s5)
       
        # Decoder
        x2_2_dec = self.depth_decoder_2(x2_9 + x8)
        x2_3_dec = self.depth_decoder_3(x2_2_dec + x2_dec_plus) 
        x2_4_dec = self.depth_decoder_4(x2_3_dec + x3_dec_plus)
        x2_5_dec = self.depth_decoder_5(x2_4_dec + x4_dec_plus)
        x2_6_dec = self.depth_decoder_6(x2_5_dec)

        d_depth, d_conf = torch.chunk(x2_6_dec, 2, dim = 1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf,d_conf), dim = 1)), 2, dim = 1)

        output = rgb_conf*rgb_depth + d_conf*d_depth

        return output, rgb_depth, d_depth

class ENet_Netz_4(nn.Module):
    def __init__(self):
        super(ENet_Netz_4, self).__init__()

        # Geofeatures
        self.geofeature = GeometryFeature()
        
        # Image branch  
        # Encoder 1
        self.img_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=4, out_planes=16, stride = 1, padding = 2, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.img_encoder_2  = BasicBlock(inplanes=16, planes=32, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_4  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_6  = BasicBlock(inplanes=64, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_8  = BasicBlock(inplanes=128, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
 
        # Decoder 1
        self.img_decoder_2 = DSCLdeconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_3 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_4 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_5 = DSCLdeconvbnrelu(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_6 = DSCLdeconvbnrelu(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)


        ### Depth branch
        # Encoder 2
        self.depth_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=2, out_planes=16, stride = 1, kernel_size=5, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.depth_encoder_2  = BasicBlock(inplanes=16, planes=32, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_4  = BasicBlock(inplanes=64, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_6  = BasicBlock(inplanes=128, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_8  = BasicBlock(inplanes=256, planes=256, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
       
        # Decoder 2
        self.depth_decoder_2 = DSCLdeconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_3 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_4 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_5 = DSCLdeconvbnrelu(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_6 = convbnrelu(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)


    def forward(self, input):
        image = input['image']
        depth = input['depth']
        position = input['position'] 
        K = input['K']
        unorm = position[:, 0:1, :, :] 
        vnorm = position[:, 1:2, :, :]

        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)

        valid_mask = torch.where(depth>0, torch.full_like(depth, 1.0), torch.full_like(depth, 0.0))
        d_s2, vm_s2 = self.sparsepooling(depth, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
       
        geo_s1 = self.geofeature(depth, vnorm, unorm, params.image_dimension[0], params.image_dimension[1], c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, params.image_dimension[0] / 2, params.image_dimension[1] / 2, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, params.image_dimension[0] / 4, params.image_dimension[1] / 4, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, params.image_dimension[0] / 8, params.image_dimension[1] / 8, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, params.image_dimension[0] / 16, params.image_dimension[1] / 16, c352, c1216, f352, f1216)

        # Image branch
        # Encoder 1
        x1 = self.img_encoder_1_init(torch.concat((image, depth), dim = 1))
        x2 = self.img_encoder_2(x1, geo_s1, geo_s2)
        x4 = self.img_encoder_4(x2, geo_s2, geo_s3)
        x6 = self.img_encoder_6(x4, geo_s3, geo_s4)
        x8 = self.img_encoder_8(x6, geo_s4, geo_s5)
       
       # Decoder 1
        x2_dec = self.img_decoder_2(x8)
        x2_dec_plus = x2_dec + x6
        x3_dec = self.img_decoder_3(x2_dec_plus) 
        x3_dec_plus = x3_dec + x4
        x4_dec = self.img_decoder_4(x3_dec_plus)
        x4_dec_plus = x4_dec + x2
        x5_dec = self.img_decoder_5(x4_dec_plus)
        x5_dec_plus = x5_dec + x1
        x6_dec = self.img_decoder_6(x5_dec_plus) 

        rgb_depth, rgb_conf = torch.chunk(x6_dec, 2, dim=1)
        
        # Depth branch
        # Encoder 2
        x2_1_con = torch.concat((depth, rgb_depth),1) 
        x2_2 = self.depth_encoder_1_init(x2_1_con)
        x2_3 = self.depth_encoder_2(x2_2, geo_s1, geo_s2)
        x2_4_con = torch.concat((x2_3, x4_dec_plus),1) # Skip Connections
        x2_5 = self.depth_encoder_4(x2_4_con, geo_s2, geo_s3)
        x2_6_con = torch.concat((x2_5, x3_dec_plus), 1) # Skip Connections
        x2_7 = self.depth_encoder_6(x2_6_con, geo_s3, geo_s4)
        x2_8_con = torch.concat((x2_7, x2_dec_plus),1) # Skip Connections
        x2_9 = self.depth_encoder_8(x2_8_con, geo_s4, geo_s5)
       
        # Decoder 2
        x2_2_dec = self.depth_decoder_2(x2_9 + x8)
        x2_3_dec = self.depth_decoder_3(x2_2_dec + x2_dec_plus) 
        x2_4_dec = self.depth_decoder_4(x2_3_dec + x3_dec_plus)
        x2_5_dec = self.depth_decoder_5(x2_4_dec + x4_dec_plus)
        x2_6_dec = self.depth_decoder_6(x2_5_dec)

        d_depth, d_conf = torch.chunk(x2_6_dec, 2, dim = 1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf,d_conf), dim = 1)), 2, dim = 1)

        output = rgb_conf*rgb_depth + d_conf*d_depth

        return output, rgb_depth, d_depth


class ENet_Netz_5(nn.Module):
    def __init__(self):
        super(ENet_Netz_5, self).__init__()

        # Geofeatures
        self.geofeature = GeometryFeature()
        
        # Image branch
        # Encoder 1
        self.img_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=4, out_planes=8, stride = 1, padding = 2, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        self.img_encoder_2  = BasicBlock(inplanes=8, planes=16, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_4  = BasicBlock(inplanes=16, planes=32, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_6  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_8  = BasicBlock(inplanes=64, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
      
        # Decoder 1
        self.img_decoder_2 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_3 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_4 = DSCLdeconvbnrelu(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_5 = DSCLdeconvbnrelu(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.img_decoder_6 = DSCLdeconvbnrelu(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)

        # Depth branch
        # Encoder 2
        self.depth_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=2, out_planes=8, stride = 1, kernel_size=5, padding = 2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))

        self.depth_encoder_2  = BasicBlock(inplanes=8, planes=16, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_4  = BasicBlock(inplanes=32, planes=32, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_6  = BasicBlock(inplanes=64, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_8  = BasicBlock(inplanes=128, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        
        # Decoder
        self.depth_decoder_2 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_3 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_4 = DSCLdeconvbnrelu(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_5 = DSCLdeconvbnrelu(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_6 = convbnrelu(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

    def forward(self, input):
        image = input['image']
        depth = input['depth']
        position = input['position'] 
        K = input['K']
        unorm = position[:, 0:1, :, :] 
        vnorm = position[:, 1:2, :, :]

        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)

        valid_mask = torch.where(depth>0, torch.full_like(depth, 1.0), torch.full_like(depth, 0.0))
        d_s2, vm_s2 = self.sparsepooling(depth, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
       
        geo_s1 = self.geofeature(depth, vnorm, unorm, params.image_dimension[0], params.image_dimension[1], c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, params.image_dimension[0] / 2, params.image_dimension[1] / 2, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, params.image_dimension[0] / 4, params.image_dimension[1] / 4, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, params.image_dimension[0] / 8, params.image_dimension[1] / 8, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, params.image_dimension[0] / 16, params.image_dimension[1] / 16, c352, c1216, f352, f1216)
       
        ### Image branch
        # Encoder 1
        x1 = self.img_encoder_1_init(torch.concat((image, depth), dim = 1))
        x2 = self.img_encoder_2(x1, geo_s1, geo_s2)
        x4 = self.img_encoder_4(x2, geo_s2, geo_s3)
        x6 = self.img_encoder_6(x4, geo_s3, geo_s4)
        x8 = self.img_encoder_8(x6, geo_s4, geo_s5)
        
        # Decoder 2 
        x2_dec = self.img_decoder_2(x8)
        x2_dec_plus = x2_dec + x6
        x3_dec = self.img_decoder_3(x2_dec_plus) 
        x3_dec_plus = x3_dec + x4
        x4_dec = self.img_decoder_4(x3_dec_plus)
        x4_dec_plus = x4_dec + x2
        x5_dec = self.img_decoder_5(x4_dec_plus)
        x5_dec_plus = x5_dec + x1
        x6_dec = self.img_decoder_6(x5_dec_plus) 

        rgb_depth, rgb_conf = torch.chunk(x6_dec, 2, dim=1)
        
        ### Depth branch
        # Encoder 2
        x2_1_con = torch.concat((depth, rgb_depth),1) 
        x2_2 = self.depth_encoder_1_init(x2_1_con)
        x2_3 = self.depth_encoder_2(x2_2, geo_s1, geo_s2)
        x2_4_con = torch.concat((x2_3, x4_dec_plus),1) 
        x2_5 = self.depth_encoder_4(x2_4_con, geo_s2, geo_s3)
        x2_6_con = torch.concat((x2_5, x3_dec_plus), 1) 
        x2_7 = self.depth_encoder_6(x2_6_con, geo_s3, geo_s4)
        x2_8_con = torch.concat((x2_7, x2_dec_plus),1) 
        x2_9 = self.depth_encoder_8(x2_8_con, geo_s4, geo_s5)
        
        # Decoder 2
        x2_2_dec = self.depth_decoder_2(x2_9 + x8)
        x2_3_dec = self.depth_decoder_3(x2_2_dec + x2_dec_plus) 
        x2_4_dec = self.depth_decoder_4(x2_3_dec + x3_dec_plus)
        x2_5_dec = self.depth_decoder_5(x2_4_dec + x4_dec_plus)
        x2_6_dec = self.depth_decoder_6(x2_5_dec)
        
        d_depth, d_conf = torch.chunk(x2_6_dec, 2, dim = 1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf,d_conf), dim = 1)), 2, dim = 1)

        output = rgb_conf*rgb_depth + d_conf*d_depth

        return output, rgb_depth, d_depth

class ENet_Netz_6(nn.Module):
    def __init__(self):
        super(ENet_Netz_6, self).__init__()

        # Geofeatures
        self.geofeature = GeometryFeature()
        
        # Image branch
        # Encoder 1
        self.img_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=4, out_planes=8, stride = 1, padding = 1, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        self.img_encoder_2  = BasicBlock(inplanes=8, planes=16, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_4  = BasicBlock(inplanes=16, planes=32, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_6  = BasicBlock(inplanes=32, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.img_encoder_8  = BasicBlock(inplanes=64, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        
        # Decoder
        self.img_decoder_2 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.img_decoder_3 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.img_decoder_4 = DSCLdeconvbnrelu(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.img_decoder_5 = DSCLdeconvbnrelu(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.img_decoder_6 = DSCLdeconvbnrelu(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)

        ### Depth branch
        # Encoder 1
        self.depth_encoder_1_init = torch.nn.Sequential(
            Depthwise_Separable_Conv(in_planes=2, out_planes=8, stride = 1, kernel_size=3, padding = 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        self.depth_encoder_2  = BasicBlock(inplanes=8, planes=16, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=2, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_4  = BasicBlock(inplanes=32, planes=32, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=4, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_6  = BasicBlock(inplanes=64, planes=64, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        self.depth_encoder_8  = BasicBlock(inplanes=128, planes=128, stride=2, geoplanes=3, use_cbam=False, cbam_reduction_ratio=8, use_depthwise_separable_conv='True', res_reduction='True')
        
        # Decoder 2
        self.depth_decoder_2 = DSCLdeconvbnrelu(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.depth_decoder_3 = DSCLdeconvbnrelu(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.depth_decoder_4 = DSCLdeconvbnrelu(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.depth_decoder_5 = DSCLdeconvbnrelu(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.depth_decoder_6 = convbnrelu(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)


    def forward(self, input):
        image = input['image']
        depth = input['depth']
        position = input['position'] 
        K = input['K']
        unorm = position[:, 0:1, :, :] 
        vnorm = position[:, 1:2, :, :]

        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
     
        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        
        valid_mask = torch.where(depth>0, torch.full_like(depth, 1.0), torch.full_like(depth, 0.0))
        d_s2, vm_s2 = self.sparsepooling(depth, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
       
        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
           
        geo_s1 = self.geofeature(depth, vnorm, unorm, params.image_dimension[0], params.image_dimension[1], c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, params.image_dimension[0] / 2, params.image_dimension[1] / 2, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, params.image_dimension[0] / 4, params.image_dimension[1] / 4, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, params.image_dimension[0] / 8, params.image_dimension[1] / 8, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, params.image_dimension[0] / 16, params.image_dimension[1] / 16, c352, c1216, f352, f1216)
        
        # Image branch
        # Encoder 1
        x1 = self.img_encoder_1_init(torch.concat((image, depth), dim = 1))
        x2 = self.img_encoder_2(x1, geo_s1, geo_s2)
        x4 = self.img_encoder_4(x2, geo_s2, geo_s3)
        x6 = self.img_encoder_6(x4, geo_s3, geo_s4)
        x8 = self.img_encoder_8(x6, geo_s4, geo_s5)
        
        # Decoder 1
        x2_dec = self.img_decoder_2(x8)
        x2_dec_plus = x2_dec + x6
        x3_dec = self.img_decoder_3(x2_dec_plus) 
        x3_dec_plus = x3_dec + x4
        x4_dec = self.img_decoder_4(x3_dec_plus)
        x4_dec_plus = x4_dec + x2
        x5_dec = self.img_decoder_5(x4_dec_plus)
        x5_dec_plus = x5_dec + x1
        x6_dec = self.img_decoder_6(x5_dec_plus) 

        rgb_depth, rgb_conf = torch.chunk(x6_dec, 2, dim=1)
        
        ### Depth branch
        # Encoder 2
        x2_1_con = torch.concat((depth, rgb_depth),1) 
        x2_2 = self.depth_encoder_1_init(x2_1_con)
        x2_3 = self.depth_encoder_2(x2_2, geo_s1, geo_s2)
        x2_4_con = torch.concat((x2_3, x4_dec_plus),1) 
        x2_5 = self.depth_encoder_4(x2_4_con, geo_s2, geo_s3)
        x2_6_con = torch.concat((x2_5, x3_dec_plus), 1) 
        x2_7 = self.depth_encoder_6(x2_6_con, geo_s3, geo_s4)
        x2_8_con = torch.concat((x2_7, x2_dec_plus),1) 
        x2_9 = self.depth_encoder_8(x2_8_con, geo_s4, geo_s5)
        
        # Decoder 2
        x2_2_dec = self.depth_decoder_2(x2_9 + x8)
        x2_3_dec = self.depth_decoder_3(x2_2_dec + x2_dec_plus) 
        x2_4_dec = self.depth_decoder_4(x2_3_dec + x3_dec_plus)
        x2_5_dec = self.depth_decoder_5(x2_4_dec + x4_dec_plus)
        x2_6_dec = self.depth_decoder_6(x2_5_dec)
        
        d_depth, d_conf = torch.chunk(x2_6_dec, 2, dim = 1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf,d_conf), dim = 1)), 2, dim = 1)

        output = rgb_conf*rgb_depth + d_conf*d_depth

        return output, rgb_depth, d_depth