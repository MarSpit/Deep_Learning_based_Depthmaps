#import torch
import torch.nn as nn
import numpy as np
#import numbers
try:
    import accimage
except ImportError:
    accimage = None
from PIL import Image

# From https://github.com/JUGGHM/PENet_ICRA2021/
# A normal loss can not be used, because the gt is sparse and we dont want to include the pixels in the image to the loss calculation, where no gt is provided (value is 0) 
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss

# Function to load intrinsic camera calibration matrix
def load_calib(path):
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open(path + "2011_09_26/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    # K[0, 2] = K[0, 2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    # K[1, 2] = K[1, 2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    K[0, 2] = K[0, 2] - 13;
    K[1, 2] = K[1, 2] - 11.5;
    return K

# From https://github.com/JUGGHM/PENet_ICRA2021/
class AddCoordsNp():
	"""Add coords to a tensor"""
	def __init__(self, x_dim=64, y_dim=64, with_r=False):
		self.x_dim = x_dim
		self.y_dim = y_dim
		self.with_r = with_r

	def call(self):
		"""
		input_tensor: (batch, x_dim, y_dim, c)
		"""

		xx_ones = np.ones([self.x_dim], dtype=np.int32)
		xx_ones = np.expand_dims(xx_ones, 1)


		xx_range = np.expand_dims(np.arange(self.y_dim), 0)

		xx_channel = np.matmul(xx_ones, xx_range)
		xx_channel = np.expand_dims(xx_channel, -1)

		yy_ones = np.ones([self.y_dim], dtype=np.int32)
		yy_ones = np.expand_dims(yy_ones, 0)

		yy_range = np.expand_dims(np.arange(self.x_dim), 1)

		yy_channel = np.matmul(yy_range, yy_ones)
		yy_channel = np.expand_dims(yy_channel, -1)

		xx_channel = xx_channel.astype('float32') / (self.y_dim - 1)
		yy_channel = yy_channel.astype('float32') / (self.x_dim - 1)

		xx_channel = xx_channel*2 - 1
		yy_channel = yy_channel*2 - 1

		ret = np.concatenate([xx_channel, yy_channel], axis=-1)

		if self.with_r:
			rr = np.sqrt( np.square(xx_channel-0.5) + np.square(yy_channel-0.5))
			ret = np.concatenate([ret, rr], axis=-1)

		return ret

# Function for adopting the learning rate after x epochs
def adaptive_lr(epoch, lr):
	if epoch == 1:
		lr_new = lr * 0.1
		print('New learning rate:', lr_new)
	if epoch == 4:
		lr_new = lr * 0.2
		print('New learning rate:', lr_new)
	if epoch == 10:
		lr_new = lr * 0.1
		print('New learning rate:', lr_new)
	if epoch == 14:
		lr_new = lr * 0.01
		print('New learning rate:', lr_new)
	if epoch == 20:
		lr_new = lr * 0.01
		print('New learning rate:', lr_new)
	if epoch == 30:
		lr_new = lr * 0.01
		print('New learning rate:', lr_new)
	if epoch == 40:
		lr_new = lr * 0.01
		print('New learning rate:', lr_new)
	if epoch == 50:
		lr_new = lr * 0.01
		print('New learning rate:', lr_new)
	if epoch == 60:
		lr_new = lr * 0.01
		print('New learning rate:', lr_new)
	else:
		lr_new = lr
	return lr_new

def is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

# Counts parameters of network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)