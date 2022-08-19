import numbers
import torchvision.transforms as transforms
import params
import numpy as np

from helper import is_numpy_image, is_pil_image

# form https://github.com/JUGGHM/PENet_ICRA2021

class BottomCrop(object):
    """Crops the given ``numpy.ndarray`` at the bottom.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for bottom crop.
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for bottom crop.
        """
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = h - th
        j = int(round((w - tw) / 2.))

        # randomized left and right cropping
        # i = np.random.randint(i-3, i+4)
        # j = np.random.randint(j-1, j+1)

        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
        Returns:
            img (numpy.ndarray (C x H x W)): Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)
        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not (is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[i:i + h, j:j + w, :]
        elif img.ndim == 2:
            return img[i:i + h, j:j + w]
        else:
            raise RuntimeError(
                'img should be ndarray with 2 or 3 dimensions. Got {}'.format(
                    img.ndim))

class HorizontalFlip(object):
    """Horizontally flip the given ``numpy.ndarray``.
    Args:
        do_flip (boolean): whether or not do horizontal flip.
    """
    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be flipped.
        Returns:
            img (numpy.ndarray (C x H x W)): flipped image.
        """
        if not (is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if self.do_flip:
            return np.fliplr(img)
        else:
            return img

transform_img_train = transforms.Compose([
    BottomCrop(params.image_dimension),
    HorizontalFlip(np.random.uniform(0.0, 1.0) < 0.5)])
    
transform_img_val = transforms.Compose([
    BottomCrop(params.image_dimension)])

transform_depth_train = transforms.Compose([
    BottomCrop(params.image_dimension),
    HorizontalFlip(np.random.uniform(0.0, 1.0) < 0.5)])

transform_depth_val = transforms.Compose([
    BottomCrop(params.image_dimension)])

transform_position = transforms.Compose([
    BottomCrop(params.image_dimension)])

