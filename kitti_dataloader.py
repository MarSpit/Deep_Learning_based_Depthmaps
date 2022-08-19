from turtle import shape
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from os import listdir
from PIL import Image
import params
import helper

# Function that generates a list with the paths to training / evaluation data
def file_list(dir_kitti_raw, root_dir_gt, root_dir_projected_pointcloud):
    input_img_days = listdir(dir_kitti_raw)
    gt_runs = listdir(root_dir_gt)
    if not input_img_days or not gt_runs: # Check if list contains folders
        print('No files found. Check paths to data.')
        exit()
    list_with_all_files = []
    # Loop over dates: 2011_09_26, 2011_09_28, 2011_09_29...
    for days in input_img_days:
        if os.path.isdir(dir_kitti_raw + days):
            input_img_runs = listdir(dir_kitti_raw + days)
            if len (input_img_runs) == 0: # Check if list contains folders
                print('No files found. Check paths to data.')
                exit()
            # Loop over runs: 2011_09_26_drive_0001_sync, 2011_09_26_drive_0002_sync ...
            for runs in input_img_runs: 
                # Check if runs is a folder. Is necessary, as the kitti calibration_files.txt are located at this level
                if runs in gt_runs and os.path.isdir(dir_kitti_raw + days + '/' + runs)==True and os.path.isdir(root_dir_projected_pointcloud + runs)==True: 
                    input_images = listdir(dir_kitti_raw + days + '/' + runs + '/image_02/data/')
                    gt_images = listdir(root_dir_gt + runs + '/proj_depth/groundtruth/image_02/')
                    projected_pointclouds = listdir(root_dir_projected_pointcloud + runs + '/proj_depth/velodyne_raw/image_02/')
                    # Check if list contains files
                    if len(input_images) == 0 or len(gt_images)==0 or len(projected_pointclouds)==0: 
                        print('No files found. Check paths to data.')
                        exit()
                    for gt_image in gt_images:
                        # check if corresponding gt, image and pointcloud exist
                        if gt_image in input_images and gt_image in projected_pointclouds:
                            list_with_all_files.append([dir_kitti_raw + days + '/' + runs + '/image_02/data/' + gt_image, # path to images
                                                        root_dir_gt + runs + '/proj_depth/groundtruth/image_02/' + gt_image, # path to gt
                                                        #root_dir_projected_pointcloud + days + '/' + runs + '/image_02/data/' + gt_image[:-3] + 'npy' # path to pointcloud, as gt_image has the .png ending, it is removed and the bin ending is added
                                                        root_dir_projected_pointcloud + runs + '/proj_depth/velodyne_raw/image_02/' + gt_image # path to pointcloud, as gt_image has the .png ending, it is removed and the bin ending is added
                                                        ])
        if len(list_with_all_files) == 0:
            print('No files found. Check paths to data.')  
            exit() 

    return list_with_all_files           

class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    """
    def __call__(self, img):
        """Convert a ``numpy.ndarray`` to tensor.
        Args:
            img (numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if not (helper.is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if isinstance(img, np.ndarray):
            # handle numpy array
            if img.ndim == 3:
                img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
            elif img.ndim == 2:
                img = torch.from_numpy(img.copy())
            else:
                raise RuntimeError(
                    'img should be ndarray with 2 or 3 dimensions. Got {}'.
                    format(img.ndim))

            return img

to_tensor = ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

# Kitti dataset class
class KITTI_Dataset(Dataset):
    def __init__(self, list_with_all_files, transform_img = None, transform_depth = None, transform_position = None):
        #super().__init__()
        self.list_with_all_files = list_with_all_files 
        self.transform_img = transform_img
        self.transform_depth = transform_depth
        self.transform_position = transform_position
        self.K = helper.load_calib(path = params.dir_kitti_raw)
            
    def __getitem__(self, index):
        image_png = Image.open(self.list_with_all_files[index][0]).convert('RGB')
        gt_png = Image.open(self.list_with_all_files[index][1])
        projected_pointcloud_png = Image.open(self.list_with_all_files[index][2])
        position = helper.AddCoordsNp(params.image_dimension[0], params.image_dimension[1])
        position = position.call()
        K = self.K
        
        if self.transform_img:
            image = np.array(image_png)
            image = self.transform_img(image)
        
        if self.transform_depth:
            gt = np.array(gt_png, dtype = int)
            assert np.max(gt_png) > 255, \
                "np.max(depth_png)={}, path={}".format(np.max(gt_png), self.list_with_all_files[index][1])
            gt = gt.astype(np.float) / 256.
            gt = np.expand_dims(gt, -1)
            gt = self.transform_depth(gt)

        if self.transform_depth:
            projected_pointcloud = np.array(projected_pointcloud_png, dtype=int)
            assert np.max(gt_png) > 255, \
                "np.max(depth_png)={}, path={}".format(np.max(projected_pointcloud_png), self.list_with_all_files[index][2])
            projected_pointcloud = projected_pointcloud.astype(np.float) / 256.
            projected_pointcloud = np.expand_dims(projected_pointcloud, -1)
            projected_pointcloud = self.transform_depth(projected_pointcloud)

        if self.transform_position:
            position = self.transform_position(position)

        # Create dict 
        data = {'image': image,
                'depth': projected_pointcloud,
                'gt': gt,
                'position': position,
                'K': K}

        data = {
            key: to_float_tensor(val)
            for key, val in data.items() if val is not None
        }

        image_png.close()
        gt_png.close()
        projected_pointcloud_png.close()

        return data

    def __len__(self):
        return len(self.list_with_all_files)
        
