# Select device type
device = 'cuda' # select 'cuda' or 'cpu'

### Select Network for evaluation 
network_family = 'MSG_CHNet' # Select 'ENet', 'MSG_CHNet', or 'PyD_Net2'
selected_model = 'MSG_CHNet_Netz_4' # Select specific model according to readme file

# Number of data samples for runtime evaluation
warmup_cycles = 100 # Amount of data the CPU / GPU warm up is done with
eval_cycles = 100 # Amount of data the evaluation is done with --> Maximum is 3426 due to amount of data in the KITTI eval dataset 

### Training_
# Hyperparameters
learning_rate = 1e-3 # initial learning rate
weight_decay = 1e-6 
num_epochs = 2 # Number of training epochs. (One epoch has 43.000 datapairs)
name_saved_model = 'xyz.pth' # Name of newly trained model

# Further parameters 
batch_size = 1
num_workers_train = 20 # Number of processes on the CPU, that are preparing the data for the network (on the GPU)
num_workers_eval = 20  # Number of processes on the CPU, that are preparing the data for the network (on the GPU)
image_dimension = (352, 1216) # Amount of pixels input data is cropped to 

# Pathes to dirctory with Kitti data 
dir_kitti_depth = '/media/_Perception_Datasets/Kitti/kitti_depth/' # path to kitti depth directory 
dir_kitti_raw =   '/media/_Perception_Datasets/Kitti/kitti_raw/' # path to kitti raw directory 

dir_gt_train = dir_kitti_depth + 'data_depth_annotated/train/' # path to kitti depth directory where the ground truth training data is stored
dir_gt_eval = dir_kitti_depth + 'data_depth_annotated/val/' # path to kitti depth directory where the ground truth evaluation data is stored
dir_projected_pointcloud_train = dir_kitti_depth + 'data_depth_velodyne/train/'# path to kitti depth directory where the projected pointclouds for training are stored
dir_projected_pointcloud_eval = dir_kitti_depth + 'data_depth_velodyne/val/' # path to kitti depth directory where the projected pointclouds for evaluation are stored

# In case you want to visualize one specific depth image uncomment the vis_output() function in the main function of the evaluation.py file
# Pathes for visualizing one specific image 
img_number = '0000000005.png'
img_path_vis = '/media/_Perception_Datasets/Kitti/kitti_raw/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/' 
gt_path_vis =  '/media/_Perception_Datasets/Kitti/kitti_depth/data_depth_annotated/val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/' 
projected_pointcloud_path_vis = '/media/_Perception_Datasets/Kitti/kitti_depth/data_depth_velodyne/val/2011_09_26_drive_0002_sync/proj_depth/velodyne_raw/image_02/' 




