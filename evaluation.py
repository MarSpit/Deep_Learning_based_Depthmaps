import torch
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import time
import math
from sys import exit
from torch.profiler import profile, record_function, ProfilerActivity

from kitti_dataloader import KITTI_Dataset, file_list, to_float_tensor
import params
import ENet_models
import MSG_CHNet_models
import PyD_Net2_models
import helper
import transform


# Function that evaluates the runtime the network needs for calculating the depth images
def runtime_eval(network_family, selected_model, dir_kitti_raw, root_dir_gt, root_dir_projected_pointcloud, warmup_cycles = 50, eval_cycles = 100, batch_size = 1, num_workers = 1):
    print('Running runtime evaluation.')
    # Evaluation of network performance on 100 image pointcloud pairs
    list_with_all_files = file_list(dir_kitti_raw = dir_kitti_raw, root_dir_gt = root_dir_gt, root_dir_projected_pointcloud = root_dir_projected_pointcloud) # Create list with the pathes to all input images and gt images
    list_with_all_files_warm_up = list_with_all_files[:warmup_cycles]
    list_with_all_files_eval = list_with_all_files[:eval_cycles] # Eval dataset counts 3426 files in total. Using every 34 file to cover all sequences --> 101 files
    dataset_KITTI_warm_up = KITTI_Dataset(list_with_all_files = list_with_all_files_warm_up, transform_img = transform.transform_img_val, transform_depth = transform.transform_depth_val, transform_position = transform.transform_position)
    dataset_KITTI_eval = KITTI_Dataset(list_with_all_files = list_with_all_files_eval, transform_img = transform.transform_img_val, transform_depth = transform.transform_depth_val, transform_position = transform.transform_position)
    warm_up_loader = DataLoader(dataset = dataset_KITTI_warm_up, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = num_workers)
    val_loader = DataLoader(dataset = dataset_KITTI_eval, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = num_workers)
    
    print("Selected number of image gt pointcloud pairs for evaluation: ", len(list_with_all_files_eval))
    model.eval()    
    with torch.no_grad():
       # GPU warm-up
        print ('Doing GPU/CPU warm-up....')
        for i, batch_data_warm_up in enumerate(warm_up_loader):
            batch_data_warm_up = {
                key: Variable(val.to(device, dtype=torch.float))
                for key, val in batch_data_warm_up.items() if val is not None
            }
            if network_family == 'ENet' or (network_family == 'MSG_CHNet' and selected_model == 'MSG_CHNet_Netz_1'):
                output, output_1, output_2  = model(batch_data_warm_up)
                        
            elif network_family == 'MSG_CHNet' and (selected_model == 'MSG_CHNet_Netz_2'):
                output, output_1 = model(batch_data_warm_up)

            elif network_family == 'PyD_Net2' or (network_family == 'MSG_CHNet' and (selected_model == 'MSG_CHNet_Netz_3' or selected_model == 'MSG_CHNet_Netz_4')):
                output = model(batch_data_warm_up)
        
        print ('Doing runtime evaluation....')
        # Measuring the total time needed per depth image (not just calculation time)
        time_list = []
        start_time_total = time.time()
        for i, batch_data_val in enumerate(val_loader):
            batch_data_val = {
                key: Variable(val.to(device, dtype=torch.float))
                for key, val in batch_data_val.items() if val is not None
            }
            if network_family == 'ENet' or (network_family == 'MSG_CHNet' and selected_model == 'MSG_CHNet_Netz_1'):
                output, output_1, output_2  = model(batch_data_val)
                        
            elif network_family == 'MSG_CHNet' and (selected_model == 'MSG_CHNet_Netz_2'):
                output, output_1 = model(batch_data_val)

            elif network_family == 'PyD_Net2' or (network_family == 'MSG_CHNet' and (selected_model == 'MSG_CHNet_Netz_3' or selected_model == 'MSG_CHNet_Netz_4')):
                output = model(batch_data_val)
        total_time = time.time()-start_time_total
        print('Average total time per depth image:', total_time/(i+1), 's')
        print('Frames per second:', 1/(total_time/(i+1)))

        # Measuring the time needed for the pure calculation of the network on the GPU / CPU
        time_list = []
        for i, batch_data_val in enumerate(val_loader):
            #print('Running', i, 'sample from', len(list_with_all_files_warm_up))
            batch_data_val = {
                key: Variable(val.to(device, dtype=torch.float))
                for key, val in batch_data_val.items() if val is not None
            }
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    if device.type == 'cpu':
                        start_time_network_calculation = time.time()    

                    if network_family == 'ENet' or (network_family == 'MSG_CHNet' and selected_model == 'MSG_CHNet_Netz_1'):
                        output, output_1, output_2  = model(batch_data_val)
                        
                    elif network_family == 'MSG_CHNet' and (selected_model == 'MSG_CHNet_Netz_2'):
                        output, output_1 = model(batch_data_val)

                    elif network_family == 'PyD_Net2' or (network_family == 'MSG_CHNet' and (selected_model == 'MSG_CHNet_Netz_3' or selected_model == 'MSG_CHNet_Netz_4')):
                        output = model(batch_data_val)
                    
                    if device.type == 'cpu':
                        time_list.append(time.time() - start_time_network_calculation)

            if device.type == 'cuda':
                print(prof.key_averages().table(sort_by = "cuda_time_total", row_limit = 1))
        if device.type == 'cpu':    
            print ('Average time for calculatiing one depth image on CPU:', sum(time_list)/(i+1), 's')


# Visualizes the depth image a network returns and the corresponding input image with overlayed pointcloud for one specific scene        
def vis_output(img_path_vis, img_number, gt_path_vis, projected_pointcloud_path_vis, dir_kitti_raw):
    model.eval()
    with torch.no_grad(): 
        img_png = Image.open(img_path_vis + img_number).convert('RGB')
        gt_png = Image.open(gt_path_vis + img_number)
        projected_pointcloud_png = Image.open(projected_pointcloud_path_vis + img_number)
        position = helper.AddCoordsNp(params.image_dimension[0], params.image_dimension[1])
        position = position.call()
        K = helper.load_calib(path = dir_kitti_raw)
        
        if transform.transform_depth_val:
            gt_arr = np.array(gt_png, dtype = int)
            gt_arr = gt_arr.astype(float) / 256.
            gt = transform.transform_depth_val(gt_arr)

        if transform.transform_depth_val:
            projected_pointcloud_arr = np.array(projected_pointcloud_png, dtype = int)
            projected_pointcloud_arr = projected_pointcloud_arr.astype(float) / 256.
            projected_pointcloud = transform.transform_depth_val(projected_pointcloud_arr)
         
        if transform.transform_img_val:
            img_arr = np.array(img_png)
            img = transform.transform_img_val(img_arr)
            
        batch_data = {
            'image': img,
            'depth': projected_pointcloud,
            'gt': gt,
            'position': position,
            'K': K}

        batch_data = {
            key: to_float_tensor(val)
            for key, val in batch_data.items() if val is not None
        }

        # Adding batch / colour channel 
        batch_data ['image'] = batch_data ['image'].unsqueeze_(0)
        batch_data ['depth'] = batch_data ['depth'].unsqueeze_(0).unsqueeze_(0)
        batch_data ['gt'] = batch_data ['gt'].unsqueeze_(0).unsqueeze_(0)
        batch_data ['position'] = batch_data ['position'].unsqueeze_(0)
        batch_data ['K'] = batch_data ['K'].unsqueeze_(0)

        batch_data = {
                key: Variable(val.to(params.device, dtype = torch.float))
                for key, val in batch_data.items() if val is not None
            }

        img_png.close()
        gt_png.close()
        projected_pointcloud_png.close()

        if params.network_family == 'ENet' or (params.network_family == 'MSG_CHNet' and params.selected_model == 'MSG_CHNet_Netz_1'):
            output, output_1, output_2  = model(batch_data)
                
        if params.network_family == 'MSG_CHNet' and (params.selected_model == 'MSG_CHNet_Netz_2'):
            output, output_1 = model(batch_data)

        if params.network_family == 'PyD_Net2' or (params.network_family == 'MSG_CHNet' and (params.selected_model == 'MSG_CHNet_Netz_3' or params.selected_model == 'MSG_CHNet_Netz_4')):
            output = model(batch_data)
              
        if str(device) == 'cuda':
            output = output.to('cpu')
        output.squeeze_(0).squeeze_(0) # remove unnecessary dimensions
        output_arr = output.detach().numpy()
        
        # Creating a list with the depth values and the u and v pixels to be able to plot them
        point_lst = []
        size = projected_pointcloud.shape
        for i in range(size[0]):
            for j in range (size[1]):
                if projected_pointcloud[i][j] != 0:# and pc[i][j] <= 85:
                    point_lst.append((i, j, projected_pointcloud[i][j]))
        # Plotting
        fig, axs = plt.subplots(4,1, constrained_layout = True)
        
        # Plot input image
        axs[0].imshow(img)
        axs[0].set_title('Kamerabild mit überlagertem spärlichem Tiefenbild', fontsize = 20)
        axs[0].axis('off')       
        x,y,depth = zip(*point_lst)
        axs0 = axs[0].scatter(y, x, s = 0.2,c = depth, cmap = 'turbo', vmin = 0, vmax = 80)
        cb0 = fig.colorbar(axs0, orientation="vertical", ax = axs[0])
        cb0.set_label(label = 'Tiefe in m', size = 17)
        cb0.ax.tick_params(labelsize = 17)
        
        # Plot predicted depth
        axs1 = axs[1].imshow(output_arr, cmap = 'turbo', vmin = 0, vmax = 80)#cmap='gist_rainbow')
        axs[1].set_title('Prädiziertes dichtes Tiefenbild ENet 3', fontsize = 20)
        cb1 = fig.colorbar(axs1, orientation="vertical", ax = axs[1])
        cb1.set_label(label = 'Tiefe in m', size = 17)
        cb1.ax.tick_params(labelsize = 17)
        axs[1].axis('off')
                
        # Plot GT depth
        axs2 = axs[3].imshow(gt, cmap = 'turbo',  vmin = 0, vmax = 80 )
        axs[3].set_title('Ground Truth', fontsize=20)
        cb2 = fig.colorbar(axs2, orientation = "vertical", ax = axs[3])
        cb2.set_label(label = 'Tiefe in m', size = 17)
        cb2.ax.tick_params(labelsize = 17)
        axs[3].axis('off')

        # Plot errormap
        out_arr_adapted = np.where(gt == 0, gt, output_arr) # Setting the predicted values to 0 where the Gt is invalid / has 0
        errormap = np.absolute(out_arr_adapted - gt)
        axs3 = axs[2].imshow(errormap, cmap = 'turbo', vmin = 0, vmax = 10)
        axs[2].set_title('Fehler der Tiefenprädiktion', fontsize = 20)
        cb3 = fig.colorbar(axs3, orientation = "vertical", ax = axs[2])
        cb3.set_label(label = 'Tiefe in m', size = 17)
        cb3.ax.tick_params(labelsize = 17)
        axs[2].axis('off')
        plt.show()

        # Calculate RMSE and MAE for selected image
        MAE = np.sum(abs(errormap))/np.count_nonzero(errormap) 
        RMSE = math.sqrt(np.sum(errormap**2)/np.count_nonzero(errormap))          
        print('The RMSE of the visualizes predicted depth is:   ', RMSE, 'meter(s)')
        print('The MAE of the visualized predicted depth is:    ', MAE, 'meter(s).')
        
# Evaluation of the accuracy of 100 depth images
def accuracy_eval(network_family, selected_model, dir_kitti_raw, root_dir_gt, root_dir_projected_pointcloud, batch_size = 1, num_workers = 1):
    model.eval()
    print('Running depthmap accuracy evaluation.')
    list_with_all_files_eval = file_list(dir_kitti_raw, root_dir_gt, root_dir_projected_pointcloud) # Create list with the pathes to all input images, sparse depth maps and the gt
    list_with_all_files_eval = list_with_all_files_eval[::34]
    dataset_KITTI_eval = KITTI_Dataset(list_with_all_files = list_with_all_files_eval, transform_img = transform.transform_img_val, transform_depth = transform.transform_depth_val, transform_position = transform.transform_position)
    print("Total number of image gt pointcloud pairs for evaluation: ", len(list_with_all_files_eval))
    val_loader = DataLoader(dataset = dataset_KITTI_eval, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers= num_workers)
    RMSE_list = []
    MAE_list = []

    with torch.no_grad():
        for i, eval_batch_data in enumerate(val_loader):
            print('Running', i, 'sample.')
            eval_batch_data = {
                key: Variable(val.to(device, dtype=torch.float))
                for key, val in eval_batch_data.items() if val is not None
            }

            if network_family == 'ENet' or (network_family == 'MSG_CHNet' and selected_model == 'MSG_CHNet_Netz_1'):
                output, output_1, output_2  = model(eval_batch_data)
                
            if network_family == 'MSG_CHNet' and (selected_model == 'MSG_CHNet_Netz_2'):
                output, output_1 = model(eval_batch_data)

            if network_family == 'PyD_Net2' or (network_family == 'MSG_CHNet' and (selected_model == 'MSG_CHNet_Netz_3' or selected_model == 'MSG_CHNet_Netz_4')):
                output = model(eval_batch_data)
                        
            if str(device) == 'cuda':
                output = output.to('cpu')
            output.squeeze_(0).squeeze_(0) # remove unnecessary dimensions
            
            pred_arr = output.detach().numpy()
            gt = eval_batch_data['gt']
            gt_arr = gt.detach().cpu().numpy()
            out_arr_depth_adapted = np.where(gt_arr == 0, gt_arr, pred_arr) # Setting the predicted values to 0 where the Gt is invalid / has 0
            errormap = np.absolute(out_arr_depth_adapted - gt_arr)
            MAE = np.sum(abs(errormap))/np.count_nonzero(errormap) 
            RMSE = math.sqrt(np.sum(errormap**2)/np.count_nonzero(errormap))
            MAE_list.append(MAE)
            RMSE_list.append(RMSE)

    average_RSME = sum(RMSE_list)/(i+1)
    print('Average RMSE for  ', i+1, 'sample depth predictions on eval dataset:', average_RSME, 'm.')
    print('Highest RMSE among', i+1, 'sample depth predictions on eval dataset:', max(RMSE_list), 'm.')
    print('Lowest RMSE among ', i+1, 'sample depth predictions on eval dataset:', min(RMSE_list), 'm.')
    average_MAE = sum(MAE_list)/(i+1)
    print('Average MAE for   ', i+1, 'sample depth predictions on eval dataset:', average_MAE, 'm.')
    print('Highest MAE among ', i+1, 'sample depth predictions on eval dataset:', max(MAE_list), 'm.')
    print('Lowest MAE among  ', i+1, 'sample depth predictions on eval dataset:', min(MAE_list), 'm.')

if __name__=="__main__":
    device = torch.device(params.device) 

    ### Load network architecture
    # Enet
    if params.network_family == 'ENet':
        if params.selected_model == 'ENet':
            model = ENet_models.ENet().to(device, dtype = torch.float)
        elif params.selected_model == 'ENet_Netz_1':
            model = ENet_models.ENet_Netz_1().to(device, dtype = torch.float)
        elif params.selected_model == 'ENet_Netz_2':
            model = ENet_models.ENet_Netz_2().to(device, dtype = torch.float)
        elif params.selected_model == 'ENet_Netz_3':
            model = ENet_models.ENet_Netz_3().to(device, dtype = torch.float)
        elif params.selected_model == 'ENet_Netz_4':
            model = ENet_models.ENet_Netz_4().to(device, dtype = torch.float)
        elif params.selected_model == 'ENet_Netz_5':
            model = ENet_models.ENet_Netz_5().to(device, dtype = torch.float)
        elif params.selected_model == 'ENet_Netz_6':
            model = ENet_models.ENet_Netz_6().to(device, dtype = torch.float)
        elif params.selected_model == 'ENet_Netz_7':
            model = ENet_models.ENet_Netz_6().to(device, dtype = torch.float) # Architecture of ENet netz 6 and 7 is identical
        elif params.selected_model == 'ENet_CBAM':
            model = ENet_models.ENet_CBAM().to(device, dtype = torch.float)
        elif params.selected_model == 'ENet_no_Geofeatures':
            model = ENet_models.ENet_no_Geofeatures().to(device, dtype = torch.float)
        elif params.selected_model == 'ENet_late_Fusion':
            model = ENet_models.ENet_late_Fusion().to(device, dtype = torch.float)
        else:
            print('Invalid model name. Please check variable eval_model in params file.')
            exit()
    # MSG_CHNet
    elif params.network_family == 'MSG_CHNet':
        if params.selected_model == 'MSG_CHNet_64':
            model = MSG_CHNet_models.MSG_CHNet_64().to(device, dtype = torch.float)
        elif params.selected_model == 'MSG_CHNet_32':
            model = MSG_CHNet_models.MSG_CHNet_32().to(device, dtype = torch.float)
        elif params.selected_model == 'MSG_CHNet_Netz_1':
            model = MSG_CHNet_models.MSG_CHNet_Netz_1().to(device, dtype = torch.float)
        elif params.selected_model == 'MSG_CHNet_Netz_2':
            model = MSG_CHNet_models.MSG_CHNet_Netz_2().to(device, dtype = torch.float)
        elif params.selected_model == 'MSG_CHNet_Netz_3':
            model = MSG_CHNet_models.MSG_CHNet_Netz_3().to(device, dtype = torch.float)
        elif params.selected_model == 'MSG_CHNet_Netz_4':
            model = MSG_CHNet_models.MSG_CHNet_Netz_4().to(device, dtype = torch.float)
        else:
            print('Invalid model name. Please check variable eval_model in params file.')
            exit()
    # PyD_Net2
    elif params.network_family == 'PyD_Net2':
        if params.selected_model == 'PyD_Net2_Netz_1':
            model = PyD_Net2_models.PyD_Net2_Netz_1().to(device, dtype = torch.float)
        elif params.selected_model == 'PyD_Net2_Netz_2':
            model = PyD_Net2_models.PyD_Net2_Netz_2().to(device, dtype = torch.float)
        elif params.selected_model == 'PyD_Net2_Netz_3':
            model = PyD_Net2_models.PyD_Net2_Netz_3().to(device, dtype = torch.float)
        elif params.selected_model == 'PyD_Net2_Netz_4':
            model = PyD_Net2_models.PyD_Net2_Netz_4().to(device, dtype = torch.float)
        elif params.selected_model == 'PyD_Net2_Netz_5':
            model = PyD_Net2_models.PyD_Net2_Netz_5().to(device, dtype = torch.float)
        else:
            print('Invalid model name. Please check variable eval_model in params file.')
            exit()
    else:
        print('Invalid model family name. Please check variable network_family in params file.')
        exit()

    # Print number of parameters / weights of network:
    print('Selected network:', params.selected_model)
    print('Network has', helper.count_parameters(model), 'parameters.')
    
    # Load pretrained weights for selected model
    path_to_eval_model = 'Trained_Networks/' + params.selected_model + '.pth'
    assert model.load_state_dict(torch.load(path_to_eval_model, map_location=torch.device(device))), 'Model not found. Process cancelled!'
    
    #Functions to evaluate network
    # This function visualizes depth image and input data for defined Kitti image in params.py file
    #vis_output(dir_kitti_raw = params.dir_kitti_raw, img_path_vis = params.img_path_vis, img_number = params.img_number, gt_path_vis = params.gt_path_vis, \
    #   projected_pointcloud_path_vis = params.projected_pointcloud_path_vis) 
    # This function runs the network on the whole KITTI evaluation dataset and return average RMSE and MAE values
    accuracy_eval(network_family = params.network_family, selected_model = params.selected_model, dir_kitti_raw = params.dir_kitti_raw, root_dir_gt = params.dir_gt_eval, \
        root_dir_projected_pointcloud = params.dir_projected_pointcloud_eval, batch_size = params.batch_size, num_workers = params.num_workers_eval)
    
    # This function runs the network on the selected amount of input data and measures times
    runtime_eval(network_family = params.network_family, selected_model = params.selected_model, warmup_cycles = params.warmup_cycles, eval_cycles = params.eval_cycles, batch_size = params.batch_size,  \
        num_workers = params.num_workers_eval, dir_kitti_raw = params.dir_kitti_raw, root_dir_gt = params.dir_gt_eval, root_dir_projected_pointcloud = params.dir_projected_pointcloud_eval)
    


