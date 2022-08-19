from numpy import average
import torch
from kitti_dataloader import KITTI_Dataset, file_list
from torch.utils.data import DataLoader
import math
import time
from torch.autograd import Variable
from tqdm import tqdm

import ENet_models
import MSG_CHNet_models
import PyD_Net2_models
import params as params
import helper
import transform


if __name__=="__main__":

    device = torch.device(params.device) 
    if device == 'cuda':
        torch.cuda.empty_cache()

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
            model = ENet_models.ENet_Netz_6().to(device, dtype = torch.float) # Architecture of netz 6 and 7 is identical
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
 
    print('Network has:', helper.count_parameters(model), 'parameters.')
    
    # Define loss
    criterion = helper.MaskedMSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr = params.learning_rate, weight_decay = params.weight_decay)

    # Generate list with the paths to training data
    list_with_all_files_train = file_list(dir_kitti_raw = params.dir_kitti_raw, root_dir_gt = params.dir_gt_train, root_dir_projected_pointcloud=params.dir_projected_pointcloud_train ) # Create list with the pathes to all input images and gt images
    train_dataset = KITTI_Dataset(list_with_all_files=list_with_all_files_train, transform_img = transform.transform_img_train, transform_depth = transform.transform_depth_train, transform_position = transform.transform_position)#transform_img=transforms.ToTensor())
    print('Total number of image gt pointcloud pairs for training: ', len(list_with_all_files_train))

    # Generate list with the paths to evaluation data
    list_with_all_files_eval = file_list(dir_kitti_raw = params.dir_kitti_raw, root_dir_gt = params.dir_gt_eval, root_dir_projected_pointcloud=params.dir_projected_pointcloud_eval ) # Create list with the pathes to all input images and gt images
    val_dataset = KITTI_Dataset(list_with_all_files=list_with_all_files_eval, transform_img = transform.transform_img_val, transform_depth = transform.transform_depth_val, transform_position = transform.transform_position)#transform_img=transforms.ToTensor())
    print("Total number of image gt pointcloud pairs for evaluation: ", len(list_with_all_files_eval))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size = params.batch_size, shuffle = True, num_workers=params.num_workers_train, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size = params.batch_size, shuffle = False, num_workers=params.num_workers_eval, pin_memory=True)
    total_samples_train = len(train_dataset) 
    total_samples_eval = len(val_dataset)
    n_iterations_train = math.ceil(total_samples_train/params.batch_size) # math.ceil rounds the result
    n_iterations_eval = math.ceil(total_samples_eval/params.batch_size) # math.ceil rounds the result

    running_loss = 0.0 # Tracks the summed loss for the training
    losses = []
    start_training = time.time()
    training_steps = 0 # tracks the total number of traing steps done over all epochs
    eval_average_loss_list = []
    for epoch in range (params.num_epochs):
        start_epoch_time = time.time()
        loop = tqdm(enumerate(train_loader), total=len(train_loader)) # tqdm is needed for visualization of the training progess in the terminal
        # Adaptive Learning rate
        if epoch == 0:
            learning_rate = params.learning_rate
        learning_rate = helper.adaptive_lr(epoch, learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = params.weight_decay)
        
        for i, batch_data in loop:
            training_steps += 1
            model.train()
            
            batch_data = {
                key: Variable(val.to(device, dtype = torch.float))
                for key, val in batch_data.items() if val is not None
            }
         
            gt = batch_data['gt']
            gt = gt.to(device, dtype = torch.float)
            gt = Variable(gt)

            ### Forward pass
            # Loss calculation for ENet models according to: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch
            if params.network_family == 'ENet':
                output, rgb_depth, d_depth = model(batch_data)
                
                rgb_loss, depth_loss, loss = 0, 0, 0
                w_st1, w_st2 = 0, 0
                round1, round2, round3 = 2, 4, None
                if(epoch <= round1):
                    w_st1, w_st2 = 0.2, 0.2
                elif(epoch <= round2):
                    w_st1, w_st2 = 0.05, 0.05
                else:
                    w_st1, w_st2 = 0, 0

                depth_loss = criterion(output, gt)
                if w_st1 != 0: 
                    st1_loss = criterion(rgb_depth, gt)
                    st2_loss = criterion(d_depth, gt)
                    loss = (1 - w_st1 - w_st2) * depth_loss + w_st1 * st1_loss + w_st2 * st2_loss
                elif w_st1 == 0:
                    loss = criterion(output, gt)

            # Loss calculation for MSG-CHNet according to: https://github.com/anglixjtu/msg_chn_wacv20
            elif params.network_family == 'MSG_CHNet':
                if params.selected_model == 'MSG_CHNet_32' or params.selected_model == 'MSG_CHNet_64' or params.selected_model == 'MSG_CHNet_Netz_1':
                    output_d11, output_d12, output_d14 = model(batch_data)
                    loss1 = criterion(output_d11, gt)
                    loss2 = criterion(output_d12, gt)
                    loss3 = criterion(output_d14, gt)
                    loss = loss1 + loss2 + loss3
                elif params.selected_model == 'MSG_CHNet_Netz_2':
                    output_d11, output_d12 = model(batch_data)
                    loss1 = criterion(output_d11, gt)
                    loss2 = criterion(output_d12, gt)
                    loss = loss1 + loss2
                elif params.selected_model == 'MSG_CHNet_Netz_3' or params.selected_model == 'MSG_CHNet_Netz_4':
                    output_d11 = model(batch_data)
                    loss = criterion(output_d11, gt)

            elif params.network_family == 'PyD_Net2':
                output = model(batch_data)
                loss = criterion(output, gt)

            loss_cpu = loss

            losses.append(loss_cpu.cpu().detach().numpy())
            running_loss += loss_cpu

            ### Backward pass and optimizer    
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()
            
            # Update tqdm progess bar
            loop.set_description(f'Epoch [{epoch}/{params.num_epochs}]')
            loop.set_postfix(loss = loss_cpu.item())

            # Save model after every 50th batch und upload values to wandb
            if training_steps != 0 and training_steps % 50 == 0 :
                torch.save(model.to('cpu').state_dict(), params.name_saved_model)
                model.to(device, dtype = torch.float)
            
                            
        # Evaluating model after every epoch 
        model.eval()
        with torch.no_grad():
            running_eval_loss = 0
            total_val_samples = val_loader
            loop_eval = tqdm(enumerate(val_loader), total=len(val_loader)) # tqdm is needed for visualization of the training progess in the terminal
            for k, eval_batch_data in loop_eval:
                eval_batch_data = {
                    key: Variable(val.to(device, dtype = torch.float))
                    for key, val in eval_batch_data.items() if val is not None
                }
                #input = Variable(input)
                eval_gt = eval_batch_data['gt']
                eval_gt = eval_gt.to(device, dtype = torch.float)
                eval_gt = Variable(eval_gt)

                ### Forward pass
              
                if params.network_family == 'ENet':
                    output, rgb_depth, d_depth = model(batch_data)
                    
                    rgb_loss, depth_loss, loss = 0, 0, 0
                    w_st1, w_st2 = 0, 0
                    round1, round2, round3 = 2, 4, None
                    if(epoch <= round1):
                        w_st1, w_st2 = 0.2, 0.2
                    elif(epoch <= round2):
                        w_st1, w_st2 = 0.05, 0.05
                    else:
                        w_st1, w_st2 = 0, 0

                    depth_loss = criterion(output, eval_gt)
                    if w_st1 != 0: 
                        st1_loss = criterion(rgb_depth, eval_gt)
                        st2_loss = criterion(d_depth, eval_gt)
                        loss = (1 - w_st1 - w_st2) * depth_loss + w_st1 * st1_loss + w_st2 * st2_loss
                    elif w_st1 == 0:
                        loss = criterion(output, eval_gt)
                
                elif params.network_family == 'MSG_CHNet':
                    if params.selected_model == 'MSG_CHNet_32' or params.selected_model == 'MSG_CHNet_64' or params.selected_model == 'MSG_CHNet_Netz_1':
                        output_d11, output_d12, output_d14 = model(batch_data)
                        loss1 = criterion(output_d11, eval_gt)
                        loss2 = criterion(output_d12, eval_gt)
                        loss3 = criterion(output_d14, eval_gt)
                        loss = loss1 + loss2 + loss3
                        running_eval_loss += loss
                    elif params.selected_model == 'MSG_CHNet_Netz_2':
                        output_d11, output_d12 = model(batch_data)
                        loss1 = criterion(output_d11, eval_gt)
                        loss2 = criterion(output_d12, eval_gt)
                        loss = loss1 + loss2
                        running_eval_loss += loss
                    elif params.selected_model == 'MSG_CHNet_Netz_3' or params.selected_model == 'MSG_CHNet_Netz_4':
                        output_d11 = model(batch_data)
                        loss = criterion(output_d11, eval_gt)
                elif params.network_family == 'PyD_Net2':
                    output = model(batch_data)
                    loss = criterion(output, gt)

                running_eval_loss += loss

                # Update tqdm progess bar
                loop_eval.set_postfix(loss = loss.item())
                
                model.to(device, dtype = torch.float)

                #break

        eval_average_loss = running_eval_loss / (k+1) 
        eval_average_loss_list.append(eval_average_loss)

        print(f'Evaluation loss after {epoch+1} epoch(s): {eval_average_loss}')
            
        stop_epoch_time = time.time()
        print('Time for one training epoch: ', stop_epoch_time-start_epoch_time, 'sec.')


        # Save model after each epoch
        torch.save(model.to('cpu').state_dict(), params.name_saved_model)
        model.to(device)
        print('Model saved.')

    stop_training = time.time()
    print('Training done.')
    print('Time for total training    : ', stop_training-start_training, 'sec.')


 