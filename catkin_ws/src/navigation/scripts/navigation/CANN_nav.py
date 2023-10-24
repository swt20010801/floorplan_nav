#!/usr/bin/env /home/swt/anaconda3/envs/agent/bin/python
from dataset.unity_dataset import unityDataset
from dataset.unity_dataset import persp2pano

import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from model.fplocnet import FpLocNet, quick_fplocnet_call
from eval_utils import *
import os
import rospy

import argparse   #步骤一
from utils.dijkstra_planner import Dijkstra
from model.Action_Policy import ONNXModel
from test_parameters.parameters import house_node_list_dict,house_color_id_dict,house_node_toplogic_dict,house_node_toplogic_edge_dict
from utils.env_actions import reset,step_,set_robot_position,get_robot_position
from utils.particle_filter import init_particles_given_coords,align_particles,update_particles,resample,move_particles
import pandas as pd
import matplotlib.pyplot as plt

from CANN.pose_cell import pose_cell
from CANN.hd_cell import hd_cell

def parse_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"                   
    parser = argparse.ArgumentParser(description=description)        
                                                                 
    help = "The path of address"
    parser.add_argument('--House_ID',type=int,help="0 as Start")  
    parser.add_argument('--save_trajectory',action='store_true')
    parser.add_argument('--save_estimated_trajectory',action='store_true')    
    parser.add_argument('--save_noisyodometry_trajectory',action='store_true') 

    parser.add_argument('--save_data_forloc',action='store_true')         

    parser.add_argument('--mode',choices=['GT',"FP_LOC","NOISY"])         
    parser.add_argument('--wo_particle_filter',action='store_true')         
    parser.add_argument('--wo_topo_nav',action='store_true')         


    args = parser.parse_args() 
    return args

def image_2_pano(image,fov=np.pi/2,image_size=256):
    """
    image : Tensor(4,3,h,w)
    """
    image=image.permute(0,2,3,1).cpu().numpy()


    image_0=image[0,:,:,:]
    image_1=image[1,:,:,:]
    image_2=image[2,:,:,:]
    image_3=image[3,:,:,:]

    pano=[]
    pano.append(persp2pano(image_0,fov,image_size))
    pano.append(persp2pano(image_1,fov,image_size))
    pano.append(persp2pano(image_2,fov,image_size))
    pano.append(persp2pano(image_3,fov,image_size))
    image=np.concatenate(pano,axis=1)
    #pano的最左边（首位）对应着机器人正面的朝向
    image=np.concatenate([image[:,image_size//2:,:],image[:,:image_size//2,:],],axis=1)
    
    # cv2.imwrite("pano.jpg",image)

    image=image/255.
    image -= (0.485, 0.456, 0.406)
    image /= (0.229, 0.224, 0.225)

    image=np.transpose(image, (2, 0, 1))
    image=torch.from_numpy(image).unsqueeze(0)
    return image

def render_map(bases, bases_feat,eval_dataset,house_id,border=100):#scale：1m对应多少个像素点
    n_bases = bases.shape[0]
    bases = np.copy(bases)
    # affine = (border - bases.min(axis=0), scale)
    # bases = (bases + affine[0]) * affine[1]
    # loc_gt = (loc_gt + affine[0]) * affine[1]#单位变成了像素
    # loc_est = (loc_est + affine[0]) * affine[1]
    for i in range(n_bases):
        bases[i]=eval_dataset.meter2pixel(bases[i],house_id)

    W, H = np.ptp(bases, axis=0).astype(np.int32) + int(2 * border )
    canvas = 255*np.ones((H, W, 3), np.uint8)

    door_label = bases_feat[:, -2]
    window_label = bases_feat[:, -1]
    for i in range(n_bases):
        color = [60, 60, 60]
        if door_label[i] > 0.5:
            color = [42,42,165]
        if window_label[i] > 0.5:
            color = [255, 255, 0]
        cv2.circle(canvas, tuple(np.round(bases[i]).astype(np.int32)), 2, tuple(color))
    return canvas


def get_score_map(particles_feat, data, model, cfg, sample_nrots=16):
    samples_feat = particles_feat

    model.eval()

    V_fov = float(data["gt_fov"][0 : 1]) / 360 * cfg["V"]
    assert V_fov % 1 == 0
    V_fov = int(V_fov)

    img_feat, _ = model(data["query_image"][0 : 1], None, V=V_fov)  # N,V,D
    img_feat = F.pad(img_feat.permute(0, 2, 1), (0, cfg["V"] - V_fov)).permute(
        0, 2, 1
    )
    score_fun = (
        lambda x, y: (F.cosine_similarity(x, y, dim=-1).sum(dim=-1) / V_fov + 1)
        * 0.5
    )

    score_list = []
    rot_samples = torch.arange(sample_nrots).float() / sample_nrots * 360

    img_feat_padded = F.pad(
        img_feat.permute(0, 2, 1), (cfg["V"], 0), mode="circular"
    )  # N,D,V
    for r in rot_samples:
        offset = r / 360 * cfg["V"]
        offset_floor, offset_ceil = int(torch.floor(offset)), int(
            torch.ceil(offset)
        )
        offset_floor_weight = offset_ceil - offset  # bilinear weight
        Vidx = torch.arange(cfg["V"])
        img_feat_roted = img_feat_padded[
            ..., cfg["V"] + Vidx - offset_floor
        ] * offset_floor_weight + img_feat_padded[
            ..., cfg["V"] + Vidx - offset_ceil
        ] * (
            1 - offset_floor_weight
        )
        img_feat_roted = img_feat_roted.permute(0, 2, 1)  # N,V,D
        score_list.append(
            score_fun(img_feat_roted.unsqueeze(1), samples_feat)
        )



    score_list = torch.stack(score_list, dim=-1)
    return score_list.squeeze(0).cpu().detach().numpy()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

MOTION_NOISE = [0.017, 0.025]

START=1
GOAL=2

SAVE_TRAJECTORY_FLAG=False
SAVE_ESTIMATED_TRAJECTORY_FLAG=False
SAVE_NOISYODOMETRY_TRAJECTORY_FLAG=False
SAVE_DATA_FORLOC=False
    
def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def create_folder(path):
    folder_name = '00'
    i = 0
    while os.path.exists(os.path.join(path, folder_name)):
        i += 1
        folder_name = f"{i:02d}"
    new_folder_path = os.path.join(path, folder_name)
    os.makedirs(new_folder_path)
    return new_folder_path

def write_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, mode='a', header=False, index=False)

if __name__ == "__main__":

    color_init=(0,0,255)
    color_init=tuple(x / 255 for x in color_init)
    color_goal=(0,255,0)
    color_goal=tuple(x / 255 for x in color_goal)
    color_trajectory=(0,0,0)
    color_trajectory=tuple(x / 255 for x in color_trajectory)
    color_ours=(255,0,0)
    color_ours=tuple(x / 255 for x in color_ours)
    color_noisy=(0,0,255)
    color_noisy=tuple(x / 255 for x in color_noisy)
    color_wall=(138/255, 138/255, 138/255)
    color_door=(189/255, 101/255, 101/255)
    color_window=(85/255, 255/255, 255/255)

    args=parse_args()

    house_id=args.House_ID #0,1,2,3,4...
    SAVE_TRAJECTORY_FLAG=args.save_trajectory
    SAVE_ESTIMATED_TRAJECTORY_FLAG=args.save_estimated_trajectory
    SAVE_NOISYODOMETRY_TRAJECTORY_FLAG=args.save_noisyodometry_trajectory

    SAVE_DATA_FORLOC=args.save_data_forloc




    House_name="House_"+str(house_id+1)
    house_color_id=house_color_id_dict[House_name]
    house_node_toplogic=house_node_toplogic_dict[House_name]
    house_node_toplogic_edge=house_node_toplogic_edge_dict[House_name]
    house_node_list=house_node_list_dict[House_name]

    House_color_map=cv2.imread("../unitydataset/House_"+str(house_id+1)+"_color.png")
    House_color_map=cv2.cvtColor(House_color_map, cv2.COLOR_BGR2RGB)

    dataforloc_dir="/home/swt/Floorplan_nav/catkin_ws/src/navigation/scripts/laser/localization_input"
    dataforloc_dir=os.path.join(dataforloc_dir,House_name)

    
    save_dir="/home/swt/Floorplan_nav/catkin_ws/src/navigation/scripts/laser/trajectory_saved"
    save_dir=os.path.join(save_dir,House_name)

    if args.wo_topo_nav:
        sub_dir="_wo_topo_nav"
    else:
        sub_dir="_w_topo_nav"
    save_dir=os.path.join(save_dir,args.mode+sub_dir)
    mkdir_if_not_exist(os.path.join(save_dir))

    csv_list=[]
    device=torch.device("cuda")


    cfg = {
        "Q": 100,
        "Q_refine": 20,
        "D": 128,
        "G": 32,
        "H": 32,
        "dist_max": 10,
        "Vr": 64,
        "V": 16,
        "disable_semantics": False,
        "disable_pointnet": False,
        "fov": 360 ,
        "view_type": "eview",
    }
    model = FpLocNet(cfg).to(device=device)

    vis_model_path="../models/try_wo_scheduler.pth"
    model.load_state_dict(torch.load(vis_model_path))

    action_model_path_1='../models/pointgoal_ddppo.onnx'
    action_model_path_2='../models/action_distribute_ddppo.onnx'

    action_model=ONNXModel(action_model_path_1,action_model_path_2)

    dataset_dir="../unitydataset"
    eval_dataset=unityDataset(dataset_dir=dataset_dir,
                              is_training=False,n_sample_points=2048,testing_set=[house_id])
    eval_dataloader=DataLoader(dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )


    rospy.init_node('listener',anonymous=True)


    for eval_data in eval_dataloader:
        for k in eval_data.keys():
            if torch.is_tensor(eval_data[k]) and not eval_data[k].is_cuda:
                eval_data[k] = eval_data[k].cuda()

        if cfg["disable_semantics"]:
            eval_data["bases_feat"][..., -2:] = 0
        sample_ret = sample_floorplan(
            eval_data,
            model,
            cfg,
            sample_grid=0.1,
            batch_size=256,
        )
        samples_loc=sample_ret["samples_loc"]
        rot_samples = torch.arange(16).float() / 16 * 360

        grid_shape  = sample_ret["samples_original_shape"]
        grid_coords = sample_ret["samples_loc"].reshape((grid_shape[0],grid_shape[1],2))
        grid_feat   = sample_ret["samples_feat"]
        break
    


    # cv2.imshow("map",floorplan_map)
    # cv2.waitKey(0)
    for ii in range(len(house_node_list)):
       for jj in range(len(house_node_list)):
            if ii==jj:
               continue
            
            if SAVE_DATA_FORLOC:
                dataforloc_path=create_folder(dataforloc_dir)

            if SAVE_TRAJECTORY_FLAG:
                floorplan_map_trajectory=render_map(eval_data["bases"][0].cpu().numpy(),eval_data["bases_feat"][0].cpu().numpy(),eval_dataset,house_id)
                floorplan_map_estimation = np.copy(floorplan_map_trajectory)

                room_lines,door_lines,window_lines=eval_dataset.get_map(house_id)

                fig, map_ax = plt.subplots()
                for iii in range(room_lines.shape[0]):
                    map_ax.plot(room_lines[iii,:,0], room_lines[iii,:,1], color=color_wall,linewidth=2.5,label='wall')
                    
                for iii in range(door_lines.shape[0]):
                    map_ax.plot(door_lines[iii,:,0], door_lines[iii,:,1], color=color_door,linewidth=2.5,label='door')
                for iii in range(window_lines.shape[0]):
                    map_ax.plot(window_lines[iii,:,0], window_lines[iii,:,1], color=color_window,linewidth=2.5,label='window')


                # wall_label = mpatches.Rectangle((0.5, 0.5), width=0.4, height=0.2, facecolor=color_wall,edgecolor='black')
                # door_label = mpatches.Rectangle((0.5, 0.5), width=0.4, height=0.2, facecolor=color_door,edgecolor='black')
                # window_label = mpatches.Rectangle((0.5, 0.5), width=0.4, height=0.2, facecolor=color_window,edgecolor='black')

                # legend_1=ax.legend(handles=[wall_label, door_label,window_label], labels=['Wall', 'Door','Window'],loc='upper right',frameon=False)

                # ax.add_artist(legend_1)
                
                # 移除轴的标题
                map_ax.set_xlabel('')
                map_ax.set_ylabel('')

                # 移除刻度标签
                map_ax.set_xticks([])
                map_ax.set_yticks([])
                plt.axis('equal')
                plt.axis('off')
                plt.gca().invert_yaxis()
                
            START=ii+1
            GOAL =jj+1

            # START=2
            # GOAL=4
            print("START:",START,"GOAL:",GOAL)

            set_robot_position(house_node_list[-1+START])

            #initialize particles
            initial_loc=house_node_list[-1+START]
            goal_loc   =house_node_list[-1+GOAL]

            if SAVE_TRAJECTORY_FLAG:
                # cv2.drawMarker(floorplan_map_trajectory, tuple(np.round(eval_dataset.meter2pixel(initial_loc,house_id)).astype(np.int)), [255,0,0],cv2.MARKER_SQUARE,6,3)
                # cv2.drawMarker(floorplan_map_trajectory, tuple(np.round(eval_dataset.meter2pixel(goal_loc,house_id)).astype(np.int)), [0,255,0],cv2.MARKER_SQUARE,6,3)

                initial_fig=map_ax.scatter(initial_loc[0],initial_loc[1],color=color_init, marker='^',s=20,label='Start')
                goal_fig=map_ax.scatter(goal_loc[0],goal_loc[1],color=color_goal, marker='o',s=20,label='Goal')
                

            ###初始化###
            ###选择初始点周围的点作为初始化备选点，此处0.1为范围，这个值很小，可以改的大一点降低初始化位置的精确度
            near_particles=[]
            init_activation_loc=[]
            num_acivate_init=10
            for i in range(grid_coords.shape[0]):
                for j in range(grid_coords.shape[1]):
                    if(abs(grid_coords[i,j,0]-initial_loc[0])<0.1 and abs(grid_coords[i,j,1]-initial_loc[1])<0.1):
                        # initial_activation.append([i,j])
                        near_particles.append([i,j])
            rand = np.random.rand
            args_coords = np.arange(len(near_particles))
            selected_args = np.random.choice(args_coords, num_acivate_init)

            for i in range(num_acivate_init):
                x = near_particles[selected_args[i]][0]
                y = near_particles[selected_args[i]][1]
                init_activation_loc.append([x, y])
            ###初始化pose cells和head cells
            pose_cell=pose_cell(GC_X_DIM=grid_shape[0],GC_Y_DIM=grid_shape[1],init_activation_loc=init_activation_loc)
            hd_cell=hd_cell(init_activation_yaw=0)
            ###
            step=0
            goal_distance=100
            image=reset()
            last_loc_gt,_=get_robot_position()
            trajectory_len=0
            motion=torch.Tensor([0,0,0])

            while(1):
                if SAVE_DATA_FORLOC:
                    image_save=image.permute(0,2,3,1).cpu().numpy()
                    
                    cv2.imwrite(os.path.join(dataforloc_path,str(step)+"_0.png"),image_save[0,:,:,:])
                    cv2.imwrite(os.path.join(dataforloc_path,str(step)+"_1.png"),image_save[1,:,:,:])
                    cv2.imwrite(os.path.join(dataforloc_path,str(step)+"_2.png"),image_save[2,:,:,:])
                    cv2.imwrite(os.path.join(dataforloc_path,str(step)+"_3.png"),image_save[3,:,:,:])
                    loc_gt_save,rot_gt_save=get_robot_position()

                    write_to_csv([[loc_gt_save[0],loc_gt_save[1],rot_gt_save]],os.path.join(dataforloc_path,"gt_pos.csv"))

            ################# localization ####################
                if SAVE_NOISYODOMETRY_TRAJECTORY_FLAG:
                    if step==0:
                        now_loc_est_noisy,rot_est_noisy=get_robot_position()
                    else:
                        r1Noise = MOTION_NOISE[0]
                        transNoise = MOTION_NOISE[1]

                        rot1 = motion[0] + r1Noise * np.random.randn(1)#顺时针
                        rot2 = 0#motion[2] + r1Noise * np.random.randn(num_particles)#顺时针
                        tras1 = motion[1] + transNoise * np.random.randn(1)

                        # update pose using motion model角度顺时针为正方向，以y轴负半轴为0°
                        now_loc_est_noisy[0] += tras1 * np.cos(rot_est_noisy + rot1 - np.pi/2) #x
                        now_loc_est_noisy[1] += tras1 * np.sin(rot_est_noisy + rot1 - np.pi/2) #y
                        rot_est_noisy += rot1+rot2 #theta

                        # last_loc_est_pixel_noisy=eval_dataset.meter2pixel(last_loc_est_noisy,house_id)
                        # now_loc_est_pixel_noisy=eval_dataset.meter2pixel(now_loc_est_noisy,house_id)    
                        # cv2.line(floorplan_map_trajectory,tuple(np.round(last_loc_est_pixel_noisy).astype(np.int)),tuple(np.round(now_loc_est_pixel_noisy).astype(np.int)),  [255, 0, 0],2)

                        noisy_fig,=map_ax.plot([last_loc_est_noisy[0], now_loc_est_noisy[0]],[last_loc_est_noisy[1], now_loc_est_noisy[1]], color=color_noisy,linewidth=1,label='Noisy Odometry')
                        

                    last_loc_est_noisy=now_loc_est_noisy.copy()    

                if args.mode=="FP_LOC":


                    pano_image=image_2_pano(image)
                    data={"gt_fov":     torch.Tensor([[360]]).float(),
                        "query_image":pano_image.to(device).float()
                        }  
                    ###根据floorplan和当前观测获得score map
                    score=get_score_map(grid_feat,data,model,cfg,sample_nrots=16)

                    ###直接全局最大值搜索的位置估计结果
                    scores, matched_rot_idxs = torch.from_numpy(score).max(dim=-1)#对环形特征子进行16个角度的旋转后及逆行比对，找到各个位置相似度最高的旋转角度
                    loc_est_search = samples_loc[scores.argmax()].reshape(2).cpu().numpy()
                    rot_est_search = matched_rot_idxs.reshape(-1)[scores.argmax()].reshape(1, 1, 1)
                    rot_est_search = (rot_samples[rot_est_search] / 180 * np.pi).cpu().numpy()


                    ###需要给pose cells输入的激活map
                    score_pose=np.max(score,axis=1)
                    score_pose=score_pose.reshape((grid_shape[0],grid_shape[1]))

                    ###获取posecell中激活的最大的区域，在该区域中获得给head cells输入的激活map
                    x_packet,y_packet=pose_cell.get_packet()
                    score=score.reshape((grid_shape[0],grid_shape[1],-1))
                    X_wrap=np.repeat(np.expand_dims(x_packet,axis=1),y_packet.shape[0],axis=1)
                    Y_wrap=np.repeat(np.expand_dims(y_packet,axis=0),x_packet.shape[0],axis=0)

                    score_packet=score[X_wrap,Y_wrap,:].reshape(-1,16)
                    score_yaw=np.max(score_packet,axis=0)

                    ###对head cells进行迭代（move+activation）
                    hd_cell.iteration(motion[0].cpu().numpy(),score_yaw)
                    curYaw=hd_cell.get_yaw()
                    # print("current_yaw:",curYaw/np.pi*180)
                    curYaw-=np.pi/2

                    ###对pose cells进行迭代（move+activation）
                    pose_cell.iteration(curYaw,motion[1].cpu().numpy()/0.1,score_pose)

                    x_est_idx , y_est_idx = pose_cell.get_pose()
                    x_est_idx , y_est_idx = int(x_est_idx) , int(y_est_idx)
                    x_est = grid_coords[x_est_idx,y_est_idx,0].cpu().numpy()
                    y_est = grid_coords[x_est_idx,y_est_idx,1].cpu().numpy()

                    now_loc_est=np.array([x_est,y_est])
                    rot_est=curYaw+np.pi/2


                elif args.mode=="GT":
                    now_loc_est,rot_est=get_robot_position()
                elif args.mode=="NOISY":
                    if step==0:
                        now_loc_est,rot_est=get_robot_position()
                    else:
                        r1Noise = MOTION_NOISE[0]
                        transNoise = MOTION_NOISE[1]

                        rot1 = motion[0] + r1Noise * np.random.randn(1)#顺时针
                        rot2 = 0#motion[2] + r1Noise * np.random.randn(num_particles)#顺时针
                        tras1 = motion[1] + transNoise * np.random.randn(1)

                        # update pose using motion model角度顺时针为正方向，以y轴负半轴为0°
                        now_loc_est[0] += tras1 * np.cos(rot_est + rot1 - np.pi/2) #x
                        now_loc_est[1] += tras1 * np.sin(rot_est + rot1 - np.pi/2) #y
                        rot_est += rot1+rot2 #theta
                        
                ##########################

                ############planner########################
                if args.wo_topo_nav:
                    local_goal=goal_loc
                    if step==0:
                        prev_actions=torch.zeros((1,1))
                        masks=torch.zeros((1,1))
                        masks[0,0]=False
                        rnn_hidden_states_1=torch.zeros((1,4,512))
                else:
                    if step%100==0 or goal_distance<0.2:#到达subpoint
                        if torch.is_tensor(now_loc_est):
                            now_loc_est=now_loc_est.cpu().numpy()
                        if torch.is_tensor(goal_loc):
                            goal_loc=goal_loc.cpu().numpy()

                        loc_est_pixel=eval_dataset.meter2pixel(now_loc_est,house_id)
                        goal_pixel=eval_dataset.meter2pixel(goal_loc,house_id)

                        now_color =House_color_map[loc_est_pixel[1],loc_est_pixel[0]]
                        goal_color=House_color_map[goal_pixel[1],goal_pixel[0]]

                        for i in range(len(house_color_id)):
                            if now_color[0]==house_color_id[i][0] and now_color[1]==house_color_id[i][1] and now_color[2]==house_color_id[i][2]:
                                now_node=i+1
                            if goal_color[0]==house_color_id[i][0] and goal_color[1]==house_color_id[i][1] and goal_color[2]==house_color_id[i][2]:
                                goal_node=i+1                
                        # print("now_node:",now_node,"goal_node:",goal_node)

                        if now_node==goal_node:#in the same room
                            local_goal=goal_loc
                        else:
                            path=Dijkstra(house_node_toplogic, now_node, goal_node)
                            # print(path)
                            cross_door=house_node_toplogic_edge[path[0]-1][path[1]-1]
                            # print(cross_door)
                            first_node=eval_dataset.pixel2meter(cross_door[0],house_id)
                            second_node=eval_dataset.pixel2meter(cross_door[1],house_id)

                            if np.sqrt((first_node[0]-now_loc_est[0])**2+(first_node[1]-now_loc_est[1])**2)<0.2:
                                local_goal=second_node
                            else:
                                local_goal=first_node

                        # print("loc_goal:",local_goal)
                        prev_actions=torch.zeros((1,1))
                        masks=torch.zeros((1,1))
                        masks[0,0]=False
                        rnn_hidden_states_1=torch.zeros((1,4,512))

                ####################################

                ############locomotion##############

                goal_distance=np.sqrt((local_goal[0]-now_loc_est[0])**2+(local_goal[1]-now_loc_est[1])**2)
                goal_angle=np.arctan2(local_goal[1]-now_loc_est[1],local_goal[0]-now_loc_est[0])+np.pi/2-rot_est

                rgb=image[0:1,:,:,:].permute(0,2,3,1)
                pointgoal=torch.zeros((1,2))
                pointgoal[0,0]=goal_distance
                pointgoal[0,1]=-goal_angle
                action, rnn_hidden_states=action_model.compute(rgb, pointgoal, rnn_hidden_states_1, prev_actions, masks)
                prev_actions=action
                rnn_hidden_states_1=rnn_hidden_states
                masks[0,0]=True

                if action[0]==0:
                    action_=3
                elif action[0]==1:
                    action_=0
                elif action[0]==2:
                    action_=1
                elif action[0]==3:
                    action_=2
                # print(action)

                # ##########################

                # action_=0

                image=step_(action=action_)

                if action_==0:
                    motion=torch.Tensor([0,0.25,0])
                elif action_==1:
                    motion=torch.Tensor([-10*np.pi/180,0,0])
                elif action_==2:
                    motion=torch.Tensor([ 10*np.pi/180,0,0])
                elif action_==3:
                    motion=torch.Tensor([0,0,0])

                if SAVE_DATA_FORLOC:
                    write_to_csv([[action]],os.path.join(dataforloc_path,"action.csv"))


                now_loc_gt,_=get_robot_position()

                if SAVE_TRAJECTORY_FLAG:

                    # last_loc_gt_pixel=eval_dataset.meter2pixel(last_loc_gt,house_id)
                    # now_loc_gt_pixel=eval_dataset.meter2pixel(now_loc_gt,house_id)
                    # cv2.line(floorplan_map_trajectory,tuple(np.round(last_loc_gt_pixel).astype(np.int)),tuple(np.round(now_loc_gt_pixel).astype(np.int)),  [0, 0, 0],2)
                    
                    trajectory_fig,=map_ax.plot([last_loc_gt[0], now_loc_gt[0]],[last_loc_gt[1], now_loc_gt[1]], color=color_trajectory,linewidth=1,label='GT Trajectory')
                    

               
               
                if SAVE_ESTIMATED_TRAJECTORY_FLAG:
                    if step>0:
                        # last_loc_est_pixel=eval_dataset.meter2pixel(last_loc_est,house_id)
                        # now_loc_est_pixel=eval_dataset.meter2pixel(now_loc_est,house_id)    
                        # cv2.line(floorplan_map_trajectory,tuple(np.round(last_loc_est_pixel).astype(np.int)),tuple(np.round(now_loc_est_pixel).astype(np.int)),  [0, 0, 255],2)

                        ours_fig,=map_ax.plot([last_loc_est[0], now_loc_est[0]],[last_loc_est[1], now_loc_est[1]], color=color_ours,linewidth=1,label='Ours')
                    




                trajectory_len+=np.sqrt((now_loc_gt[0]-last_loc_gt[0])**2+(now_loc_gt[1]-last_loc_gt[1])**2)
                last_loc_gt=now_loc_gt.copy()
                last_loc_est=now_loc_est.copy()    


                step+=1
                
                trajectory_save_path=os.path.join(save_dir,str(START)+"_"+str(GOAL)+".svg")
                legend_ = map_ax.legend(handles=[initial_fig,goal_fig,trajectory_fig], loc=(0.8,-0.1),frameon=False)
                map_ax.add_artist(legend_)

                plt.savefig(trajectory_save_path, format='svg')
                if np.sqrt((goal_loc[0]-now_loc_gt[0])**2+(goal_loc[1]-now_loc_gt[1])**2)<0.2:
                    if SAVE_TRAJECTORY_FLAG:
                        # trajectory_save_path=os.path.join(save_dir,str(START)+"_"+str(GOAL)+".png")
                        # cv2.imwrite(trajectory_save_path,floorplan_map_trajectory)
                        
                        trajectory_save_path=os.path.join(save_dir,str(START)+"_"+str(GOAL)+".svg")
                        legend_ = map_ax.legend(handles=[initial_fig,goal_fig,trajectory_fig], loc=(0.8,-0.1),frameon=False)
                        map_ax.add_artist(legend_)

                        plt.savefig(trajectory_save_path, format='svg')
                        plt.clf()

                    print("finally move ",trajectory_len,"m !!!!")
                    csv_list.append([START,GOAL,trajectory_len])
                    df1 = pd.DataFrame(data=csv_list)#,columns=['filepath', 'label'])
                    df1.to_csv(os.path.join(save_dir,"trajectory_len.csv"))
                    break
                if step>500:
                    if SAVE_TRAJECTORY_FLAG:
                        # trajectory_save_path=os.path.join(save_dir,str(START)+"_"+str(GOAL)+".png")
                        # cv2.imwrite(trajectory_save_path,floorplan_map_trajectory)


                        trajectory_save_path=os.path.join(save_dir,str(START)+"_"+str(GOAL)+".svg")
                        legend_ = map_ax.legend(handles=[initial_fig,goal_fig,trajectory_fig], loc=(0.8,-0.1),frameon=False)
                        map_ax.add_artist(legend_)

                        plt.savefig(trajectory_save_path, format='svg')
                        plt.clf()
                    print("nav failed!!!")
                    csv_list.append([START,GOAL,-1])
                    df1 = pd.DataFrame(data=csv_list)#,columns=['filepath', 'label'])
                    df1.to_csv(os.path.join(save_dir,"trajectory_len.csv"))

                    break


