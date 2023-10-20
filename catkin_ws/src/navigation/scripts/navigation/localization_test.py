#!/usr/bin/env /home/swt/anaconda3/envs/LLL/bin/python

from dataset.unity_dataset import unityDataset
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from model.fplocnet import FpLocNet, quick_fplocnet_call
from eval_utils import *
import os
import pickle
import imageio
from utils.particle_filter import init_particles_given_coords,align_particles,update_particles,resample,move_particles
import argparse
import pandas as pd
from dataset.unity_dataset import persp2pano
import pandas as pd
import matplotlib.pyplot as plt
gif_images = []
def render_map(bases, bases_feat,eval_dataset,house_id,border=100):#scale：1m对应多少个像素点
    n_bases = bases.shape[0]
    bases = np.copy(bases)
    # affine = (border - bases.min(axis=0), scale)
    # bases = (bases + affine[0]) * affine[1]
    # loc_gt = (loc_gt + affine[0]) * affine[1]#单位变成了像素
    # loc_est = (loc_est + affine[0]) * affine[1]
    for i in range(n_bases):
        bases[i]=eval_dataset.meter2pixel(bases[i],house_id)

    W, H = np.ptp(bases, axis=0).astype(np.int) + int(2 * border )
    canvas = 255*np.ones((H, W, 3), np.uint8)

    door_label = bases_feat[:, -2]
    window_label = bases_feat[:, -1]
    for i in range(n_bases):
        color = [60, 60, 60]
        if door_label[i] > 0.5:
            color = [42,42,165]
        if window_label[i] > 0.5:
            color = [255, 255, 0]
        cv2.circle(canvas, tuple(np.round(bases[i]).astype(np.int)), 2, tuple(color))
    return canvas

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
    
    image_for_save=image
    # cv2.imwrite("pano.jpg",image)

    image=image/255.
    image -= (0.485, 0.456, 0.406)
    image /= (0.229, 0.224, 0.225)

    image=np.transpose(image, (2, 0, 1))
    image=torch.from_numpy(image).unsqueeze(0)
    return image,image_for_save

def parse_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"                   
    parser = argparse.ArgumentParser(description=description)        
                                                                 
    help = "The path of address"
    parser.add_argument('--House_ID',type=int,help="0 as Start")         
    parser.add_argument('--save_pics',action='store_true')         

    parser.add_argument('--mode',choices=["FP_LOC","FP_LOC_NOPF"])         

    args = parser.parse_args() 
    return args

def get_csv_row(filename, row_number):
    df = pd.read_csv(filename)
    row = df.iloc[row_number].values.tolist()
    return row

def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

@torch.no_grad()
def match_wo_pf(
    sample_ret, data, model, cfg, mode="match", sample_nrots=16, max_refine_its=3
):
    assert mode in ["match", "refine"]

    samples_loc, samples_feat = sample_ret["samples_loc"], sample_ret["samples_feat"]

    model.eval()
    n_images = data["query_image"].shape[0]

    loc_ests = []
    rot_ests = []

    img_feats = []
    for i in range(n_images):

        if cfg["view_type"] == "pview":
            img_feat, _ = model(data["query_image"][i : i + 1], None)
            img_feat, img_feat_mask = persp2equir(
                img_feat, data["gt_fov"][i : i + 1], cfg["V"]#将平面上的特征向量转为环形
            )
            score_fun = (
                lambda x, y: (
                    F.cosine_similarity(x, y, dim=-1).sum(dim=-1)
                    / img_feat_mask.sum(dim=-1)
                    + 1
                )
                * 0.5
            )
        elif cfg["view_type"] == "eview":
            if cfg["V"] != 1:  # V=1 is disable circular feat
                V_fov = float(data["gt_fov"][i : i + 1]) / 360 * cfg["V"]
                assert V_fov % 1 == 0
                V_fov = int(V_fov)
            else:
                V_fov = 1
            img_feat, _ = model(data["query_image"][i : i + 1], None, V=V_fov)  # N,V,D
            img_feat = F.pad(img_feat.permute(0, 2, 1), (0, cfg["V"] - V_fov)).permute(
                0, 2, 1
            )
            score_fun = (
                lambda x, y: (F.cosine_similarity(x, y, dim=-1).sum(dim=-1) / V_fov + 1)
                * 0.5
            )
        else:
            raise "Unknown view_type"

        img_feats.append(img_feat.cpu().numpy())

        if mode in ["refine", "match"]:
            score_list = []
            rot_samples = torch.arange(sample_nrots).float() / sample_nrots * 360
            # Note bilinear interpolation cannot be applied to partially masked image-feat except V=sample_nrots
            # Decide to rotate image-feat(fast) or map-feat(a bit slower tiny bit slower)
            if data["gt_fov"][i] < 360 and sample_nrots != cfg["V"]:
                samples_feat_padded = F.pad(
                    samples_feat.squeeze(0).permute(0, 2, 1),
                    (cfg["V"], 0),
                    mode="circular",
                )  # N,D,V
                for r in rot_samples:
                    offset = r / 360 * cfg["V"]
                    offset_floor, offset_ceil = int(torch.floor(offset)), int(
                        torch.ceil(offset)
                    )
                    offset_floor_weight = offset_ceil - offset  # bilinear weight
                    Vidx = torch.arange(cfg["V"])
                    samples_feat_roted = samples_feat_padded[
                        ..., Vidx + offset_floor
                    ] * offset_floor_weight + samples_feat_padded[
                        ..., Vidx + offset_ceil
                    ] * (
                        1 - offset_floor_weight
                    )
                    samples_feat_roted = samples_feat_roted.permute(0, 2, 1).unsqueeze(
                        0
                    )  # N,Q,V,D
                    score_list.append(score_fun(img_feat, samples_feat_roted))
            else:
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
            scores, matched_rot_idxs = score_list.max(dim=-1)#对环形特征子进行16个角度的旋转后及逆行比对，找到各个位置相似度最高的旋转角度
            loc_est = samples_loc[scores.argmax()].reshape(2)
            rot_est = matched_rot_idxs.reshape(-1)[scores.argmax()].reshape(1, 1, 1)
            rot_est = (rot_samples[rot_est] / 180 * np.pi).to(scores.device)


        loc_est = loc_est.reshape(2).cpu().numpy()
        rot_est = rot_est.reshape(1).cpu().numpy()


        loc_ests.append(loc_est)
        rot_ests.append(rot_est)



    return {
        "loc_ests": np.stack(loc_ests, axis=0),
        "rot_ests": np.stack(rot_ests, axis=0),
    }


if __name__ == "__main__":
    color_gt=(0,0,0)
    color_gt=tuple(x / 255 for x in color_gt)
    color_est=(255,0,0)
    color_est=tuple(x / 255 for x in color_est)

    color_wall=(138/255, 138/255, 138/255)
    color_door=(189/255, 101/255, 101/255)
    color_window=(85/255, 255/255, 255/255)

    args=parse_args()
    house_id=args.House_ID
    device=torch.device("cuda")

    out_path="/home/swt/Floorplan_nav/catkin_ws/src/navigation/scripts/laser/localization_output"
    out_path=os.path.join(out_path,args.mode)
    mkdir_if_not_exist(out_path)

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
    model = FpLocNet(cfg).cuda()
    model.load_state_dict(torch.load("/home/swt/Floorplan_nav/catkin_ws/src/navigation/scripts/laser/try_wo_scheduler.pth"))

    eval_dataset=unityDataset(dataset_dir="/home/swt/catkin_ws/src/communication/scripts/laser/unitydataset",is_training=False,n_sample_points=2048,testing_set=[house_id])
    eval_dataloader=DataLoader(dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )

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
        grid_shape=sample_ret["samples_original_shape"]
        grid_coords=sample_ret["samples_loc"].reshape((grid_shape[0],grid_shape[1],2))
        break

    floorplan_map_trajectory=render_map(eval_data["bases"][0].cpu().numpy(),eval_data["bases_feat"][0].cpu().numpy(),eval_dataset,house_id)

    t_err_sum=0
    r_err_sum=0
    one_meter_num=0
    half_meter_num=0
    pointone_meter_num=0

    example_num=0



    data_dir_="/home/swt/Floorplan_nav/catkin_ws/src/navigation/scripts/laser/localization_input/House_8/"

    for i in range(6):
        data_dir=os.path.join(data_dir_,f"{i:02d}")
        step_num = (len(os.listdir(data_dir))-2)//4

        initial_pose=get_csv_row(os.path.join(data_dir,"gt_pos.csv"),0)
        initial_loc=np.array([initial_pose[0],initial_pose[1]])

        particles =init_particles_given_coords(500,grid_coords,initial_loc)

        for step in range(step_num-1):

            if args.mode=="FP_LOC":
                if step>0:
                    action=get_csv_row(os.path.join(data_dir,"action.csv"),step-1)[0]

                    if action=="tensor([[0]])":
                        action=3
                    elif action=="tensor([[1]])":
                        action=0
                    elif action=="tensor([[2]])":
                        action=1
                    elif action=="tensor([[3]])":
                        action=2

                    # print(action)
                    if action==0:
                        motion=torch.Tensor([0,0.25,0])
                    elif action==1:
                        motion=torch.Tensor([-10*np.pi/180,0,0])
                    elif action==2:
                        motion=torch.Tensor([ 10*np.pi/180,0,0])
                    elif action==3:
                        motion=torch.Tensor([0,0,0])
                    particles=move_particles(particles,motion)

                particles,nearest_idx=align_particles(particles,grid_coords)#与采样网格对齐

                particles_feat=sample_ret["samples_feat"][0:1,nearest_idx,:,:]  

                image_0=torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir,str(step)+"_0.png")),cv2.COLOR_BGR2RGB)).permute(2,0,1)
                image_1=torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir,str(step)+"_1.png")),cv2.COLOR_BGR2RGB)).permute(2,0,1)
                image_2=torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir,str(step)+"_2.png")),cv2.COLOR_BGR2RGB)).permute(2,0,1)
                image_3=torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir,str(step)+"_3.png")),cv2.COLOR_BGR2RGB)).permute(2,0,1)

                image=torch.stack([image_0,image_1,image_2,image_3],dim=0)

                pano_image,pano_image_for_save=image_2_pano(image)
                data={"gt_fov":     torch.Tensor([[360]]).float(),
                    "query_image":pano_image.to(device).float()
                    }      
                
                if step==0:
                    particles=update_particles(particles,particles_feat,data,model,cfg,sample_nrots=16,heading_mask=False)
                else:
                    particles=update_particles(particles,particles_feat,data,model,cfg,sample_nrots=16,heading_mask=True)
                
                particles = resample(particles)


                best_match_idx=np.argmax(particles[:,3])

                now_loc_est=particles[best_match_idx,0:2]
                rot_est=particles[best_match_idx,2]

                gt_pose=get_csv_row(os.path.join(data_dir,"gt_pos.csv"),step)


                # print("est:",now_loc_est,"gt:",gt_pose)
                # if step>0:
                #     last_loc_est_pixel=eval_dataset.meter2pixel(last_loc_est,house_id)
                #     now_loc_est_pixel=eval_dataset.meter2pixel(now_loc_est,house_id)    
                #     cv2.line(floorplan_map_trajectory,tuple(np.round(last_loc_est_pixel).astype(np.int)),tuple(np.round(now_loc_est_pixel).astype(np.int)),  [0, 0, 255],2)
        
            elif args.mode=="FP_LOC_NOPF":
                    
                image_0=torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir,str(step)+"_0.png")),cv2.COLOR_BGR2RGB)).permute(2,0,1)
                image_1=torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir,str(step)+"_1.png")),cv2.COLOR_BGR2RGB)).permute(2,0,1)
                image_2=torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir,str(step)+"_2.png")),cv2.COLOR_BGR2RGB)).permute(2,0,1)
                image_3=torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir,str(step)+"_3.png")),cv2.COLOR_BGR2RGB)).permute(2,0,1)

                image=torch.stack([image_0,image_1,image_2,image_3],dim=0)

                pano_image,pano_image_for_save=image_2_pano(image)
                data={"gt_fov":     torch.Tensor([[360]]).float(),
                    "query_image":pano_image.to(device).float()
                    }
                
                match_ret = match_wo_pf(
                                sample_ret,
                                data,
                                model,
                                cfg,
                                mode="match",
                                sample_nrots=16,
                                max_refine_its=3,
                            )
                now_loc_est=match_ret["loc_ests"][0]
                rot_est=match_ret["rot_ests"][0][0]
                gt_pose=get_csv_row(os.path.join(data_dir,"gt_pos.csv"),step)

            #     now_loc_est_pixel=eval_dataset.meter2pixel(now_loc_est,house_id)    
            #     cv2.circle(floorplan_map_trajectory, (now_loc_est_pixel[0], now_loc_est_pixel[1]), 2, [0, 0, 128], -1)   
                                
            # last_loc_est=now_loc_est.copy()    
            # cv2.imwrite("est_traj.png",floorplan_map_trajectory)

            if args.save_pics:


                room_lines,door_lines,window_lines=eval_dataset.get_map(house_id)

                fig, map_ax = plt.subplots()
                for iii in range(room_lines.shape[0]):
                    map_ax.plot(room_lines[iii,:,0], room_lines[iii,:,1], color=color_wall,linewidth=2.5,label='wall')
                    
                for iii in range(door_lines.shape[0]):
                    map_ax.plot(door_lines[iii,:,0], door_lines[iii,:,1], color=color_door,linewidth=2.5,label='door')
                for iii in range(window_lines.shape[0]):
                    map_ax.plot(window_lines[iii,:,0], window_lines[iii,:,1], color=color_window,linewidth=2.5,label='window')
                
                # 移除轴的标题
                map_ax.set_xlabel('')
                map_ax.set_ylabel('')

                # 移除刻度标签
                map_ax.set_xticks([])
                map_ax.set_yticks([])
                plt.axis('equal')
                plt.axis('off')
                plt.gca().invert_yaxis()

                gt_fig=map_ax.scatter(gt_pose[0],gt_pose[1],color=color_gt, marker='+',s=20,label='GT')
                est_fig=map_ax.scatter(now_loc_est[0],now_loc_est[1],color=color_est, marker='o',s=20,label='EST')
                plt.savefig(os.path.join(out_path,"map"+str(example_num)+".svg"), format='svg')
                plt.clf()

                map=floorplan_map_trajectory.copy()
                now_loc_est_pixel=eval_dataset.meter2pixel(now_loc_est,house_id) 
                now_loc_gt_pixel=eval_dataset.meter2pixel(gt_pose[:2],house_id)    

                cv2.circle(map, (now_loc_est_pixel[0], now_loc_est_pixel[1]), 7, [0, 0, 255], -1)  
                cv2.drawMarker(map, (now_loc_gt_pixel[0], now_loc_gt_pixel[1]), [0, 0, 0],cv2.MARKER_CROSS,14,2)  
            
                cv2.imwrite(os.path.join(out_path,"map"+str(example_num)+".png"),map)
                cv2.imwrite(os.path.join(out_path,"pano"+str(example_num)+".png"),pano_image_for_save)
                
            example_num+=1

            t_err=np.linalg.norm(now_loc_est-gt_pose[:2])
            r_err=((rot_est-gt_pose[2])/np.pi*180)%360
            if r_err<180:
                r_err=r_err
            else:
                r_err=360-r_err
            t_err_sum+=t_err
            r_err_sum+=r_err

            if t_err<1:
                one_meter_num+=1
            if t_err<0.5:
                half_meter_num+=1
            if t_err<0.1:
                pointone_meter_num+=1

    print(args.mode)
    print("t_error",t_err_sum/example_num)
    print("r_error",r_err_sum/example_num)
    print("1m recall:",one_meter_num/example_num)
    print("0.5m recall:",half_meter_num/example_num)
    print("0.1m recall:",pointone_meter_num/example_num)

    print(example_num)



