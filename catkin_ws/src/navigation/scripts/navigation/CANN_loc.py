from dataset.unity_dataset import unityDataset
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from model.fplocnet import FpLocNet, quick_fplocnet_call
from eval_utils import *
import os
import pickle
from CANN.pose_cell import pose_cell
from CANN.hd_cell import hd_cell
import matplotlib.pyplot as plt

def get_motion(last_gt_loc,last_gt_rot,gt_loc,gt_rot):
    #动作模式：先旋转，再直行一段距离
  if torch.is_tensor(last_gt_loc):
    last_gt_loc=last_gt_loc.cpu().numpy()

  if torch.is_tensor(last_gt_rot):
    last_gt_rot=last_gt_rot.cpu().numpy()

  if torch.is_tensor(gt_loc):
    gt_loc=gt_loc.cpu().numpy()

  if torch.is_tensor(gt_rot):
    gt_rot=gt_rot.cpu().numpy()
  
  motion=np.zeros(3)
  motion[1]=np.linalg.norm(gt_loc-last_gt_loc)#距离
  motion[0]=np.arctan2(gt_loc[1]-last_gt_loc[1],gt_loc[0]-last_gt_loc[0])+np.pi/2-last_gt_rot#旋转
  motion[2]=gt_rot-last_gt_rot#旋转

  return motion

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
    # print(score_list.shape)
    # scores, matched_rot_idxs = score_list.max(dim=-1)#对环形特征子进行16个角度的旋转后及逆行比对，找到各个位置相似度最高的旋转角度
    # particles[:,3]*=scores.squeeze(0).detach().cpu().numpy()
    # particles[:, 3] = particles[:, 3] / np.max(particles[:, 3])

    # particles[:,2] = rot_samples[matched_rot_idxs.squeeze(0)].cpu().numpy() / 180 * np.pi
    # return particles

def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)


folder_name="fov90"

if __name__ == "__main__":

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
    model.load_state_dict(torch.load("try_wo_scheduler.pth"))

    eval_dataset=unityDataset(dataset_dir="unitydataset_new",is_training=False,n_sample_points=2048,testing_set=[1])
    eval_dataloader=DataLoader(dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )



    save_dir = os.path.join("eval_unitydataset", folder_name)
    mkdir_if_not_exist(os.path.join(save_dir, "score_maps"))
    mkdir_if_not_exist(os.path.join(save_dir, "rot_maps"))
    mkdir_if_not_exist(os.path.join(save_dir, "results"))
    mkdir_if_not_exist(os.path.join(save_dir, "query_images"))
    mkdir_if_not_exist(os.path.join(save_dir, "terrs"))
    mkdir_if_not_exist(os.path.join(save_dir, "rerrs"))
    mkdir_if_not_exist(os.path.join(save_dir, "raws"))

    idx=0
    num_1meter_ours=0
    num_halfmeter_ours=0
    num_01meter_ours=0
    total_loc_err_ours=0

    num_1meter_noisy=0
    num_halfmeter_noisy=0
    num_01meter_noisy=0
    total_loc_err_noisy=0

    num_1meter_search=0
    num_halfmeter_search=0
    num_01meter_search=0
    total_loc_err_search=0

    gt_traj_x=[]
    gt_traj_y=[]

    est_traj_x=[]
    est_traj_y=[]

    noisy_traj_x=[]
    noisy_traj_y=[]


    for eval_data in eval_dataloader:
        for k in eval_data.keys():
            if torch.is_tensor(eval_data[k]) and not eval_data[k].is_cuda:
                eval_data[k] = eval_data[k].cuda()

        if cfg["disable_semantics"]:
            eval_data["bases_feat"][..., -2:] = 0
        # print("==========",idx,"=========")

        if idx==0:#初始位置初始化
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


            initial_loc=eval_data["gt_loc"][0]
            near_particles=[]
            init_activation_loc=[]
            num_acivate_init=10
            ###选择初始点周围的点作为初始化备选点，此处0.1为范围，这个值很小，可以改的大一点降低初始化位置的精确度
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
            # print(init_activation_loc)


            last_gt_loc=eval_data["gt_loc"][0].cpu().numpy()
            last_gt_rot=eval_data["gt_rot"][0]

            noisy_x=last_gt_loc[0]
            noisy_y=last_gt_loc[1]
            noisy_theta=last_gt_rot.cpu().numpy()

            noisy_traj_x.append(noisy_x)
            noisy_traj_y.append(noisy_y)

            ###初始化pose cells和head cells
            pose_cell=pose_cell(GC_X_DIM=grid_shape[0],GC_Y_DIM=grid_shape[1],init_activation_loc=init_activation_loc)
            hd_cell=hd_cell(init_activation_yaw=last_gt_rot)

            ###根据floorplan和当前观测获得score map
            score=get_score_map(grid_feat,eval_data,model,cfg,sample_nrots=16)

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
            hd_cell.iteration(0,score_yaw)
            curYaw=hd_cell.get_yaw()
            # print("current_yaw:",curYaw/np.pi*180)
            curYaw-=np.pi/2

            ###对pose cells进行迭代（move+activation）
            pose_cell.iteration(curYaw,0,score_pose)

        else:


            gt_loc=eval_data["gt_loc"][0].cpu().numpy()
            gt_traj_x.append(gt_loc[0])
            gt_traj_y.append(gt_loc[1])
            
            ###获取motion的信息
            motion=get_motion(last_gt_loc,last_gt_rot,eval_data["gt_loc"][0],eval_data["gt_rot"][0])

            ###对motion信息加上噪声
            motion[0] += 5*np.pi/180 * np.random.randn(1)
            motion[1] += 0.05 * np.random.randn(1)
            motion[2] += 2*np.pi/180 * np.random.randn(1)
            

            ###噪声里程计累加的估计结果
            noisy_x += motion[1] * np.cos(noisy_theta + motion[2] - np.pi/2) #x
            noisy_y += motion[1] * np.sin(noisy_theta + motion[2] - np.pi/2) #y
            noisy_theta += motion[2] #theta
            noisy_x=noisy_x[0]
            noisy_y=noisy_y[0]

            noisy_traj_x.append(noisy_x)
            noisy_traj_y.append(noisy_y)

            ###根据floorplan和当前观测获得score map
            score=get_score_map(grid_feat,eval_data,model,cfg,sample_nrots=16)
            
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

            hd_cell.iteration(motion[2],score_yaw)
            curYaw=hd_cell.get_yaw()
            # print("current_yaw:",curYaw/np.pi*180)
            # print("gt_yaw:",eval_data["gt_rot"][0]/np.pi*180)
            # print("move_dir:",(np.arctan2(gt_loc[1]-last_gt_loc[1],gt_loc[0]-last_gt_loc[0])+np.pi/2)/np.pi*180)
            curYaw-=np.pi/2

            pose_cell.iteration(curYaw,motion[1]/0.1,score_pose)

            # plt.imshow(score_pose, cmap='viridis', interpolation='nearest')
            # plt.savefig('CANN/score/'+str(idx)+'.png')
            # plt.clf()

            last_gt_loc=eval_data["gt_loc"][0].cpu().numpy()
            last_gt_rot=eval_data["gt_rot"][0]
        # print(last_gt_rot.cpu().numpy())

        # print(score_pose.shape,score_yaw.shape)
        x_est_idx , y_est_idx = pose_cell.get_pose()
        x_est_idx , y_est_idx = int(x_est_idx) , int(y_est_idx)
        x_est = grid_coords[x_est_idx,y_est_idx,0].cpu().numpy()
        y_est = grid_coords[x_est_idx,y_est_idx,1].cpu().numpy()

        loc_est=np.array([x_est,y_est])
        noisy_loc=np.array([noisy_x,noisy_y])

        loc_err_ours=np.linalg.norm(loc_est-last_gt_loc)
        loc_err_noisy=np.linalg.norm(noisy_loc-last_gt_loc)
        loc_err_search=np.linalg.norm(loc_est_search-last_gt_loc)

        total_loc_err_noisy+=loc_err_noisy
        total_loc_err_ours +=loc_err_ours
        total_loc_err_search +=loc_err_search



        est_traj_x.append(x_est)
        est_traj_y.append(y_est)


        fig, ax = plt.subplots(1, 1)

        eval_dataset.get_floorplan(1,fig, ax)

        ax.plot(gt_traj_x,gt_traj_y,color='black',linewidth=1)
        ax.plot(est_traj_x,est_traj_y,color='r',linewidth=1)
        ax.plot(noisy_traj_x,noisy_traj_y,color='b',linewidth=1)
        ax.set_xlim(-10, 2)
        ax.set_ylim(-12, 2)
        ax.axis('auto')
        ax.axis('equal')

        # plt.imshow(pose_cell.GRIDCELLS, cmap='viridis', interpolation='nearest')
        # plt.savefig('CANN/temp/'+str(idx)+'.png')
        # plt.clf()

        # img1=ax[1,0].imshow(pose_cell.GRIDCELLS, cmap='viridis', interpolation='nearest')
        # plt.colorbar(img1, ax=ax[1,0])
        # img2=ax[1,1].imshow(score_pose, cmap='viridis', interpolation='nearest')
        # plt.colorbar(img2, ax=ax[1,1])

        plt.savefig('traj.png')
        plt.clf()

        # plt.plot(loc_est[0],loc_est[1],'o',color='r',linewidth=6)
        # print("loc_err:",loc_err,"m")
        if loc_err_ours<1:
            num_1meter_ours+=1
        if loc_err_ours<0.5:
            num_halfmeter_ours+=1
        if loc_err_ours<0.1:
            num_01meter_ours+=1

        if loc_err_noisy<1:
            num_1meter_noisy+=1
        if loc_err_noisy<0.5:
            num_halfmeter_noisy+=1
        if loc_err_noisy<0.1:
            num_01meter_noisy+=1

        if loc_err_search<1:
            num_1meter_search+=1
        if loc_err_search<0.5:
            num_halfmeter_search+=1
        if loc_err_search<0.1:
            num_01meter_search+=1

        idx+=1
    print("Ours:")
    print("1m recall:",num_1meter_ours/idx)
    print("0.5m recall:",num_halfmeter_ours/idx)
    print("0.1m recall:",num_01meter_ours/idx)
    print("t_error:",total_loc_err_ours/idx)

    print("Noisy odometry")
    print("1m recall:",num_1meter_noisy/idx)
    print("0.5m recall:",num_halfmeter_noisy/idx)
    print("0.1m recall:",num_01meter_noisy/idx)
    print("t_error:",total_loc_err_noisy/idx)

    print("Only search")
    print("1m recall:",num_1meter_search/idx)
    print("0.5m recall:",num_halfmeter_search/idx)
    print("0.1m recall:",num_01meter_search/idx)
    print("t_error:",total_loc_err_search/idx)



    

