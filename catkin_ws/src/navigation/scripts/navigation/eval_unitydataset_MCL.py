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

gif_images = []

def init_particles_given_coords(numParticles, coords,initial_loc, init_weight=1.0):
  """ Initialize particles uniformly given the road coordinates.
    Args:
      numParticles: number of particles.
      coords: road coordinates
      initial_loc:给定初始位置
    Return:
      particles.
  """
  #initial location add random noise
  if torch.is_tensor(initial_loc):
      initial_loc=initial_loc.cpu().numpy()
  if torch.is_tensor(coords):
      coords=coords.cpu().numpy()


  LOC_NOISE = [0.05, 0.05]
  xNoise = LOC_NOISE[0]
  yNoise = LOC_NOISE[1]

  initial_loc[0] = initial_loc[0] + xNoise * np.random.randn(1)
  initial_loc[1] = initial_loc[1] + yNoise * np.random.randn(1)


  particles = []
  near_particles=[]
  for i in range(coords.shape[0]):
      for j in range(coords.shape[1]):
          if(abs(coords[i,j,0]-initial_loc[0])<0.5 and abs(coords[i,j,1]-initial_loc[1])<0.5):
              near_particles.append(coords[i,j])


  rand = np.random.rand
  args_coords = np.arange(len(near_particles))
  selected_args = np.random.choice(args_coords, numParticles)
  
  for i in range(numParticles):
    x = near_particles[selected_args[i]][0]
    y = near_particles[selected_args[i]][1]
    # theta = 2 * np.pi * rand(1)
    theta = -np.pi + 2 * np.pi * rand(1)
    particles.append([x, y, theta, init_weight])
  
  return np.array(particles, dtype=float)

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
  motion[0]=np.arctan2(gt_loc[1]-last_gt_loc[1],gt_loc[0]-last_gt_loc[0])+np.pi/2-last_gt_rot#旋转1
  motion[2]=gt_rot-(np.arctan2(gt_loc[1]-last_gt_loc[1],gt_loc[0]-last_gt_loc[0])+np.pi/2)#旋转2

  return motion


def move_particles(particles, motion):
    if torch.is_tensor(motion):
        motion=motion.cpu().numpy()
    
    num_particles = len(particles)

    MOTION_NOISE = [0.01, 0.1]

    r1Noise = MOTION_NOISE[0]
    transNoise = MOTION_NOISE[1]

    rot1 = motion[0] + r1Noise * np.random.randn(num_particles)#顺时针
    rot2 = motion[2] + r1Noise * np.random.randn(num_particles)#顺时针
    tras1 = motion[1] + transNoise * np.random.randn(num_particles)

    # update pose using motion model角度顺时针为正方向，以y轴负半轴为0°
    particles[:, 0] += tras1 * np.cos(particles[:, 2] + rot1 - np.pi/2) #x
    particles[:, 1] += tras1 * np.sin(particles[:, 2] + rot1 - np.pi/2) #y
    particles[:, 2] += rot1+rot2 #theta


    return particles

def align_particles(particles,grid_coords):
    if torch.is_tensor(grid_coords):
        grid_coords=grid_coords.cpu().numpy()

    grid_coords=grid_coords.reshape(-1,2)#展平
    distances=np.expand_dims(particles[:,0:2],axis=1)-np.expand_dims(grid_coords,axis=0) #(p,1,2) - (1,g,2) = (p,g,2)
    distances=np.linalg.norm(distances,axis=2) #(p,g)
    nearest_idx=np.argmin(distances,axis=1)
    particles[:,0:2]=grid_coords[nearest_idx]
    return particles,nearest_idx
    
def resample(particles):
  """ Low variance re-sampling.
  """
  weights = particles[:, 3]
  
  # normalize the weights
  weights = weights / sum(weights)
  
  # compute effective number of particles
  eff_N = 1 / sum(weights ** 2)
  
  # resample
  new_particles = np.zeros(particles.shape)
  i = 0
  if eff_N < len(particles)*1.0/2.0:
    r = np.random.rand(1) * 1.0/len(particles)
    c = weights[0]
    for idx in range(len(particles)):
      u = r + idx/len(particles)
      while u > c:
        if i >= len(particles) - 1:
          break
        i += 1
        c += weights[i]
      new_particles[idx] = particles[i]
  else:
    new_particles = particles
    
  return new_particles


# @torch.no_grad()
def update_particles(particles,particles_feat, data, model, cfg, sample_nrots=16,heading_mask=True):
    
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

    if heading_mask:
        range_len=5
        out_range_idx=np.round(particles[:,2]/(2*np.pi)*sample_nrots).astype(int)
        m_idx=np.arange(range_len)-(range_len-1)//2
        out_range_idx=(np.expand_dims(out_range_idx,axis=1)+np.expand_dims(m_idx,axis=0))%sample_nrots
        idx_0=np.repeat(np.expand_dims(np.arange(particles.shape[0]),axis=1),range_len,axis=1)
        out_range_mask=np.zeros((particles.shape[0],sample_nrots))
        out_range_mask[idx_0.flatten(),out_range_idx.flatten()]=1
        score_list=score_list*torch.from_numpy(out_range_mask).to(device=score_list.device)
    scores, matched_rot_idxs = score_list.max(dim=-1)#对环形特征子进行16个角度的旋转后及逆行比对，找到各个位置相似度最高的旋转角度
    particles[:,3]*=scores.squeeze(0).detach().cpu().numpy()
    particles[:, 3] = particles[:, 3] / np.max(particles[:, 3])

    particles[:,2] = rot_samples[matched_rot_idxs.squeeze(0)].cpu().numpy() / 180 * np.pi
    return particles


       





def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

folder_name="results_norefine_MCL"

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
    model.load_state_dict(torch.load("/home/swt/catkin_ws/src/communication/scripts/laser/try_wo_scheduler.pth"))

    eval_dataset=unityDataset(dataset_dir="/home/swt/catkin_ws/src/communication/scripts/laser/unitydataset",is_training=False,n_sample_points=2048,testing_set=[7])
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
    for eval_data in eval_dataloader:
        for k in eval_data.keys():
            if torch.is_tensor(eval_data[k]) and not eval_data[k].is_cuda:
                eval_data[k] = eval_data[k].cuda()

        if cfg["disable_semantics"]:
            eval_data["bases_feat"][..., -2:] = 0
        if idx==0:
            sample_ret = sample_floorplan(
                eval_data,
                model,
                cfg,
                sample_grid=0.1,
                batch_size=256,
            )
            grid_shape=sample_ret["samples_original_shape"]
            grid_coords=sample_ret["samples_loc"].reshape((grid_shape[0],grid_shape[1],2))
            #initialize particles
            particles =init_particles_given_coords(500,grid_coords,eval_data["gt_loc"][0])
            
            last_gt_loc=eval_data["gt_loc"][0]
            last_gt_rot=eval_data["gt_rot"][0]

        else:
            motion=get_motion(last_gt_loc,last_gt_rot,eval_data["gt_loc"][0],eval_data["gt_rot"][0])
            last_gt_loc=eval_data["gt_loc"][0]
            last_gt_rot=eval_data["gt_rot"][0]
            particles=move_particles(particles,motion)

        particles,nearest_idx=align_particles(particles,grid_coords)#与采样网格对齐
        particles_feat=sample_ret["samples_feat"][0:1,nearest_idx,:,:]        
        if idx==0:
            particles=update_particles(particles,particles_feat,eval_data,model,cfg,sample_nrots=16,heading_mask=False)
        else:
            particles=update_particles(particles,particles_feat,eval_data,model,cfg,sample_nrots=16,heading_mask=True)

        particles = resample(particles)


        best_match_idx=np.argmax(particles[:,3])

        loc_gt=eval_data["gt_loc"][0].cpu().numpy()
        rot_gt=eval_data["gt_rot"][0][0].cpu().numpy()

        loc_est=particles[best_match_idx,0:2]
        rot_est=particles[best_match_idx,2]
        result_viz = render_result(
            eval_data["bases"][0].cpu().numpy(),
            eval_data["bases_feat"][0].cpu().numpy(),
            loc_gt,
            loc_est,
            rot_gt=rot_gt,
            rot_est=rot_est,
        )
        cv2.imwrite(
            # os.path.join("eval_MCL",f"result_{idx:04d}.png"),
            os.path.join("MCL_pics/MCL"+str(idx)+".png"),

            result_viz,
        )
        gif_images.append(cv2.cvtColor(result_viz, cv2.COLOR_BGR2RGB))
        idx+=1

    imageio.mimsave("House_7_MCL.gif", gif_images, fps=5)   # 转化为gif动画



