
import cv2
import torch
import numpy as np
from CANN.pose_cell import pose_cell
from CANN.hd_cell import hd_cell
import torch.nn.functional as F

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

def CANN_loc(pano_image,device,model,cfg,grid_feat,samples_loc,rot_samples,grid_shape,pose_cell:pose_cell,hd_cell:hd_cell,grid_coords,motion):

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

    return now_loc_est,rot_est