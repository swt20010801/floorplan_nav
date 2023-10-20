#!/usr/bin/env /home/swt/anaconda3/envs/LLL/bin/python
from dataset.unity_dataset import unityDataset
from dataset.unity_dataset import persp2pano

import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from model.fplocnet import FpLocNet, quick_fplocnet_call
from eval_utils import *
import os
import pickle
import imageio
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose
from tf2_msgs.msg import TFMessage
from model.pano2ego import pano2ego
from RRT_star.RRT_star import *
import onnxruntime

def render_map(bases, bases_feat,eval_dataset,house_id,border=30):#scale：1m对应多少个像素点
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

def Dijkstra(network, s, d):  # 迪杰斯特拉算法算s-d的最短路径，并返回该路径和值
    # print("Start Dijstra Path……")
    path = []  # 用来存储s-d的最短路径
    n = len(network)  # 邻接矩阵维度，即节点个数
    fmax = float('inf')
    w = [[0 for _ in range(n)] for j in range(n)]  # 邻接矩阵转化成维度矩阵，即0→max

    book = [0 for _ in range(n)]  # 是否已经是最小的标记列表
    dis = [fmax for i in range(n)]  # s到其他节点的最小距离
    book[s - 1] = 1  # 节点编号从1开始，列表序号从0开始
    midpath = [-1 for i in range(n)]  # 上一跳列表
    for i in range(n):
      for j in range(n):
        if network[i][j] != 0:
          w[i][j] = network[i][j]  # 0→max
        else:
          w[i][j] = fmax
        if i == s - 1 and network[i][j] != 0:  # 直连的节点最小距离就是network[i][j]
          dis[j] = network[i][j]
    for i in range(n - 1):  # n-1次遍历，除了s节点
      min = fmax
      for j in range(n):
        if book[j] == 0 and dis[j] < min:  # 如果未遍历且距离最小
          min = dis[j]
          u = j
      book[u] = 1
      for v in range(n):  # u直连的节点遍历一遍
        if dis[v] > dis[u] + w[u][v]:
          dis[v] = dis[u] + w[u][v]
          midpath[v] = u + 1  # 上一跳更新
    j = d - 1  # j是序号
    path.append(d)  # 因为存储的是上一跳，所以先加入目的节点d，最后倒置
    while (midpath[j] != -1):
      path.append(midpath[j])
      j = midpath[j] - 1
    path.append(s)
    path.reverse()  # 倒置列表

    return path
    print("path:",path)
    # print(midpath)
    print("dis:",dis)
    # return path

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class ONNXModel:
    def __init__(self, encoder_onnx_path,distribute_onnx_path):

        self.encoder_onnx_session = onnxruntime.InferenceSession(encoder_onnx_path)
        self.distribute_onnx_session = onnxruntime.InferenceSession(distribute_onnx_path)


    def compute(self,rgb, pointgoal, rnn_hidden_states_1, prev_actions, masks):
        rgb=to_numpy(rgb).astype(np.uint8)
        pointgoal=to_numpy(pointgoal).astype(np.float32) 
        rnn_hidden_states_1=to_numpy(rnn_hidden_states_1).astype(np.float32) 
        prev_actions=to_numpy(prev_actions).astype(np.int64)
        masks=to_numpy(masks).astype(bool)        
        # print(rgb.shape,masks,pointgoal)

        input_feed={'rgb':rgb, 'pointgoal':pointgoal, 'rnn_hidden_states.1':rnn_hidden_states_1, 'prev_actions':prev_actions, 'masks':masks}
        output_name=['features','rnn_hidden_states']
        features, rnn_hidden_states = self.encoder_onnx_session.run(output_name, input_feed=input_feed)

        input_feed={'features':features}
        output_name=['action']
        action = self.distribute_onnx_session.run(output_name, input_feed=input_feed)

        return torch.from_numpy(action[0]),torch.from_numpy(rnn_hidden_states)
    

gif_images = []

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
MOTION_NOISE = [0.08, 0.1]
LOC_NOISE    = [0.05, 0.05]

START=1
GOAL=2

WO_FLOORPLAN_FLAG=False

SAVE_TRAJECTORY_FLAG=False

if WO_FLOORPLAN_FLAG:
   MOTION_NOISE=[0.01,0.02]

house_node_list_dict={
    "House_8":
    [
    torch.Tensor([-1.166203-4.65,-(2.246365+7.23)]),#1
    torch.Tensor([-1.166203-4.78,-(2.246365+0.08999991)]),#2
    torch.Tensor([-1.166203-3.97,-(2.246365+4.87)]),#3
    torch.Tensor([-1.166203-2.9,-(2.246365+2.12)]),#4
    torch.Tensor([-1.166203-0.8499999,-(2.246365+1.93)]),#5
    torch.Tensor([-1.166203+0.98,-(2.246365+0.9200001)]),#6
    torch.Tensor([-1.166203-5.31,-(2.246365-1.02)]),#7
    torch.Tensor([-1.166203-0.46,-(2.246365-1.42)]),#8

    ]
    ,
    "House_12":
    [
    torch.Tensor([-1.166203-2.791,-(2.246365-1.829)]),#1
    torch.Tensor([-1.166203-2.051063,-(2.246365-4.021405)]),#2
    torch.Tensor([-1.166203-2.494206,-(2.246365-5.704995)]),#3
    torch.Tensor([-1.166203-4.322511,-(2.246365-7.217872)]),#4
    torch.Tensor([-1.166203-2.618146,-(2.246365-7.138311)]),#5
    torch.Tensor([-1.166203+1.286348,-(2.246365-5.886983)]),#6
    torch.Tensor([-1.166203+2.02355,-(2.246365-2.612659)]),#7
    ]
    }

house_color_id_dict={
    "House_8":
    [
    [0,255,255],#1
    [0,0,255],
    [255,0,0],
    [255,0,255],
    [0,255,0],
    [255,255,0],
    [128,128,0],
    [128,0,128],#8
    [0,128,128],#9
    ]
    ,
   "House_12":
   [
   [0,255,255],         #1
   [0,0,255],
   [255,0,255],
   [255,0,0],
   [255,255,0],
   [255,128,128],
   [0,255,0]            #7
    ]
    }

house_node_toplogic_dict = {
    "House_8":
[
    [0,1,0,0,0,0,0,0,0],#1
    [1,0,1,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,0,0],#3
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,1],#5
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,1],#7
    [0,0,0,0,0,0,0,0,1],
    [0,1,0,1,1,1,1,1,0],#9

]
    ,
    "House_12":
    [
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
    ]
}
house_node_toplogic_edge_dict = {
    "House_8":
    [
        [0,[[121,158],[121,243]],0,0,0,0,0,0,0],#1
        [[[121,243],[121,158]],0,[[121,243],[172,243]],0,0,0,0,0,[[99,474],[141,500]]],
        [0,[[172,243],[121,243]],0,0,0,0,0,0,0],#3
        [0,0,0,0,0,0,0,0,[[238,441],[238,498]]],
        [0,0,0,0,0,0,0,0,[[381,443],[381,499]]],#5
        [0,0,0,0,0,0,0,0,[[457,497],[381,499]]],
        [0,0,0,0,0,0,0,0,[[238,556],[238,498]]],#7
        [0,0,0,0,0,0,0,0,[[381,559],[381,499]]],
        [0,[[141,500],[99,474]],0,[[238,498],[238,441]],[[381,499],[381,443]],[[381,499],[457,497]],[[238,498],[238,556]],[[381,499],[381,559]],0]#9
    ]
    ,
    "House_12":
    [
        [0, 0, 0, 0, 0, [[574,242],[574,372]], 0],
        [0, 0, 0, 0, 0, [[472,372],[574,372]], 0],
        [0, 0, 0, 0, 0, [[357,652],[460,652]], 0],
        [0, 0, 0, 0, [[137,823],[220,823]], 0, 0],
        [0, 0, 0, [[220,823],[137,823]], 0, [[459,758],[460,652]], 0],
        [[[574,372],[574,242]], [[574,372],[472,372]], [[641,652],[357,652]], 0, [[460,652],[459,758]], 0, [[574,372],[692,370]]],
        [0, 0, 0, 0, 0, [[692,370],[574,372]], 0],
    ]
}
bridge = CvBridge()
observation_shape=(3,256,256)
observation_goal_distance_shape=1
observation_goal_angle_shape   =1

action_space=3
num_processes=1

pub_stay=Pose()
pub_stay.position.x=0
pub_stay.position.y=0
pub_stay.position.z=0

pub_forward=Pose()
pub_forward.position.x=0
pub_forward.position.y=0
pub_forward.position.z=0.25

pub_turnleft=Pose()
pub_turnleft.position.x=0
pub_turnleft.position.y=-10
pub_turnleft.position.z=0

pub_turnright=Pose()
pub_turnright.position.x=0
pub_turnright.position.y= 10
pub_turnright.position.z=0

pub_stop_send=Pose()
pub_stop_send.position.x=0
pub_stop_send.position.y=0
pub_stop_send.position.z=-1.0

def process_msg(msg):
    buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
    image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,observation_shape[1:3]).astype(np.float32)
    # image=image/255.
    # image -= (0.485, 0.456, 0.406)
    # image /= (0.229, 0.224, 0.225)
    image=np.transpose(image, (2, 0, 1))
    image=torch.from_numpy(image)

    return image

def reset():
    pub = rospy.Publisher('/pose_info', Pose,queue_size=1000)
    
    for i in range(10):
        rospy.sleep(0.1)     
        pub.publish(pub_stay) 

    msg = rospy.wait_for_message("/RobotAtVirtualHome/VirtualCameraRGB", CompressedImage, timeout=None)
    # info=msg.header.frame_id
    # goal_distance,goal_angle=float(info[0:6]),float(info[7:])
    # assert info[6]==";"
    image_0=process_msg(msg)

    # pub.publish(pub_stay) 
    msg = rospy.wait_for_message("/RobotAtVirtualHome/VirtualCameraRGB_1", CompressedImage, timeout=None)
    image_1=process_msg(msg)

    # pub.publish(pub_stay) 
    msg = rospy.wait_for_message("/RobotAtVirtualHome/VirtualCameraRGB_2", CompressedImage, timeout=None)
    image_2=process_msg(msg)

    # pub.publish(pub_stay) 
    msg = rospy.wait_for_message("/RobotAtVirtualHome/VirtualCameraRGB_3", CompressedImage, timeout=None)
    image_3=process_msg(msg)

    pub.publish(pub_stop_send) 

    image=torch.stack([image_0,image_1,image_2,image_3],dim=0)

    return image#,goal_distance,goal_angle


def step_(action):
    assert action in [0,1,2,3]
    pub = rospy.Publisher('/pose_info', Pose,queue_size=10)
    
    pub_topic=Pose()

    if action==0:
        pub.publish(pub_forward)
    elif action==1:
        pub.publish(pub_turnleft)
    elif action==2:
        pub.publish(pub_turnright)
    elif action==3:
        pub.publish(pub_stay)


    msg = rospy.wait_for_message("/RobotAtVirtualHome/VirtualCameraRGB", CompressedImage, timeout=None)
    # info=msg.header.frame_id
    # goal_distance,goal_angle=float(info[0:6]),float(info[7:])
    # assert info[6]==";"
    image_0=process_msg(msg)
    # rospy.sleep(0.1)     

    # pub.publish(pub_stay) 
    msg = rospy.wait_for_message("/RobotAtVirtualHome/VirtualCameraRGB_1", CompressedImage, timeout=None)
    image_1=process_msg(msg)
    # rospy.sleep(0.1)     

    # pub.publish(pub_stay) 
    msg = rospy.wait_for_message("/RobotAtVirtualHome/VirtualCameraRGB_2", CompressedImage, timeout=None)
    image_2=process_msg(msg)
    # rospy.sleep(0.1)     

    # pub.publish(pub_stay) 
    msg = rospy.wait_for_message("/RobotAtVirtualHome/VirtualCameraRGB_3", CompressedImage, timeout=None)
    image_3=process_msg(msg)
    # rospy.sleep(0.1)     
    pub.publish(pub_stop_send) 

    image=torch.stack([image_0,image_1,image_2,image_3],dim=0)

    return image#,goal_distance,goal_angle

def set_robot_position(position):

    pub = rospy.Publisher('/pose_set', Pose,queue_size=10)
    
    pub_topic=Pose()
    pub_topic.position.x=position[0]
    pub_topic.position.y=0
    pub_topic.position.z=-position[1]
    # print("set robot position")
    for i in range(10):
        rospy.sleep(0.1)     
        pub.publish(pub_topic) 

def get_robot_position():
    robot_pos=np.zeros(2)

    msg = rospy.wait_for_message("/tf", TFMessage, timeout=None)

    robot_pos[0]=  msg.transforms[0].transform.translation.x+msg.transforms[1].transform.translation.x
    robot_pos[1]=-(msg.transforms[0].transform.translation.y+msg.transforms[1].transform.translation.y)

    return robot_pos
    
    

def trans_goal_pos(x,y,goal_distance,goal_angle):

    x=x+goal_distance*np.sin(goal_angle)
    y=y-goal_distance*np.cos(goal_angle)

    x=np.round(x).astype(np.int64)
    y=np.round(y).astype(np.int64)
    return x,y

def image_2_pano(image,fov=np.pi/2,image_size=256):
    """
    image : Tensor(4,3,h,w)
    """
    image=image.permute(0,2,3,1).cpu().numpy()

    # image  = image* (0.229, 0.224, 0.225)
    # image += (0.485, 0.456, 0.406)
    # image *= 255.

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
  
  motion=np.zeros(2)
  motion[1]=np.linalg.norm(gt_loc-last_gt_loc)#距离
  motion[0]=np.arctan2(gt_loc[1]-last_gt_loc[1],gt_loc[0]-last_gt_loc[0])+np.pi/2-last_gt_rot#旋转

  return motion


def move_particles(particles, motion):
    if torch.is_tensor(motion):
        motion=motion.cpu().numpy()
    
    num_particles = len(particles)


    r1Noise = MOTION_NOISE[0]
    transNoise = MOTION_NOISE[1]

    rot1 = motion[0] + r1Noise * np.random.randn(num_particles)#顺时针
    rot2 = 0#motion[2] + r1Noise * np.random.randn(num_particles)#顺时针
    tras1 = motion[1] + transNoise * np.random.randn(num_particles)

    # update pose using motion model角度顺时针为正方向，以y轴负半轴为0°
    particles[:, 0] += tras1 * np.cos(particles[:, 2] + rot1 - np.pi/2) #x
    particles[:, 1] += tras1 * np.sin(particles[:, 2] + rot1 - np.pi/2) #y
    particles[:, 2] += rot1+rot2 #theta


    return particles

def align_particles(particles,grid_coords):
    if torch.is_tensor(grid_coords):
        grid_coords=grid_coords.cpu().numpy()

    nearest_idx=np.round((particles[:,0:2]-grid_coords[0,0])/0.1).astype(int)

    nearest_idx[:,0]=np.clip(nearest_idx[:,0],0,grid_coords.shape[1]-1)
    nearest_idx[:,1]=np.clip(nearest_idx[:,1],0,grid_coords.shape[0]-1)

    nearest_idxx=nearest_idx[:,0]+nearest_idx[:,1]*grid_coords.shape[1]
    grid_coords=grid_coords.reshape(-1,2)#展平

    particles[:,0:2]=grid_coords[nearest_idxx]

    # distances=np.expand_dims(particles[:,0:2],axis=1)-np.expand_dims(grid_coords,axis=0) #(p,1,2) - (1,g,2) = (p,g,2)
    # distances=np.linalg.norm(distances,axis=2) #(p,g)
    # nearest_idx=np.argmin(distances,axis=1)
    # particles[:,0:2]=grid_coords[nearest_idx]
    return particles,nearest_idxx

# def align_particles(particles,grid_coords):
#     if torch.is_tensor(grid_coords):
#         grid_coords=grid_coords.cpu().numpy()

#     grid_coords=grid_coords.reshape(-1,2)#展平
#     distances=np.expand_dims(particles[:,0:2],axis=1)-np.expand_dims(grid_coords,axis=0) #(p,1,2) - (1,g,2) = (p,g,2)
#     distances=np.linalg.norm(distances,axis=2) #(p,g)
#     nearest_idx=np.argmin(distances,axis=1)
#     particles[:,0:2]=grid_coords[nearest_idx]
#     return particles,nearest_idx
    
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

    particles[:,2] = rot_samples[matched_rot_idxs.cpu().squeeze(0)].cpu().numpy() / 180 * np.pi
    
    return particles


       

def get_topo_goal(topologic_node,global_route,start,end):

    topologic_node_=np.expand_dims(topologic_node,axis=0)
    global_route_=np.expand_dims(global_route,axis=1)

    d_vector=np.linalg.norm(topologic_node_-global_route_,axis=-1)
    nearest_topo_id=np.argmin(d_vector,axis=1)

    idx=-1
    first_node_flag=False
    second_node_flag=False

    while(-idx<=len(global_route)):
        if d_vector[idx,nearest_topo_id[idx]]<60:
            if first_node_flag==False:
                first_node_id=nearest_topo_id[idx]
                first_node_flag=True
            
            if first_node_flag==True:
                if nearest_topo_id[idx]!=first_node_id:
                    second_node_id=nearest_topo_id[idx]
                    second_node_flag=True
                    break
        idx-=1
    
    if first_node_flag==False and second_node_flag==False:
        return end
    elif first_node_flag==True and second_node_flag==False:
        first_node=topologic_node[first_node_id]
        second_node=end
    else:
        first_node=topologic_node[first_node_id]
        second_node=topologic_node[second_node_id]

    v_2fisrtnode=first_node-start
    v_2secondnode=second_node-start

    if np.dot(v_2fisrtnode,v_2secondnode)<0 or np.linalg.norm(v_2fisrtnode)<30:
        local_goal=second_node

        # print("second",second_node_id)
    else:
        local_goal=first_node

        # print("first",first_node_id)
    return local_goal





def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)





if __name__ == "__main__":
    from fastdepth import models,main
    from astar.a_star import *
    house_id=7 #0,1,2,3,4...

    House_name="House_"+str(house_id+1)
    house_color_id=house_color_id_dict[House_name]
    house_node_toplogic=house_node_toplogic_dict[House_name]
    house_node_toplogic_edge=house_node_toplogic_edge_dict[House_name]
    house_node_list=house_node_list_dict[House_name]

    House_color_map=cv2.imread("/home/swt/Floorplan_nav/catkin_ws/src/navigation/scripts/laser/unitydataset/House_"+str(house_id+1)+"_color.png")
    House_color_map=cv2.cvtColor(House_color_map, cv2.COLOR_BGR2RGB)


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

    model.load_state_dict(torch.load("/home/swt/Floorplan_nav/catkin_ws/src/navigation/scripts/laser/try_wo_scheduler.pth"))


    action_model=ONNXModel('/home/swt/onnx_test/pointgoal_ddppo.onnx','/home/swt/onnx_test/action_distribute_ddppo.onnx')

    eval_dataset=unityDataset(dataset_dir="/home/swt/Floorplan_nav/catkin_ws/src/navigation/scripts/laser/unitydataset",
                              is_training=False,n_sample_points=2048,testing_set=[house_id])
    eval_dataloader=DataLoader(dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )

    # for aaa in range(len(house_node_toplogic_edge)):
    #     pp=house_node_toplogic_edge[aaa]
    #     for x in range(len(pp)):
    #         if pp[x]!=0:
    #             cv2.circle(floorplan_map,(pp[x][0][0],pp[x][0][1]),radius=10,color=(255,0,0))
    #             cv2.circle(floorplan_map,(pp[x][1][0],pp[x][1][1]),radius=10,color=(255,0,0))

    # cv2.imshow("show",floorplan_map)
    # cv2.waitKey(0)

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
        grid_shape=sample_ret["samples_original_shape"]
        grid_coords=sample_ret["samples_loc"].reshape((grid_shape[0],grid_shape[1],2))
        break
    
    if SAVE_TRAJECTORY_FLAG:
        floorplan_map_trajectory=render_map(eval_data["bases"][0].cpu().numpy(),eval_data["bases_feat"][0].cpu().numpy(),eval_dataset,house_id)
        floorplan_map_estimation = np.copy(floorplan_map_trajectory)

    # cv2.imshow("map",floorplan_map)
    # cv2.waitKey(0)
    for ii in range(len(house_node_list)):
       for jj in range(len(house_node_list)):
            if ii==jj:
               continue


            START=ii+1
            GOAL =jj+1
            print("START:",START,"GOAL:",GOAL)

            set_robot_position(house_node_list[-1+START])

            #initialize particles
            initial_loc=house_node_list[-1+START]
            goal_loc   =house_node_list[-1+GOAL]

            if SAVE_TRAJECTORY_FLAG:
                cv2.drawMarker(floorplan_map_trajectory, tuple(np.round(eval_dataset.meter2pixel(initial_loc,house_id)).astype(np.int)), [255,0,0],cv2.MARKER_SQUARE,6,3)
                cv2.drawMarker(floorplan_map_trajectory, tuple(np.round(eval_dataset.meter2pixel(goal_loc,house_id)).astype(np.int)), [0,255,0],cv2.MARKER_SQUARE,6,3)


            particles =init_particles_given_coords(500,grid_coords,initial_loc)


            step=0
            goal_distance=100
            image=reset()
            last_loc_gt=get_robot_position()
            trajectory_len=0
            while(1):

            ################# localization ####################

                particles,nearest_idx=align_particles(particles,grid_coords)#与采样网格对齐

                particles_feat=sample_ret["samples_feat"][0:1,nearest_idx,:,:]  
                pano_image=image_2_pano(image)

                data={"gt_fov":     torch.Tensor([[360]]).float(),
                    "query_image":pano_image.to(device).float()
                    }      
                
                if step==0:
                    particles=update_particles(particles,particles_feat,data,model,cfg,sample_nrots=16,heading_mask=False)
                else:
                    if WO_FLOORPLAN_FLAG==False:
                        particles=update_particles(particles,particles_feat,data,model,cfg,sample_nrots=16,heading_mask=True)
                
                particles = resample(particles)


                best_match_idx=np.argmax(particles[:,3])

                loc_est=particles[best_match_idx,0:2]
                rot_est=particles[best_match_idx,2]

                # result_viz = render_result(
                #     eval_data["bases"][0].cpu().numpy(),
                #     eval_data["bases_feat"][0].cpu().numpy(),
                #     loc_gt,
                #     loc_est,
                #     rot_est=rot_est
                # )
                # cv2.imwrite(
                #     os.path.join("MCL.png"),
                #     result_viz,
                # )
                ##########################

                ############planner########################

                if step%100==0 or goal_distance<0.2:#到达subpoint
                    if torch.is_tensor(loc_est):
                        loc_est=loc_est.cpu().numpy()
                    if torch.is_tensor(goal_loc):
                        goal_loc=goal_loc.cpu().numpy()

                    loc_est_pixel=eval_dataset.meter2pixel(loc_est,house_id)
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

                        if np.sqrt((first_node[0]-loc_est[0])**2+(first_node[1]-loc_est[1])**2)<0.2:
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

                goal_distance=np.sqrt((local_goal[0]-loc_est[0])**2+(local_goal[1]-loc_est[1])**2)
                goal_angle=np.arctan2(local_goal[1]-loc_est[1],local_goal[0]-loc_est[0])+np.pi/2-rot_est

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

                particles=move_particles(particles,motion)

                step+=1

                now_loc_gt=get_robot_position()

                if SAVE_TRAJECTORY_FLAG:

                    last_loc_gt_pixel=eval_dataset.meter2pixel(last_loc_gt,house_id)
                    now_loc_gt_pixel=eval_dataset.meter2pixel(now_loc_gt,house_id)
                    cv2.line(floorplan_map_trajectory,tuple(np.round(last_loc_gt_pixel).astype(np.int)),tuple(np.round(now_loc_gt_pixel).astype(np.int)),  [0, 0, 255],3)

                
                trajectory_len+=np.sqrt((now_loc_gt[0]-last_loc_gt[0])**2+(now_loc_gt[1]-last_loc_gt[1])**2)
                last_loc_gt=now_loc_gt

                # print(np.sqrt((goal_loc[0]-now_loc_gt[0])**2+(goal_loc[1]-now_loc_gt[1])**2))
                if np.sqrt((goal_loc[0]-now_loc_gt[0])**2+(goal_loc[1]-now_loc_gt[1])**2)<0.2:
                    if SAVE_TRAJECTORY_FLAG:
                        cv2.imwrite("ours_House_08_1_7_trajectory.png",floorplan_map_trajectory)

                    print("finally move ",trajectory_len,"m !!!!")
                    break
                if step>500:
                    print("nav failed!!!")
                    break

