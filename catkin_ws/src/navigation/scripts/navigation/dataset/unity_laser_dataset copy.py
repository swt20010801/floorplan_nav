import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import json

from .zind_utils import *
from .s3d_utils import *
import pandas as pd
from torchvision import transforms
from PIL.Image import Image
from .unity_dataset import unityDataset
def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def rot_pano(pano, rot):
    pano_rot = np.zeros_like(pano)
    W = pano.shape[1]
    W_move = np.round(W * (rot % 360 / 360)).astype(np.int)
    pano_rot[:, : (W - W_move)] = pano[:, W_move:]
    pano_rot[:, (W - W_move) :] = pano[:, :W_move]
    return pano_rot

def persp2pano(img,fov=np.pi/2,size=1000):
    width =img.shape[1]
    height=img.shape[0]
    
    
    lon=np.arange(size)/size*fov-fov/2
    lat=np.arange(size)/size*fov-fov/2
    lon,lat=np.meshgrid(lon,lat)
    # print(lon.shape,lat.shape)
    R=128
    x=R*np.cos(lat)*np.sin(lon)
    y=-R*np.sin(lat)
    z=R*np.cos(lat)*np.cos(lon)

    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

    xyz=xyz/np.expand_dims(z,axis=-1) 
    # print(xyz)

    f=width/2/np.tan(fov/2)
    XY=(xyz*f)[:,:,:2].astype(np.float32)
    XY[:,:,0]=XY[:,:,0]+width/2
    XY[:,:,1]=-XY[:,:,1]+height/2

    return cv2.remap(img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

def compute_egomap_onedrt(depth,fov=90,scale=0.05,map_size=201,min_h=80,max_h=100,max_depth=3.5):
    '''
    depth:(1,1,h,w)
    '''
    
    h=depth.shape[2]
    w=depth.shape[3]
    assert h==w
    assert map_size%2==1
    f=(w/2)/np.tan(fov/2/180*np.pi)

    X=np.arange(0,w)-w/2
    Y=np.arange(h-1,-1,-1)-h/2

    X,Y=np.meshgrid(X,Y)
    Z=depth.squeeze(0).squeeze(0).cpu().numpy()
    invalid_mask=Z>max_depth
    X=Z*X/f
    Y=Z*Y/f
    XYZ=(np.stack([X,Y,Z],axis=0)/scale)
    XYZ[0]+=(map_size-1)/2
    XYZ[1]+=(map_size-1)/2
    XYZ=np.round(XYZ).astype(np.longlong)

    space=np.zeros((map_size,map_size,map_size))
    map=np.zeros((map_size,map_size))


    invalid_points=XYZ[:,invalid_mask]
    space[XYZ[2,:],XYZ[1,:],XYZ[0,:]]=1
    space[invalid_points[2,:],invalid_points[1,:],invalid_points[0,:]]=0

    map=np.max(space[:,min_h:max_h,:],axis=1)
    map=cv2.flip(map,0)
    return map

def compute_egomap(depth,fov=90,scale=0.05,map_size=201,min_h=80,max_h=100,max_depth=3.5):
    '''
    depth:(4,1,h,w)
    '''
    depth=depth.detach()
    whole_map=np.zeros((map_size,map_size))

    whole_map_0=np.zeros((map_size,map_size))
    whole_map_1=np.zeros((map_size,map_size))
    whole_map_2=np.zeros((map_size,map_size))
    whole_map_3=np.zeros((map_size,map_size))
    
    map_0=compute_egomap_onedrt(depth=depth[0:1,:,:,:],fov=fov,scale=scale,map_size=map_size,min_h=min_h,max_h=max_h,max_depth=max_depth)
    map_1=compute_egomap_onedrt(depth=depth[1:2,:,:,:],fov=fov,scale=scale,map_size=map_size,min_h=min_h,max_h=max_h,max_depth=max_depth)
    map_2=compute_egomap_onedrt(depth=depth[2:3,:,:,:],fov=fov,scale=scale,map_size=map_size,min_h=min_h,max_h=max_h,max_depth=max_depth)
    map_3=compute_egomap_onedrt(depth=depth[3:4,:,:,:],fov=fov,scale=scale,map_size=map_size,min_h=min_h,max_h=max_h,max_depth=max_depth)

    whole_map_0[0:(map_size-1)//2,(map_size-1)//4:(map_size-1)//4*3]=map_0[(map_size-1)//2:map_size-1,(map_size-1)//4:(map_size-1)//4*3]
    whole_map_1[0:(map_size-1)//2,(map_size-1)//4:(map_size-1)//4*3]=map_1[(map_size-1)//2:map_size-1,(map_size-1)//4:(map_size-1)//4*3]
    whole_map_2[0:(map_size-1)//2,(map_size-1)//4:(map_size-1)//4*3]=map_2[(map_size-1)//2:map_size-1,(map_size-1)//4:(map_size-1)//4*3]
    whole_map_3[0:(map_size-1)//2,(map_size-1)//4:(map_size-1)//4*3]=map_3[(map_size-1)//2:map_size-1,(map_size-1)//4:(map_size-1)//4*3]

    whole_map_1 = cv2.rotate(whole_map_1, cv2.ROTATE_90_CLOCKWISE)
    whole_map_2 = cv2.rotate(whole_map_2, cv2.ROTATE_180)
    whole_map_3 = cv2.rotate(whole_map_3, cv2.ROTATE_90_COUNTERCLOCKWISE)

    whole_map[(whole_map_0==1) | (whole_map_1==1) | (whole_map_2==1) | (whole_map_3==1)]=1
    # cv2.imwrite("map.png",whole_map*255.)

    return whole_map

class unityintentionDataset(Dataset):



    def __init__(
        self,
        dataset_dir,
        map_dir,
        image_size=256,
        fov=np.pi/2,
        is_training=False,
        line_sampling_interval=0.1,
        n_sample_points=None,
        return_empty_when_invalid=False,
        return_all_panos=False,
        training_set=[7],
        testing_set=[7],
        map_size=101,
        crop_map_size=71,
        scale=0.2,
        
    ):

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.fov=fov
        self.is_training = is_training
        self.line_sampling_interval = line_sampling_interval
        self.n_sample_points = n_sample_points
        self.return_empty_when_invalid = return_empty_when_invalid
        self.return_all_panos = return_all_panos
        self.Houseid_find=[]
        self.trajectoryid_find=[]
        self.Imageid_find=[]
        self.training_set = training_set    ########
        self.testing_set =  testing_set                ########
        self.N=0
        self.map_size=map_size
        self.crop_map_size=crop_map_size
        self.scale=scale
        self.House_list=[]
        
        if is_training:
            for house_id in self.training_set:
                house_dir=os.path.join(self.dataset_dir, f"House_{(house_id+1):02d}")
                trajectorys=os.listdir(house_dir)
                for trajectoryid in range(len(trajectorys)):
                        trajectory_path=os.path.join(house_dir,"ManualAgent"+str(trajectoryid))
                        images=os.listdir(trajectory_path)
                        images_num=(len(images)-1)//4
                        for Imageid in range(images_num):
                            self.Imageid_find.append(Imageid)
                            self.trajectoryid_find.append(trajectoryid)
                            self.Houseid_find.append(house_id)
                            self.N+=1
        else:
            for house_id in self.testing_set:
                house_dir=os.path.join(self.dataset_dir, f"House_{(house_id+1):02d}")
                trajectorys=os.listdir(house_dir)
                for trajectoryid in range(len(trajectorys)):
                        trajectory_path=os.path.join(house_dir,"ManualAgent"+str(trajectoryid))
                        images=os.listdir(trajectory_path)
                        images_num=(len(images)-1)//4
                        for Imageid in range(images_num):
                            self.Imageid_find.append(Imageid)
                            self.trajectoryid_find.append(trajectoryid)
                            self.Houseid_find.append(house_id)
                            self.N+=1
            np.random.seed(123456789)

        print(
            f"{self.N} samples loaded from {type(self)} dataset. (is_training={is_training})"
        )
        self.cite_dataset=unityDataset(map_dir)
    def __len__(self):
        return self.N

    def fetch_another(self):
        if self.return_empty_when_invalid:
            return {}
        else:
            return self.__getitem__(np.random.randint(self.N))

    def __getitem__(self, idx):


        House_id=self.Houseid_find[idx]+1
        Trajectory_id=self.trajectoryid_find[idx]
        Image_id=self.Imageid_find[idx]

        instance_path = os.path.join(self.dataset_dir, f"House_{House_id:02d}")
        Trajectory_path=os.path.join(instance_path,"ManualAgent"+str(Trajectory_id))
        image_path    =os.path.join(Trajectory_path,str(Image_id)+"_rgb.jpg")
        image=cv2.imread(image_path, cv2.IMREAD_COLOR)
        query_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        query_image -= (0.485, 0.456, 0.406)
        query_image /= (0.229, 0.224, 0.225)

        trajectory_log_path  =os.path.join(Trajectory_path,"LogImg.csv")
        cols=[1,3,5]

        trajectory_data = np.loadtxt(open(trajectory_log_path,"rb"),delimiter=",",skiprows=1,usecols=cols) 
        x    = trajectory_data[Image_id,0]
        y    =-trajectory_data[Image_id,1]
        theta= trajectory_data[Image_id,2]

        if Image_id==trajectory_data.shape[0]-1:
            x_1    = trajectory_data[Image_id,0]
            y_1    =-trajectory_data[Image_id,1]
            d_d=np.array([0])
            d_theta=np.array([0])
        else:
            x_1    = trajectory_data[Image_id+1,0]
            y_1    =-trajectory_data[Image_id+1,1]
            theta_1=trajectory_data[Image_id+1,2]

            d_d=np.sqrt((x_1-x)**2+(y_1-y)**2)
            # d_theta=np.arctan2(y_1-y,x_1-x)+np.pi/2-theta*np.pi/180
            # d_theta=(d_theta/np.pi*180)%360

            d_theta=theta_1-theta
            d_theta=(d_theta)%360

            if d_theta>180:
                d_theta=d_theta-360

        if d_d<1e-4 and d_theta<1e-4:#invalid state and action
            return self.fetch_another()

        x_goal    = trajectory_data[-1,0]
        y_goal    =-trajectory_data[-1,1]
        


        map_thick=7
        local_map_sz=301
        border=100
        point_sz=3

        floorplan_map_=self.cite_dataset.get_passable_space(self.Houseid_find[idx])
        floorplan_map=np.zeros_like(floorplan_map_)

        for i in range(floorplan_map_.shape[0]):
            for j in range(floorplan_map_.shape[1]):
                if floorplan_map_[i,j]==1:
                    floorplan_map[i-map_thick:i+map_thick+1,j-map_thick:j+map_thick+1]=1
        floorplan_map_border=np.zeros((floorplan_map.shape[0]+2*border,floorplan_map.shape[1]+2*border))
        floorplan_map_border[border:border+floorplan_map.shape[0],border:border+floorplan_map.shape[1]]=floorplan_map

        pos=self.cite_dataset.meter2pixel([x,y],self.Houseid_find[idx])
        pos_goal=self.cite_dataset.meter2pixel([x_goal,y_goal],self.Houseid_find[idx])
        local_map=np.zeros((local_map_sz,local_map_sz,3))
        x_min=pos[0]+border-(local_map_sz-1)//2
        y_min=pos[1]+border-(local_map_sz-1)//2        
        local_map[:,:,0]=floorplan_map_border[y_min:y_min+local_map_sz,x_min:x_min+local_map_sz]
        local_map[:,:,1]=floorplan_map_border[y_min:y_min+local_map_sz,x_min:x_min+local_map_sz]
        local_map[:,:,2]=floorplan_map_border[y_min:y_min+local_map_sz,x_min:x_min+local_map_sz]

        local_map[(local_map_sz-1)//2-point_sz:(local_map_sz-1)//2+point_sz+1,(local_map_sz-1)//2-point_sz:(local_map_sz-1)//2+point_sz+1]=[1,0,0]
        
        goal_pixel_x=pos_goal[0]-pos[0]+(local_map_sz-1)//2
        goal_pixel_y=pos_goal[1]-pos[1]+(local_map_sz-1)//2

        local_map[goal_pixel_y-point_sz:goal_pixel_y+point_sz+1,goal_pixel_x-point_sz:goal_pixel_x+point_sz+1]=[0,0,1]

        local_map=local_map#*255.

        M = cv2.getRotationMatrix2D(((local_map_sz-1)//2,(local_map_sz-1)//2), theta, 1)
        local_map=cv2.warpAffine(local_map, M, (local_map_sz, local_map_sz), borderValue=(0, 0, 0))

        local_map=cv2.resize(local_map,(self.image_size,self.image_size))
        query_image=cv2.resize(query_image,(self.image_size,self.image_size))

        local_map=np.transpose(local_map, (2, 0, 1))
        query_image=np.transpose(query_image, (2, 0, 1))

        input = {
            "query_image": query_image.astype(np.float32),
            "local_map":local_map.astype(np.float32),
            "d_d":d_d.astype(np.float32),
            "d_theta":d_theta.astype(np.float32),
        }  # B,D

        return input
    

        laser_path  =os.path.join(images_dir,"LogScan.csv")
        cols=np.arange(13,13+720).tolist()
        laser_data = np.loadtxt(open(laser_path,"rb"),delimiter=",",skiprows=1,usecols=cols) 

        laser=laser_data[idx]
        laser_num=laser.shape[0]
        ego_map=np.zeros((self.map_size,self.map_size))
        center=(self.map_size-1)//2
        max_distance=3
        for j in range(laser_num):
            distance=laser[j]
            if distance>max_distance:
                continue
            angle=j/laser_num*2*np.pi

            x=center-distance*np.sin(angle)/self.scale
            y=center-distance*np.cos(angle)/self.scale

            x=np.round(x).astype(np.int32)
            y=np.round(y).astype(np.int32)

            ego_map[y:y+1,x:x+1]=1

        ego_map=ego_map[(self.map_size-self.crop_map_size)//2:(self.map_size-self.crop_map_size)//2+self.crop_map_size,(self.map_size-self.crop_map_size)//2:(self.map_size-self.crop_map_size)//2+self.crop_map_size]

        rand_rot=np.random.rand() * 360#逆时针旋转
        if self.is_training:
            center = ((self.crop_map_size-1) // 2, (self.crop_map_size-1) // 2)
            M = cv2.getRotationMatrix2D(center, rand_rot, 1)

            ego_map = cv2.warpAffine(ego_map, M, (self.crop_map_size,self.crop_map_size))>0.5
        else:
            ego_map=ego_map

        image_path_0=os.path.join(images_dir,str(idx)+"_rgb.jpg")
        image_path_1=os.path.join(images_dir,str(idx)+"_rgb_1.jpg")
        image_path_2=os.path.join(images_dir,str(idx)+"_rgb_2.jpg")
        image_path_3=os.path.join(images_dir,str(idx)+"_rgb_3.jpg")

        image_0 = cv2.imread(image_path_0, cv2.IMREAD_COLOR)
        image_1 = cv2.imread(image_path_1, cv2.IMREAD_COLOR)
        image_2 = cv2.imread(image_path_2, cv2.IMREAD_COLOR)
        image_3 = cv2.imread(image_path_3, cv2.IMREAD_COLOR)

        assert image_0.shape==image_1.shape
        assert image_1.shape==image_2.shape
        assert image_2.shape==image_3.shape
        assert image_0.shape[0]==image_0.shape[1]#长宽相等
        # height=image_0.shape[0]
        # width =image_0.shape[1]
        # assert width%2==0
        pano=[]
        pano.append(persp2pano(image_0,self.fov,self.image_size))
        pano.append(persp2pano(image_1,self.fov,self.image_size))
        pano.append(persp2pano(image_2,self.fov,self.image_size))
        pano.append(persp2pano(image_3,self.fov,self.image_size))
        image=np.concatenate(pano,axis=1)
        #pano的最左边（首位）对应着机器人正面的朝向
        image=np.concatenate([image[:,self.image_size//2:,:],image[:,:self.image_size//2,:],],axis=1)


        if self.is_training:
            image = rot_pano(image, rand_rot)

            image=transforms.ToPILImage()(image) 
            image=transforms.ColorJitter(0.4, 0.4, 0.4)(image)
            image=np.array(image)        
        else:
            image=image
        query_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        query_image -= (0.485, 0.456, 0.406)
        query_image /= (0.229, 0.224, 0.225)
        query_image=np.transpose(query_image, (2, 0, 1))

        input = {
            "query_image": query_image.astype(np.float32),
            "ego_map":ego_map.astype(np.float32)
        }  # B,D

        return input

 