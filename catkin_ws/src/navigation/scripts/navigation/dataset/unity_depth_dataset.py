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

class unitydepthDataset(Dataset):



    def __init__(
        self,
        dataset_dir,
        image_size=256,
        fov=np.pi/2,
        is_training=False,
        line_sampling_interval=0.1,
        n_sample_points=None,
        return_empty_when_invalid=False,
        return_all_panos=False,
        training_set=[0,1,2,3,4,5,6,8,9,10],
        testing_set=[7,11],
        map_size=101,
        crop_map_size=71,
        scale=0.2
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
        self.Imageid_find=[]
        self.training_set = training_set    ########
        self.testing_set =  testing_set                ########
        self.N=0
        self.map_size=map_size
        self.crop_map_size=crop_map_size
        self.scale=scale
        self.House_list=[]
        for i in range(1,13):
           img_path = os.path.join(self.dataset_dir, f"House_{i:02d}","images")
           images=os.listdir(img_path)
           self.House_list.append((len(images)-1)//8)

        if is_training:
            for id in self.training_set:
                self.N+=self.House_list[id]

                for i in range(self.House_list[id]):
                    self.Houseid_find.append(id)
                    self.Imageid_find.append(i)

        else:
            for id in self.testing_set:
                self.N+=self.House_list[id]
                for i in range(self.House_list[id]):
                    self.Houseid_find.append(id)
                    self.Imageid_find.append(i)

            np.random.seed(123456789)
        print(self.House_list)
        print(
            f"{self.N} samples loaded from {type(self)} dataset. (is_training={is_training})"
        )

    def __len__(self):
        return self.N

    def fetch_another(self):
        if self.return_empty_when_invalid:
            return {}
        else:
            return self.__getitem__(np.random.randint(self.N))

    def __getitem__(self, idx):


        House_id=self.Houseid_find[idx]+1
        idx=self.Imageid_find[idx]
        instance_path = os.path.join(self.dataset_dir, f"House_{House_id:02d}")
  
        images_dir    =os.path.join(instance_path,"images")
        ego_map_dir   =os.path.join(instance_path,"ego_map")
        ego_map_path  =os.path.join(ego_map_dir,str(idx)+".npy")
        ego_map=np.load(ego_map_path)

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

    def generate_ego_map(self):
        for id in range(12):
            instance_path = os.path.join(self.dataset_dir, f"House_{(id+1):02d}")
            images_dir=os.path.join(instance_path,"images")

            egomap_dir=os.path.join(instance_path,"ego_map")
            mkdir_if_not_exist(egomap_dir)
            for i in range(self.House_list[id]):
                depth_path_0=os.path.join(images_dir,str(i)+"_depth.png")
                depth_path_1=os.path.join(images_dir,str(i)+"_depth_1.png")
                depth_path_2=os.path.join(images_dir,str(i)+"_depth_2.png")
                depth_path_3=os.path.join(images_dir,str(i)+"_depth_3.png")

                depth_0=cv2.imread(depth_path_0,cv2.IMREAD_COLOR)
                depth_1=cv2.imread(depth_path_1,cv2.IMREAD_COLOR)
                depth_2=cv2.imread(depth_path_2,cv2.IMREAD_COLOR)
                depth_3=cv2.imread(depth_path_3,cv2.IMREAD_COLOR)

                depth_0 = depth_0[:,:,0] / 255.0 * 10 - 0.1
                depth_1 = depth_1[:,:,0] / 255.0 * 10 - 0.1
                depth_2 = depth_2[:,:,0] / 255.0 * 10 - 0.1
                depth_3 = depth_3[:,:,0] / 255.0 * 10 - 0.1

                depth_0=torch.from_numpy(depth_0).unsqueeze(0).unsqueeze(0)
                depth_1=torch.from_numpy(depth_1).unsqueeze(0).unsqueeze(0)
                depth_2=torch.from_numpy(depth_2).unsqueeze(0).unsqueeze(0)
                depth_3=torch.from_numpy(depth_3).unsqueeze(0).unsqueeze(0)

                depth  =torch.cat([depth_0,depth_1,depth_2,depth_3],dim=0)

                ego_map=compute_egomap(depth,fov=90,scale=self.scale,map_size=self.map_size,min_h=95,max_h=97)
                ego_map=ego_map[(self.map_size-self.crop_map_size)//2:(self.map_size-self.crop_map_size)//2+self.crop_map_size,(self.map_size-self.crop_map_size)//2:(self.map_size-self.crop_map_size)//2+self.crop_map_size]
                # cv2.imwrite("ego_map.png",ego_map*255.)
                np.save(os.path.join(egomap_dir,str(i)+".npy"),ego_map)
    # def gen_line(self,line):#点的坐标必须都是整数   
    #     point_A=line[0]
    #     point_B=line[1]
    #     assert point_A[0]%1==0
    #     assert point_A[1]%1==0
    #     assert point_B[0]%1==0
    #     assert point_B[1]%1==0
    #     point_A=point_A.astype(int)
    #     point_B=point_B.astype(int)

    #     d_x=point_A[0]-point_B[0]
    #     d_y=point_A[1]-point_B[1]

    #     line_points=[]
    #     if abs(d_x)<abs(d_y):
    #         for i in range(min(point_A[1],point_B[1]),max(point_A[1],point_B[1])+1):
    #             point=np.zeros(2)
    #             point[1]=i
    #             point[0]=((i-point_A[1])/d_y*d_x+point_A[0])
    #             line_points.append(point.astype(int))
    #     else:
    #         for i in range(min(point_A[0],point_B[0]),max(point_A[0],point_B[0])+1):
    #             point=np.zeros(2)
    #             point[0]=i
    #             point[1]=((i-point_A[0])/d_x*d_y+point_A[1])
    #             line_points.append(point.astype(int))
    #     return line_points


    # def get_passable_space(self,House_id):
        
    #     House_id += 1
    #     instance_path = os.path.join(self.dataset_dir, f"House_{House_id:02d}")
    #     room_lines_path = os.path.join(instance_path, "room_lines.npy")
    #     door_lines_path = os.path.join(instance_path, "door_lines.npy")
    #     window_lines_path = os.path.join(instance_path, "window_lines.npy")



    #     room_lines=np.load(room_lines_path).astype(np.float64)
    #     door_lines=np.load(door_lines_path).astype(np.float64)
    #     window_lines=np.load(window_lines_path).astype(np.float64)

    #     doors_num=door_lines.shape[0]

    #     x_max=np.max(room_lines[:,0,0]).astype(int)
    #     y_max=np.max(room_lines[:,0,1]).astype(int)
    #     border=50

    #     space_mat=np.zeros((y_max+border,x_max+border))
    #     for line in room_lines:
    #         line_points=self.gen_line(line)
    #         for point in line_points:
    #             space_mat[point[1],point[0]]=1

    #     for i in range(doors_num):
    #         for j in range(i+1,doors_num):
    #             door_A=door_lines[i]
    #             door_B=door_lines[j]

    #             door_A_v=door_A[0]-door_A[1]
    #             door_B_v=door_B[0]-door_B[1]
    #             cos_AB=np.dot(door_A_v,door_B_v)/(np.linalg.norm(door_A_v)*np.linalg.norm(door_B_v))
    #             if(1-cos_AB<=0):
    #                 #door_A,door_B平行
    #                 distance=np.cross((door_A[0]-door_B[0]),door_A_v/np.linalg.norm(door_A_v))
    #                 if abs(distance)<40:
    #                     #是一对相对而开的门   

    #                     if np.linalg.norm(door_A[0]-door_B[0])<np.linalg.norm(door_A[0]-door_B[1]):
    #                         connect_line_1=np.stack([door_A[0],door_B[0]])  
    #                     else:
    #                         connect_line_1=np.stack([door_A[0],door_B[1]])
    #                     if np.linalg.norm(door_A[1]-door_B[0])<np.linalg.norm(door_A[1]-door_B[1]):
    #                         connect_line_2=np.stack([door_A[1],door_B[0]]) 
    #                     else:
    #                         connect_line_2=np.stack([door_A[1],door_B[1]])
    #                     if np.linalg.norm(connect_line_1[0]-connect_line_1[1])>40 or np.linalg.norm(connect_line_2[0]-connect_line_2[1])>40:
    #                         continue
    #                     line_points=self.gen_line(door_A)
    #                     for point in line_points:
    #                         space_mat[point[1],point[0]]=0
    #                     line_points=self.gen_line(door_B)
    #                     for point in line_points:
    #                         space_mat[point[1],point[0]]=0#将两个门的位置设为freespace
                            
    #                     line_points=self.gen_line(connect_line_1)
    #                     for point in line_points:
    #                         space_mat[point[1],point[0]]=1
    #                     line_points=self.gen_line(connect_line_2)
    #                     for point in line_points:
    #                         space_mat[point[1],point[0]]=1#将两个门之间连接的位置设为obsspace

    #     cv2.imwrite("passable.jpg",space_mat*255.)
    #     return space_mat

    # def meter2pixel(self,position,House_id):
    #     House_id += 1
    #     instance_path = os.path.join(self.dataset_dir, f"House_{House_id:02d}")

    #     txt_path=os.path.join(instance_path,"pixel_xy_robot_xy.txt")
    #                                          #          0,1,          2,3            4,5           6,7
    #     pixel_robot_info=np.loadtxt(txt_path)#左下pixel(x,y),左下robot(x,y),右上pixel(x,y),右上robot(x,y)

    #     pixel_A=pixel_robot_info[0:2]
    #     pixel_B=pixel_robot_info[4:6]
    #     robot_A=pixel_robot_info[2:4]
    #     robot_B=pixel_robot_info[6:8]

    #     out_pos=np.zeros((2))
    #     out_pos[0]=( position[0]-robot_A[0])*(pixel_B[0]-pixel_A[0])/(robot_B[0]-robot_A[0])+pixel_A[0]
    #     out_pos[1]=(-position[1]-robot_A[1])*(pixel_B[1]-pixel_A[1])/(robot_B[1]-robot_A[1])+pixel_A[1]
    #     return out_pos.astype(np.int64)

    # def pixel2meter(self,position,House_id):
    #     House_id += 1
    #     instance_path = os.path.join(self.dataset_dir, f"House_{House_id:02d}")

    #     txt_path=os.path.join(instance_path,"pixel_xy_robot_xy.txt")
    #                                          #          0,1,          2,3            4,5           6,7
    #     pixel_robot_info=np.loadtxt(txt_path)#左下pixel(x,y),左下robot(x,y),右上pixel(x,y),右上robot(x,y)

    #     pixel_A=pixel_robot_info[0:2]
    #     pixel_B=pixel_robot_info[4:6]
    #     robot_A=pixel_robot_info[2:4]
    #     robot_B=pixel_robot_info[6:8]

    #     out_pos=np.zeros((2))
    #     out_pos[0]=  (position[0]-pixel_A[0])/(pixel_B[0]-pixel_A[0])*(robot_B[0]-robot_A[0])+robot_A[0]
    #     out_pos[1]= -((position[1]-pixel_A[1])/(pixel_B[1]-pixel_A[1])*(robot_B[1]-robot_A[1])+robot_A[1])
    #     return out_pos

