import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import json

from .zind_utils import *
from .s3d_utils import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

class unityDataset(Dataset):



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
        testing_set=[7,11]
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

        self.House_list=[]
        for i in range(1,13):
           img_path = os.path.join(self.dataset_dir, f"House_{i:02d}","images")
           images=os.listdir(img_path)
           self.House_list.append((len(images)-1)//4)

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
        # else:
        #     House_id=1
            
        instance_path = os.path.join(self.dataset_dir, f"House_{House_id:02d}")
        room_lines_path = os.path.join(instance_path, "room_lines.npy")
        door_lines_path = os.path.join(instance_path, "door_lines.npy")
        window_lines_path = os.path.join(instance_path, "window_lines.npy")

        txt_path=os.path.join(instance_path,"pixel_xy_robot_xy.txt")

        global_rot = np.random.rand() * 360 if self.is_training else 0#整个地图的随机旋转，同时坐标也会旋转

# ----->x
# |
# |
# V
# y
#角度顺时针为正

                                             #          0,1,          2,3            4,5           6,7
        pixel_robot_info=np.loadtxt(txt_path)#左下pixel(x,y),左下robot(x,y),右上pixel(x,y),右上robot(x,y)

        pixel_A=pixel_robot_info[0:2]
        pixel_B=pixel_robot_info[4:6]
        robot_A=pixel_robot_info[2:4]
        robot_B=pixel_robot_info[6:8]

        room_lines=np.load(room_lines_path).astype(np.float64)
        door_lines=np.load(door_lines_path).astype(np.float64)
        window_lines=np.load(window_lines_path).astype(np.float64)

#把像素坐标投影到robot中的以m为单位的坐标
        room_lines[:,:,0]=(room_lines[:,:,0]-pixel_A[0])/(pixel_B[0]-pixel_A[0])*(robot_B[0]-robot_A[0])+robot_A[0]
        room_lines[:,:,1]=(room_lines[:,:,1]-pixel_A[1])/(pixel_B[1]-pixel_A[1])*(robot_B[1]-robot_A[1])+robot_A[1]
        door_lines[:,:,0]=(door_lines[:,:,0]-pixel_A[0])/(pixel_B[0]-pixel_A[0])*(robot_B[0]-robot_A[0])+robot_A[0]
        door_lines[:,:,1]=(door_lines[:,:,1]-pixel_A[1])/(pixel_B[1]-pixel_A[1])*(robot_B[1]-robot_A[1])+robot_A[1]     
        window_lines[:,:,0]=(window_lines[:,:,0]-pixel_A[0])/(pixel_B[0]-pixel_A[0])*(robot_B[0]-robot_A[0])+robot_A[0]
        window_lines[:,:,1]=(window_lines[:,:,1]-pixel_A[1])/(pixel_B[1]-pixel_A[1])*(robot_B[1]-robot_A[1])+robot_A[1]


#由于像素坐标的y轴向下为正方向，与robot系相反，而我们希望使用的y轴正方向指向下方，因此全部取负号

        room_lines[:,:,1]=-room_lines[:,:,1]
        door_lines[:,:,1]=-door_lines[:,:,1]
        window_lines[:,:,1]=-window_lines[:,:,1]


        room_lines = rot_verts(room_lines, global_rot)                                        
        door_lines = rot_verts(door_lines, global_rot)
        window_lines = rot_verts(window_lines, global_rot)


        if self.n_sample_points is not None:#设定共采多少个point
            perimeter = np.linalg.norm(
                room_lines[:, 0, :] - room_lines[:, 1, :], axis=-1
            ).sum()
            bases, bases_normal = sample_points_from_lines(
                room_lines, 0.9 * perimeter / self.n_sample_points
            )
            rnd_sample_idx = np.random.permutation(bases.shape[0])[
                : self.n_sample_points
            ]
            bases = bases[rnd_sample_idx]
            bases_normal = bases_normal[rnd_sample_idx]
        else:#设定采样point的间隔
            bases, bases_normal = sample_points_from_lines(
                room_lines, self.line_sampling_interval
            )

        bases_door_mask = points_on_lines(bases, door_lines)
        bases_window_mask = points_on_lines(bases, window_lines)
        bases_feat = np.concatenate(
            [
                (bases - bases.mean(axis=0, keepdims=True))
                / 5.0,  # 2 ,normalize with 5.0
                np.zeros_like(bases[:, 0:1]),  # 1
                bases_normal,  # 2
                bases_door_mask.reshape(-1, 1),  # 1
                bases_window_mask.reshape(-1, 1),
            ],  # 1
            axis=1,
        )  # N,D
        images_dir=os.path.join(instance_path,"images")

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

        # image=np.concatenate([image_0[:,width//2:,:],image_1,image_2,image_3,image_0[:,:width//2,:],],axis=1)
        # image = cv2.flip(image, 1)#应该是因为地图坐标轴方向的原因，再看看??????

        Log_path=os.path.join(images_dir,"LogImg.csv")
        Log_data = pd.read_csv(Log_path,sep=',',header='infer',usecols=[1,3,5])
        x_loc=Log_data.values[idx,0]
        y_loc=Log_data.values[idx,1]
        rot=Log_data.values[idx,2]#顺时针

#对robot坐标中的y坐标取负号，使其y轴指向下方
        gt_loc=np.array([x_loc,-y_loc])   

        gt_loc = rot_verts(gt_loc, global_rot)

        gt_rot = global_rot+rot###############*******************

        rnd_rot = np.random.rand() * 360#仅在坐标点旋转，对该点的图像和坐标进行旋转
        image = rot_pano(image, rnd_rot)#
        gt_rot = gt_rot + rnd_rot
        gt_rot=np.array([gt_rot])
####写获取坐标点的代码————position、rotatio，看一下方向！

        # query_image = cv2.resize(
        #     pano_image, (self.image_size * 2, self.image_size)
        # )
        query_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        query_image -= (0.485, 0.456, 0.406)
        query_image /= (0.229, 0.224, 0.225)
        query_image=np.transpose(query_image, (2, 0, 1))

        input = {
            "bases": bases.astype(np.float32),  # B,2
            "bases_normal": bases_normal.astype(np.float32),  # B,2
            "bases_feat": bases_feat.astype(np.float32),
            "gt_loc": gt_loc.astype(np.float32),  #
            "gt_rot": (gt_rot % 360 / 180 * np.pi).astype(np.float32),  #
            "query_image": query_image.astype(np.float32),
            "gt_fov":np.array([360]).astype(np.float32),
        }  # B,D

        return input


    def gen_line(self,line):#点的坐标必须都是整数   
        point_A=line[0]
        point_B=line[1]
        assert point_A[0]%1==0
        assert point_A[1]%1==0
        assert point_B[0]%1==0
        assert point_B[1]%1==0
        point_A=point_A.astype(int)
        point_B=point_B.astype(int)

        d_x=point_A[0]-point_B[0]
        d_y=point_A[1]-point_B[1]

        line_points=[]
        if abs(d_x)<abs(d_y):
            for i in range(min(point_A[1],point_B[1]),max(point_A[1],point_B[1])+1):
                point=np.zeros(2)
                point[1]=i
                point[0]=((i-point_A[1])/d_y*d_x+point_A[0])
                line_points.append(point.astype(int))
        else:
            for i in range(min(point_A[0],point_B[0]),max(point_A[0],point_B[0])+1):
                point=np.zeros(2)
                point[0]=i
                point[1]=((i-point_A[0])/d_x*d_y+point_A[1])
                line_points.append(point.astype(int))
        return line_points

    def get_map(self,House_id):
        House_id+=1
    
        instance_path = os.path.join(self.dataset_dir, f"House_{House_id:02d}")
        room_lines_path = os.path.join(instance_path, "room_lines.npy")
        door_lines_path = os.path.join(instance_path, "door_lines.npy")
        window_lines_path = os.path.join(instance_path, "window_lines.npy")

        txt_path=os.path.join(instance_path,"pixel_xy_robot_xy.txt")


# ----->x
# |
# |
# V
# y
#角度顺时针为正

                                             #          0,1,          2,3            4,5           6,7
        pixel_robot_info=np.loadtxt(txt_path)#左下pixel(x,y),左下robot(x,y),右上pixel(x,y),右上robot(x,y)

        pixel_A=pixel_robot_info[0:2]
        pixel_B=pixel_robot_info[4:6]
        robot_A=pixel_robot_info[2:4]
        robot_B=pixel_robot_info[6:8]

        room_lines=np.load(room_lines_path).astype(np.float64)
        door_lines=np.load(door_lines_path).astype(np.float64)
        window_lines=np.load(window_lines_path).astype(np.float64)

#把像素坐标投影到robot中的以m为单位的坐标
        room_lines[:,:,0]=(room_lines[:,:,0]-pixel_A[0])/(pixel_B[0]-pixel_A[0])*(robot_B[0]-robot_A[0])+robot_A[0]
        room_lines[:,:,1]=(room_lines[:,:,1]-pixel_A[1])/(pixel_B[1]-pixel_A[1])*(robot_B[1]-robot_A[1])+robot_A[1]
        door_lines[:,:,0]=(door_lines[:,:,0]-pixel_A[0])/(pixel_B[0]-pixel_A[0])*(robot_B[0]-robot_A[0])+robot_A[0]
        door_lines[:,:,1]=(door_lines[:,:,1]-pixel_A[1])/(pixel_B[1]-pixel_A[1])*(robot_B[1]-robot_A[1])+robot_A[1]     
        window_lines[:,:,0]=(window_lines[:,:,0]-pixel_A[0])/(pixel_B[0]-pixel_A[0])*(robot_B[0]-robot_A[0])+robot_A[0]
        window_lines[:,:,1]=(window_lines[:,:,1]-pixel_A[1])/(pixel_B[1]-pixel_A[1])*(robot_B[1]-robot_A[1])+robot_A[1]


#由于像素坐标的y轴向下为正方向，与robot系相反，而我们希望使用的y轴正方向指向下方，因此全部取负号

        room_lines[:,:,1]=-room_lines[:,:,1]
        door_lines[:,:,1]=-door_lines[:,:,1]
        window_lines[:,:,1]=-window_lines[:,:,1]

        return room_lines,door_lines,window_lines



    def get_passable_space(self,House_id):
        
        House_id += 1
        instance_path = os.path.join(self.dataset_dir, f"House_{House_id:02d}")
        room_lines_path = os.path.join(instance_path, "room_lines.npy")
        door_lines_path = os.path.join(instance_path, "door_lines.npy")
        window_lines_path = os.path.join(instance_path, "window_lines.npy")



        room_lines=np.load(room_lines_path).astype(np.float64)
        door_lines=np.load(door_lines_path).astype(np.float64)
        window_lines=np.load(window_lines_path).astype(np.float64)

        doors_num=door_lines.shape[0]

        x_max=np.max(room_lines[:,0,0]).astype(int)
        y_max=np.max(room_lines[:,0,1]).astype(int)
        border=50

        space_mat=np.zeros((y_max+border,x_max+border))
        for line in room_lines:
            line_points=self.gen_line(line)
            for point in line_points:
                space_mat[point[1],point[0]]=1

        for i in range(doors_num):
            for j in range(i+1,doors_num):
                door_A=door_lines[i]
                door_B=door_lines[j]

                door_A_v=door_A[0]-door_A[1]
                door_B_v=door_B[0]-door_B[1]
                cos_AB=np.dot(door_A_v,door_B_v)/(np.linalg.norm(door_A_v)*np.linalg.norm(door_B_v))
                if(1-cos_AB<=0):
                    #door_A,door_B平行
                    distance=np.cross((door_A[0]-door_B[0]),door_A_v/np.linalg.norm(door_A_v))
                    if abs(distance)<40:
                        #是一对相对而开的门   

                        if np.linalg.norm(door_A[0]-door_B[0])<np.linalg.norm(door_A[0]-door_B[1]):
                            connect_line_1=np.stack([door_A[0],door_B[0]])  
                        else:
                            connect_line_1=np.stack([door_A[0],door_B[1]])
                        if np.linalg.norm(door_A[1]-door_B[0])<np.linalg.norm(door_A[1]-door_B[1]):
                            connect_line_2=np.stack([door_A[1],door_B[0]]) 
                        else:
                            connect_line_2=np.stack([door_A[1],door_B[1]])
                        if np.linalg.norm(connect_line_1[0]-connect_line_1[1])>40 or np.linalg.norm(connect_line_2[0]-connect_line_2[1])>40:
                            continue
                        line_points=self.gen_line(door_A)
                        for point in line_points:
                            space_mat[point[1],point[0]]=0
                        line_points=self.gen_line(door_B)
                        for point in line_points:
                            space_mat[point[1],point[0]]=0#将两个门的位置设为freespace
                            
                        line_points=self.gen_line(connect_line_1)
                        for point in line_points:
                            space_mat[point[1],point[0]]=1
                        line_points=self.gen_line(connect_line_2)
                        for point in line_points:
                            space_mat[point[1],point[0]]=1#将两个门之间连接的位置设为obsspace

        cv2.imwrite("passable.jpg",space_mat*255.)
        return space_mat

    def meter2pixel(self,position,House_id):
        House_id += 1
        instance_path = os.path.join(self.dataset_dir, f"House_{House_id:02d}")

        txt_path=os.path.join(instance_path,"pixel_xy_robot_xy.txt")
                                             #          0,1,          2,3            4,5           6,7
        pixel_robot_info=np.loadtxt(txt_path)#左下pixel(x,y),左下robot(x,y),右上pixel(x,y),右上robot(x,y)

        pixel_A=pixel_robot_info[0:2]
        pixel_B=pixel_robot_info[4:6]
        robot_A=pixel_robot_info[2:4]
        robot_B=pixel_robot_info[6:8]

        out_pos=np.zeros((2))
        out_pos[0]=( position[0]-robot_A[0])*(pixel_B[0]-pixel_A[0])/(robot_B[0]-robot_A[0])+pixel_A[0]
        out_pos[1]=(-position[1]-robot_A[1])*(pixel_B[1]-pixel_A[1])/(robot_B[1]-robot_A[1])+pixel_A[1]
        return out_pos.astype(np.int64)

    def pixel2meter(self,position,House_id):
        House_id += 1
        instance_path = os.path.join(self.dataset_dir, f"House_{House_id:02d}")

        txt_path=os.path.join(instance_path,"pixel_xy_robot_xy.txt")
                                             #          0,1,          2,3            4,5           6,7
        pixel_robot_info=np.loadtxt(txt_path)#左下pixel(x,y),左下robot(x,y),右上pixel(x,y),右上robot(x,y)

        pixel_A=pixel_robot_info[0:2]
        pixel_B=pixel_robot_info[4:6]
        robot_A=pixel_robot_info[2:4]
        robot_B=pixel_robot_info[6:8]

        out_pos=np.zeros((2))
        out_pos[0]=  (position[0]-pixel_A[0])/(pixel_B[0]-pixel_A[0])*(robot_B[0]-robot_A[0])+robot_A[0]
        out_pos[1]= -((position[1]-pixel_A[1])/(pixel_B[1]-pixel_A[1])*(robot_B[1]-robot_A[1])+robot_A[1])
        return out_pos
# #把像素坐标投影到robot中的以m为单位的坐标
#         room_lines[:,:,0]=(room_lines[:,:,0]-pixel_A[0])/(pixel_B[0]-pixel_A[0])*(robot_B[0]-robot_A[0])+robot_A[0]
#         room_lines[:,:,1]=(room_lines[:,:,1]-pixel_A[1])/(pixel_B[1]-pixel_A[1])*(robot_B[1]-robot_A[1])+robot_A[1]
#         door_lines[:,:,0]=(door_lines[:,:,0]-pixel_A[0])/(pixel_B[0]-pixel_A[0])*(robot_B[0]-robot_A[0])+robot_A[0]
#         door_lines[:,:,1]=(door_lines[:,:,1]-pixel_A[1])/(pixel_B[1]-pixel_A[1])*(robot_B[1]-robot_A[1])+robot_A[1]     
#         window_lines[:,:,0]=(window_lines[:,:,0]-pixel_A[0])/(pixel_B[0]-pixel_A[0])*(robot_B[0]-robot_A[0])+robot_A[0]
#         window_lines[:,:,1]=(window_lines[:,:,1]-pixel_A[1])/(pixel_B[1]-pixel_A[1])*(robot_B[1]-robot_A[1])+robot_A[1]

# #由于像素坐标的y轴向下为正方向，与robot系相反，而我们希望使用的y轴正方向指向下方，因此全部取负号

#         room_lines[:,:,1]=-room_lines[:,:,1]
#         door_lines[:,:,1]=-door_lines[:,:,1]
#         window_lines[:,:,1]=-window_lines[:,:,1]

