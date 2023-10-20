import numpy as np
import os
import cv2

from dataset.unity_dataset import unityDataset
from RRT_star.RRT_star import *


log_path="/home/swt/intention_dataset/Home01/ManualAgent5/LogImg.csv"
cols=[1,3]
data = np.loadtxt(open(log_path,"rb"),delimiter=",",skiprows=1,usecols=cols) 
trajactory=[]
house_id=7
eval_dataset=unityDataset(dataset_dir="unitydataset",is_training=True,n_sample_points=2048,testing_set=[7])

map=eval_dataset.get_passable_space(house_id)
map_extend=np.zeros_like(map)
border_len=7
for i in range(map.shape[0]):
    for j in range(map.shape[1]):
        if map[i,j]==1:
            map_extend[i-border_len:i+border_len+1,j-border_len:j+border_len+1]=1
map=map_extend.copy()




for i in range(data.shape[0]):
    pos=np.zeros(2)
    pos[0] = data[i,0]
    pos[1] =-data[i,1]
    pos = eval_dataset.meter2pixel(pos,house_id)#   data[i,0]/scale
    # pos[1] =-data[i,1]/scale

    trajactory.append(pos.astype(np.int32))

trajactory=np.stack(trajactory,axis=0)

for i in range(trajactory.shape[0]):
    map[trajactory[i,1],trajactory[i,0]]=1


cv2.imshow("trajactory",map*255.)
cv2.waitKey(0)


map = Map(map_extend)

start =trajactory[0].tolist()
end = trajactory[-1].tolist()


a = RRTStar(map, 5, 0)
t0=time.time()
a.Path(start, end,"global_map",5)
print(time.time()-t0)
cv2.waitKey(0)