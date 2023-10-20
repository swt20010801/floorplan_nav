import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import json

from .zind_utils import *
from .s3d_utils import *
import pandas as pd


class unityILDataset(Dataset):

    def __init__(
        self,
        dataset_dir="/home/swt/IL_dataset/Home01/",
        image_size=256,
        is_training=False,
        training_set=[0,1,2,3,4,5,6,8,9,10],
        testing_set=[7,11],
        interval_step=4
    ):

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.is_training = is_training
        self.Houseid_find=[]
        self.Imageid_find=[]
        self.training_set = training_set    ########
        self.testing_set =  testing_set                ########
        self.N=0
        self.interval_step=interval_step

        self.House_list=[]
        # self.idx=[]

        for i in range(1,13):
           img_path = os.path.join(self.dataset_dir, f"House_{i:02d}")
           images=os.listdir(img_path)
           self.House_list.append((len(images)-1)-1000)
        #    self.idx.append(0)

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
        print(
            f"{self.N} samples loaded from {type(self)} dataset. (is_training={is_training})"
        )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):


        House_id=self.Houseid_find[idx]+1
        idx=self.Imageid_find[idx]

        instance_path = os.path.join(self.dataset_dir, f"House_{House_id:02d}")
        
        images_dir=instance_path

        image_path=os.path.join(images_dir,str(idx)+"_rgb.jpg")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        assert image.shape[0]==image.shape[1]

        image=cv2.resize(image,(self.image_size,self.image_size))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0        
        image -= (0.485, 0.456, 0.406)
        image /= (0.229, 0.224, 0.225)
        image=np.transpose(image, (2, 0, 1))

        Log_path=os.path.join(images_dir,"LogImg.csv")
        Log_data = pd.read_csv(Log_path,sep=',',header='infer',usecols=[1,2,3])
        distance2goal=Log_data.values[idx,0]
        angle2goal   =Log_data.values[idx,1]
        action       =Log_data.values[idx,2]



        distance2goal=np.array([distance2goal])
        angle2goal   =np.array([angle2goal])
        action       =np.array([action])


        input = {
            "image": image.astype(np.float32),  
            "distance2goal": distance2goal.astype(np.float32), 
            "angle2goal": angle2goal.astype(np.float32),
            "action": action.astype(np.int32),  
        }  

        return input
