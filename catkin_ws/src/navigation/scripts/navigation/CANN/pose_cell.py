import numpy as np
import matplotlib.pyplot as plt
import time
def create_gc_weights(xDim, yDim, xVar, yVar):

    xDimCentre = np.floor((xDim-1) / 2)
    yDimCentre = np.floor((yDim-1) / 2)

    weight = np.zeros((xDim, yDim))
    
    for x in range(xDim):
        for y in range(yDim):
            weight[x,y] = 1/(xVar*np.sqrt(2*np.pi))*np.exp((-(x - xDimCentre) ** 2) / (2 * xVar ** 2)) * 1/(yVar*np.sqrt(2*np.pi))*np.exp((-(y - yDimCentre) ** 2) / (2 * yVar ** 2))  


    total = (sum(sum(weight)))
    weight = weight/total      

    return weight




class pose_cell():
    def __init__(
        self,
        GC_X_DIM=64,
        GC_Y_DIM=64,
        GC_EXCIT_X_DIM=9,
        GC_EXCIT_Y_DIM=9,
        GC_INHIB_X_DIM=5,
        GC_INHIB_Y_DIM=5,
        GC_GLOBAL_INHIB=0.005,   
        GC_VT_INJECT_ENERGY=0.2,
        # GC_EXCIT_X_WRAP,
        # GC_EXCIT_Y_WRAP,
        # GC_INHIB_X_WRAP,
        # GC_INHIB_Y_WRAP,
        # GC_EXCIT_WEIGHT,
        # GC_INHIB_WEIGHT,
        GC_EXCIT_X_VAR=1.5,
        GC_EXCIT_Y_VAR=1.5,
        
        GC_INHIB_X_VAR=2,
        GC_INHIB_Y_VAR=2,
        GC_PACKET_SIZE=4,
        init_activation_loc=None
        ):
        self.GC_VT_INJECT_ENERGY=GC_VT_INJECT_ENERGY
        self.GC_X_DIM=GC_X_DIM
        self.GC_Y_DIM=GC_Y_DIM
        self.GC_EXCIT_X_DIM=GC_EXCIT_X_DIM
        self.GC_EXCIT_Y_DIM=GC_EXCIT_Y_DIM
        self.GC_INHIB_X_DIM=GC_INHIB_X_DIM
        self.GC_INHIB_Y_DIM=GC_INHIB_Y_DIM
        self.GC_GLOBAL_INHIB=GC_GLOBAL_INHIB
        self.GC_EXCIT_WEIGHT = create_gc_weights(GC_EXCIT_X_DIM,  GC_EXCIT_Y_DIM, GC_EXCIT_X_VAR, GC_EXCIT_Y_VAR)
        self.GC_INHIB_WEIGHT = create_gc_weights(GC_INHIB_X_DIM,  GC_INHIB_Y_DIM, GC_INHIB_X_VAR, GC_INHIB_X_VAR)

        self.GC_PACKET_SIZE=GC_PACKET_SIZE

        GC_EXCIT_X_DIM_HALF = np.floor(GC_EXCIT_X_DIM / 2)
        GC_EXCIT_Y_DIM_HALF = np.floor(GC_EXCIT_Y_DIM / 2)
        
        GC_INHIB_X_DIM_HALF = np.floor(GC_INHIB_X_DIM / 2)
        GC_INHIB_Y_DIM_HALF = np.floor(GC_INHIB_Y_DIM / 2)
        
        self.GC_EXCIT_X_WRAP = np.concatenate((np.arange(GC_X_DIM - GC_EXCIT_X_DIM_HALF,GC_X_DIM),np.arange(GC_X_DIM),np.arange(GC_EXCIT_X_DIM_HALF)))#[(GC_X_DIM - GC_EXCIT_X_DIM_HALF + 1) : GC_X_DIM  1 : GC_X_DIM  1 : GC_EXCIT_X_DIM_HALF]
        self.GC_EXCIT_Y_WRAP = np.concatenate((np.arange(GC_Y_DIM - GC_EXCIT_Y_DIM_HALF,GC_Y_DIM),np.arange(GC_Y_DIM),np.arange(GC_EXCIT_Y_DIM_HALF)))#[(GC_X_DIM - GC_EXCIT_X_DIM_HALF + 1) : GC_X_DIM  1 : GC_X_DIM  1 : GC_EXCIT_X_DIM_HALF]

        self.GC_INHIB_X_WRAP = np.concatenate((np.arange(GC_X_DIM - GC_INHIB_X_DIM_HALF,GC_X_DIM),np.arange(GC_X_DIM),np.arange(GC_INHIB_X_DIM_HALF)))#[(GC_X_DIM - GC_EXCIT_X_DIM_HALF + 1) : GC_X_DIM  1 : GC_X_DIM  1 : GC_EXCIT_X_DIM_HALF]
        self.GC_INHIB_Y_WRAP = np.concatenate((np.arange(GC_Y_DIM - GC_INHIB_Y_DIM_HALF,GC_Y_DIM),np.arange(GC_Y_DIM),np.arange(GC_INHIB_Y_DIM_HALF)))#[(GC_X_DIM - GC_EXCIT_X_DIM_HALF + 1) : GC_X_DIM  1 : GC_X_DIM  1 : GC_EXCIT_X_DIM_HALF]

        self.GC_EXCIT_X_WRAP=self.GC_EXCIT_X_WRAP.astype(np.int32)
        self.GC_EXCIT_Y_WRAP=self.GC_EXCIT_Y_WRAP.astype(np.int32)
        self.GC_INHIB_X_WRAP=self.GC_INHIB_X_WRAP.astype(np.int32)
        self.GC_INHIB_Y_WRAP=self.GC_INHIB_Y_WRAP.astype(np.int32)

        self.GC_X_TH_SIZE = 2*np.pi / GC_X_DIM
        self.GC_Y_TH_SIZE = 2*np.pi / GC_Y_DIM
  
        self.GC_X_SUM_SIN_LOOKUP = np.sin(np.arange(GC_X_DIM) * self.GC_X_TH_SIZE)
        self.GC_X_SUM_COS_LOOKUP = np.cos(np.arange(GC_X_DIM) * self.GC_X_TH_SIZE)
    
        self.GC_Y_SUM_SIN_LOOKUP = np.sin(np.arange(GC_Y_DIM) * self.GC_Y_TH_SIZE)
        self.GC_Y_SUM_COS_LOOKUP = np.cos(np.arange(GC_Y_DIM) * self.GC_Y_TH_SIZE)
    
        # self.GC_MAX_X_WRAP = np.concatenate((np.arange(GC_X_DIM - GC_PACKET_SIZE , GC_X_DIM), np.arange(GC_X_DIM),np.arange(GC_PACKET_SIZE)))
        # self.GC_MAX_Y_WRAP = np.concatenate((np.arange(GC_Y_DIM - GC_PACKET_SIZE , GC_Y_DIM), np.arange(GC_Y_DIM),np.arange(GC_PACKET_SIZE)))

        self.GC_MAX_X_WRAP = np.concatenate((np.arange(GC_X_DIM - GC_PACKET_SIZE , GC_X_DIM), np.arange(GC_X_DIM),np.arange(GC_PACKET_SIZE)))
        self.GC_MAX_Y_WRAP = np.concatenate((np.arange(GC_Y_DIM - GC_PACKET_SIZE , GC_Y_DIM), np.arange(GC_Y_DIM),np.arange(GC_PACKET_SIZE)))

        self.GRIDCELLS = np.zeros((GC_X_DIM, GC_Y_DIM))
        if init_activation_loc != None:
            # print(init_activation_loc)
            for loc in init_activation_loc:
                self.GRIDCELLS[loc[0], loc[1]] = 1

        else:
            gcX, gcY = self.get_gc_initial_pos()
            self.GRIDCELLS[int(gcX), int(gcY)] = 1
        
        self.activation()

        # plt.imshow(self.GRIDCELLS, cmap='viridis', interpolation='nearest')
        # plt.savefig('before.png')

    def get_gc_initial_pos(self):
        gcX = np.floor(self.GC_X_DIM / 2)
        gcY = np.floor(self.GC_Y_DIM / 2)
        return gcX , gcY
    
    def move(self,curYawThetaInRadian,transV):
        # motion

        if curYawThetaInRadian == 0:
            GRIDCELLS_shift=np.roll(self.GRIDCELLS, shift=1,axis=1)
            # GRIDCELLS_shift[:,0]=0
            self.GRIDCELLS = self.GRIDCELLS*(1.0 - transV)+GRIDCELLS_shift* transV
        elif curYawThetaInRadian == np.pi/2:
            GRIDCELLS_shift=np.roll(self.GRIDCELLS, shift=1,axis=0)
            # GRIDCELLS_shift[0,:]=0
            self.GRIDCELLS = self.GRIDCELLS*(1.0 - transV)+GRIDCELLS_shift* transV
        elif curYawThetaInRadian == np.pi:
            GRIDCELLS_shift=np.roll(self.GRIDCELLS, shift=-1,axis=1)
            # GRIDCELLS_shift[:,-1]=0
            self.GRIDCELLS = self.GRIDCELLS*(1.0 - transV)+GRIDCELLS_shift* transV
        elif curYawThetaInRadian == 3*np.pi/2:
            GRIDCELLS_shift=np.roll(self.GRIDCELLS, shift=-1,axis=0)
            # GRIDCELLS_shift[-1,:]=0
            self.GRIDCELLS = self.GRIDCELLS*(1.0 - transV)+GRIDCELLS_shift* transV   
        else:

            gcInZPlane90 = np.rot90(self.GRIDCELLS, np.floor(curYawThetaInRadian *2/np.pi))

            dir90 = curYawThetaInRadian - np.floor(curYawThetaInRadian *2/np.pi)* np.pi/2

            gcInZPlaneNew = np.zeros((gcInZPlane90.shape[0] + 2, gcInZPlane90.shape[1] + 2))            
            gcInZPlaneNew[1:-1,1:-1] = gcInZPlane90

            weight_sw = transV**2 * np.cos(dir90) * np.sin(dir90)
            weight_se = transV * np.sin(dir90) - transV**2 * np.cos(dir90) * np.sin(dir90)
            weight_nw = transV * np.cos(dir90) - transV**2 * np.cos(dir90) * np.sin(dir90)
            weight_ne = 1.0 - weight_sw - weight_se - weight_nw

            GRIDCELLS_nw=np.roll(gcInZPlaneNew, shift=1,axis=1)
            # GRIDCELLS_nw[:,0]=0
            GRIDCELLS_se=np.roll(gcInZPlaneNew, shift=1,axis=0)
            # GRIDCELLS_se[0,:]=0
            GRIDCELLS_sw=np.roll(np.roll(gcInZPlaneNew, shift=1,axis=1),shift=1,axis=0)
            # GRIDCELLS_sw[:,0]=0
            # GRIDCELLS_sw[0,:]=0


            gcInZPlaneNew = gcInZPlaneNew*weight_ne + GRIDCELLS_nw*weight_nw + GRIDCELLS_se*weight_se + GRIDCELLS_sw*weight_sw

            gcInZPlane90 = gcInZPlaneNew[1:-1,1:-1]
            gcInZPlane90[1:,0] = gcInZPlane90[1:,0] + gcInZPlaneNew[2:-1,-1]
            gcInZPlane90[0,1:] = gcInZPlane90[0,1:] + gcInZPlaneNew[-1,2:-1]
            gcInZPlane90[0,0] = gcInZPlane90[0,0] + gcInZPlaneNew[-1,-1]

            self.GRIDCELLS = np.rot90(gcInZPlane90, 4 - np.floor(curYawThetaInRadian * 2/np.pi))

    def iteration(self,curYawThetaInRadian,transV,score_pose=None):
        
        # print("curYawThetaInRadian",curYawThetaInRadian,"transV",transV)
        # if transV>0.9:
        #     self.move(curYawThetaInRadian,0.9)
        #     transV-=0.9
        self.move(curYawThetaInRadian,transV)
        
        last_x,last_y=self.get_pose()

        # print(last_x,last_y)
        for i in range(score_pose.shape[0]):
            for j in range(score_pose.shape[1]):
                self.GRIDCELLS[i,j]=self.GRIDCELLS[i,j]+self.GC_VT_INJECT_ENERGY*score_pose[i,j] #* 1/10 * (10 - np.exp(1/50 * np.sqrt((i-last_x)**2+(j-last_y)**2)))

##activation###########
        self.activation()

        

        # plt.imshow(self.GRIDCELLS, cmap='viridis', interpolation='nearest')
        # plt.savefig('after.png')
        # plt.clf()
    def activation(self):
        # excitation and inhibition
        gridcell_local_excit_new = np.zeros((self.GC_X_DIM, self.GC_Y_DIM))
        for x in range(self.GC_X_DIM):
            for y in range(self.GC_Y_DIM):
                if self.GRIDCELLS[x,y] != 0:
                    X_wrap=np.repeat(np.expand_dims(self.GC_EXCIT_X_WRAP[x : x + self.GC_EXCIT_X_DIM],axis=1),self.GC_EXCIT_Y_DIM,axis=1)
                    Y_wrap=np.repeat(np.expand_dims(self.GC_EXCIT_Y_WRAP[y : y + self.GC_EXCIT_Y_DIM],axis=0),self.GC_EXCIT_X_DIM,axis=0)
                    # X_wrap=np.clip(X_wrap,0,self.GC_X_DIM-1)
                    # Y_wrap=np.clip(Y_wrap,0,self.GC_Y_DIM-1)

                    gridcell_local_excit_new[X_wrap, Y_wrap] = \
                        gridcell_local_excit_new[X_wrap, Y_wrap] \
                        + self.GRIDCELLS[x,y] * self.GC_EXCIT_WEIGHT

        self.GRIDCELLS = gridcell_local_excit_new


        gridcell_local_inhib_new = np.zeros((self.GC_X_DIM, self.GC_Y_DIM))  
        for x in range(self.GC_X_DIM):
            for y in range(self.GC_Y_DIM):
                if self.GRIDCELLS[x,y] != 0:
                    X_wrap=np.repeat(np.expand_dims(self.GC_INHIB_X_WRAP[x : x + self.GC_INHIB_X_DIM],axis=1),self.GC_INHIB_Y_DIM,axis=1)
                    Y_wrap=np.repeat(np.expand_dims(self.GC_INHIB_Y_WRAP[y : y + self.GC_INHIB_Y_DIM],axis=0),self.GC_INHIB_X_DIM,axis=0)
                    # X_wrap=np.clip(X_wrap,0,self.GC_X_DIM-1)
                    # Y_wrap=np.clip(Y_wrap,0,self.GC_Y_DIM-1)

                    gridcell_local_inhib_new[X_wrap,Y_wrap] = \
                        gridcell_local_inhib_new[X_wrap,Y_wrap] \
                        + self.GRIDCELLS[x,y] * self.GC_INHIB_WEIGHT

        self.GRIDCELLS = self.GRIDCELLS - gridcell_local_inhib_new


        self.GRIDCELLS = (self.GRIDCELLS >= self.GC_GLOBAL_INHIB) * (self.GRIDCELLS - self.GC_GLOBAL_INHIB)

        total = (sum(sum(self.GRIDCELLS)))
        self.GRIDCELLS = self.GRIDCELLS/total

    def get_pose(self):
        # indexes = np.where(self.GRIDCELLS)  # Find non-zero elements' indices
        values = self.GRIDCELLS    # Values at those indices
        max_value_index = np.argmax(values)  # Index of the maximum value

        # print(indexes,self.GRIDCELLS.shape)
        # Convert linear index to subscripts
        subscripts = np.unravel_index(max_value_index, self.GRIDCELLS.shape)
        x, y = subscripts

        tempGridcells = np.zeros((self.GC_X_DIM, self.GC_Y_DIM))

        tempGridcells[self.GC_MAX_X_WRAP[x : x + self.GC_PACKET_SIZE * 2], \
            self.GC_MAX_Y_WRAP[y : y + self.GC_PACKET_SIZE * 2]] = \
            self.GRIDCELLS[self.GC_MAX_X_WRAP[x : x + self.GC_PACKET_SIZE * 2], \
            self.GC_MAX_Y_WRAP[y : y + self.GC_PACKET_SIZE * 2]]

        xSumSin = np.sum(self.GC_X_SUM_SIN_LOOKUP * np.sum(tempGridcells,1))
        xSumCos = np.sum(self.GC_X_SUM_COS_LOOKUP * np.sum(tempGridcells,1))

        ySumSin = np.sum(self.GC_Y_SUM_SIN_LOOKUP * np.sum(tempGridcells,0))
        ySumCos = np.sum(self.GC_Y_SUM_COS_LOOKUP * np.sum(tempGridcells,0))

        gcX = (np.arctan2(xSumSin, xSumCos) / self.GC_X_TH_SIZE) % self.GC_X_DIM
        gcY = (np.arctan2(ySumSin, ySumCos) / self.GC_Y_TH_SIZE) % self.GC_Y_DIM

        return gcX, gcY
    
    def get_packet(self):
        values = self.GRIDCELLS    # Values at those indices
        max_value_index = np.argmax(values)  # Index of the maximum value

        # print(indexes,self.GRIDCELLS.shape)
        # Convert linear index to subscripts
        subscripts = np.unravel_index(max_value_index, self.GRIDCELLS.shape)
        x, y = subscripts
        return self.GC_MAX_X_WRAP[x : x + self.GC_PACKET_SIZE * 2], self.GC_MAX_Y_WRAP[y : y + self.GC_PACKET_SIZE * 2]

        # return self.GC_MAX_X_WRAP[x : x + self.GC_PACKET_SIZE * 2], self.GC_MAX_Y_WRAP[y : y + self.GC_PACKET_SIZE * 2]

pc=pose_cell(GC_X_DIM=120,GC_Y_DIM=80,init_activation_loc=[[0,0]])

# look=pc.GC_EXCIT_X_WRAP[119 : 119 + pc.GC_EXCIT_X_DIM]
# print(pc.GC_EXCIT_X_WRAP.shape)
# print(look)
# for i in range(100):
#     pc.iteration(curYawThetaInRadian=0,transV=0)
#     plt.imshow(self.GRIDCELLS, cmap='viridis', interpolation='nearest')
#     plt.savefig('after.png')
#     plt.clf()


