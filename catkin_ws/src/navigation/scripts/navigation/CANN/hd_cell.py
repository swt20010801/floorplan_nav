import numpy as np
import matplotlib.pyplot as plt
import time
def create_hdc_weights(yawDim, yawVar):

    yawDimCentre = np.floor((yawDim-1) / 2) 

    weight = np.zeros(yawDim)

    for y in range(yawDim):
        weight[y] =  1/(yawVar*np.sqrt(2*np.pi))*np.exp((-(y - yawDimCentre) ** 2 ) / (2 * yawVar ** 2))


    total = sum(weight)
    weight = weight/total     

    return weight

class hd_cell():
    def __init__(
        self,
        YAW_HEIGHT_HDC_Y_DIM=16,
        YAW_HEIGHT_HDC_EXCIT_Y_DIM=7,
        YAW_HEIGHT_HDC_INHIB_Y_DIM=5,
        YAW_HEIGHT_HDC_EXCIT_Y_VAR=1.9,
        YAW_HEIGHT_HDC_INHIB_Y_VAR=3.1,
        YAW_HEIGHT_HDC_GLOBAL_INHIB=0.0002,
        YAW_HEIGHT_HDC_VT_INJECT_ENERGY=0.1,
        YAW_HEIGHT_HDC_PACKET_SIZE=5,
        init_activation_yaw=None
    ):

        self.YAW_HEIGHT_HDC_Y_DIM=YAW_HEIGHT_HDC_Y_DIM
        self.YAW_HEIGHT_HDC_EXCIT_Y_DIM=YAW_HEIGHT_HDC_EXCIT_Y_DIM
        self.YAW_HEIGHT_HDC_INHIB_Y_DIM=YAW_HEIGHT_HDC_INHIB_Y_DIM
        self.YAW_HEIGHT_HDC_EXCIT_Y_VAR=YAW_HEIGHT_HDC_EXCIT_Y_VAR
        self.YAW_HEIGHT_HDC_INHIB_Y_VAR=YAW_HEIGHT_HDC_INHIB_Y_VAR
        self.YAW_HEIGHT_HDC_GLOBAL_INHIB=YAW_HEIGHT_HDC_GLOBAL_INHIB
        self.YAW_HEIGHT_HDC_VT_INJECT_ENERGY=YAW_HEIGHT_HDC_VT_INJECT_ENERGY
        self.YAW_HEIGHT_HDC_PACKET_SIZE=YAW_HEIGHT_HDC_PACKET_SIZE

        self.YAW_HEIGHT_HDC_EXCIT_WEIGHT = create_hdc_weights(YAW_HEIGHT_HDC_EXCIT_Y_DIM, YAW_HEIGHT_HDC_EXCIT_Y_VAR)
        self.YAW_HEIGHT_HDC_INHIB_WEIGHT = create_hdc_weights(YAW_HEIGHT_HDC_INHIB_Y_DIM, YAW_HEIGHT_HDC_INHIB_Y_VAR)

        self.YAW_HEIGHT_HDC_EXCIT_Y_DIM_HALF = np.floor(YAW_HEIGHT_HDC_EXCIT_Y_DIM / 2)
        self.YAW_HEIGHT_HDC_INHIB_Y_DIM_HALF = np.floor(YAW_HEIGHT_HDC_INHIB_Y_DIM / 2)
    
        self.YAW_HEIGHT_HDC_Y_TH_SIZE = (2 * np.pi) / YAW_HEIGHT_HDC_Y_DIM

        self.YAW_HEIGHT_HDC_Y_SUM_SIN_LOOKUP = np.sin(np.arange(YAW_HEIGHT_HDC_Y_DIM) * self.YAW_HEIGHT_HDC_Y_TH_SIZE)
        self.YAW_HEIGHT_HDC_Y_SUM_COS_LOOKUP = np.cos(np.arange(YAW_HEIGHT_HDC_Y_DIM) * self.YAW_HEIGHT_HDC_Y_TH_SIZE)

        self.YAW_HEIGHT_HDC_EXCIT_Y_WRAP = np.concatenate((np.arange(YAW_HEIGHT_HDC_Y_DIM - self.YAW_HEIGHT_HDC_EXCIT_Y_DIM_HALF,YAW_HEIGHT_HDC_Y_DIM),np.arange(YAW_HEIGHT_HDC_Y_DIM),np.arange(self.YAW_HEIGHT_HDC_EXCIT_Y_DIM_HALF)))
        self.YAW_HEIGHT_HDC_INHIB_Y_WRAP = np.concatenate((np.arange(YAW_HEIGHT_HDC_Y_DIM - self.YAW_HEIGHT_HDC_INHIB_Y_DIM_HALF,YAW_HEIGHT_HDC_Y_DIM),np.arange(YAW_HEIGHT_HDC_Y_DIM),np.arange(self.YAW_HEIGHT_HDC_INHIB_Y_DIM_HALF)))
        self.YAW_HEIGHT_HDC_EXCIT_Y_WRAP=self.YAW_HEIGHT_HDC_EXCIT_Y_WRAP.astype(np.int32)
        self.YAW_HEIGHT_HDC_INHIB_Y_WRAP=self.YAW_HEIGHT_HDC_INHIB_Y_WRAP.astype(np.int32)



        self.YAW_HEIGHT_HDC_MAX_Y_WRAP = np.concatenate((np.arange(YAW_HEIGHT_HDC_Y_DIM - self.YAW_HEIGHT_HDC_PACKET_SIZE,YAW_HEIGHT_HDC_Y_DIM),np.arange(YAW_HEIGHT_HDC_Y_DIM),np.arange(self.YAW_HEIGHT_HDC_PACKET_SIZE)))
        self.YAW_HEIGHT_HDC_MAX_Y_WRAP=self.YAW_HEIGHT_HDC_MAX_Y_WRAP.astype(np.int32)


        if init_activation_yaw !=None:
            curYawTheta=int(init_activation_yaw/self.YAW_HEIGHT_HDC_Y_TH_SIZE)
        else:
            curYawTheta = self.get_hdc_initial_value()

        self.YAW_HEIGHT_HDC = np.zeros(YAW_HEIGHT_HDC_Y_DIM)
        self.YAW_HEIGHT_HDC[curYawTheta] = 1

        self.activation()


    def get_hdc_initial_value(self):
        curYawTheta=0
        return curYawTheta

    def iteration(self,yawRotV,score_yaw):
        # motion

        if yawRotV != 0:

            weight = (abs(yawRotV) / self.YAW_HEIGHT_HDC_Y_TH_SIZE) % 1
            if weight == 0:
                weight = 1.0

            self.YAW_HEIGHT_HDC = np.roll(self.YAW_HEIGHT_HDC,shift= \
                (np.sign(yawRotV) * np.floor(np.mod(abs(yawRotV) / self.YAW_HEIGHT_HDC_Y_TH_SIZE, self.YAW_HEIGHT_HDC_Y_DIM))).astype(np.int32),axis=0) * (1.0 - weight) \
                + np.roll(self.YAW_HEIGHT_HDC, shift=\
                (np.sign(yawRotV) * np.ceil(np.mod(abs(yawRotV) / self.YAW_HEIGHT_HDC_Y_TH_SIZE, self.YAW_HEIGHT_HDC_Y_DIM))).astype(np.int32),axis=0) * (weight)

        # print("yaw_score",score_yaw)
        # print("hdc",self.YAW_HEIGHT_HDC)



        for i in range(score_yaw.shape[0]):
            self.YAW_HEIGHT_HDC[i]=self.YAW_HEIGHT_HDC[i]+self.YAW_HEIGHT_HDC_VT_INJECT_ENERGY*score_yaw[i]

##activation########
        self.activation()
        # print("hdc_after",self.YAW_HEIGHT_HDC)

        # plt.imshow(np.expand_dims(self.YAW_HEIGHT_HDC,axis=0), cmap='viridis', interpolation='nearest')
        # plt.savefig('hdc.png')
        # plt.clf()

    def activation(self):
        # excitation and inhibition
        yaw_height_hdc_local_excit_new = np.zeros(self.YAW_HEIGHT_HDC_Y_DIM)
        for y in range(self.YAW_HEIGHT_HDC_Y_DIM):
            if self.YAW_HEIGHT_HDC[y] != 0:
                yaw_height_hdc_local_excit_new[self.YAW_HEIGHT_HDC_EXCIT_Y_WRAP[y : y + self.YAW_HEIGHT_HDC_EXCIT_Y_DIM]] = \
                    yaw_height_hdc_local_excit_new[self.YAW_HEIGHT_HDC_EXCIT_Y_WRAP[y : y + self.YAW_HEIGHT_HDC_EXCIT_Y_DIM]] \
                        + self.YAW_HEIGHT_HDC[y] * self.YAW_HEIGHT_HDC_EXCIT_WEIGHT

        self.YAW_HEIGHT_HDC = yaw_height_hdc_local_excit_new

        yaw_height_hdc_local_inhib_new = np.zeros(self.YAW_HEIGHT_HDC_Y_DIM)  
        for y in range(self.YAW_HEIGHT_HDC_Y_DIM):
            if self.YAW_HEIGHT_HDC[y] != 0:
                yaw_height_hdc_local_inhib_new[self.YAW_HEIGHT_HDC_INHIB_Y_WRAP[y : y + self.YAW_HEIGHT_HDC_INHIB_Y_DIM]] = \
                    yaw_height_hdc_local_inhib_new[self.YAW_HEIGHT_HDC_INHIB_Y_WRAP[y : y + self.YAW_HEIGHT_HDC_INHIB_Y_DIM]] \
                    + self.YAW_HEIGHT_HDC[y] * self.YAW_HEIGHT_HDC_INHIB_WEIGHT

        self.YAW_HEIGHT_HDC = self.YAW_HEIGHT_HDC - yaw_height_hdc_local_inhib_new

        self.YAW_HEIGHT_HDC = (self.YAW_HEIGHT_HDC >= self.YAW_HEIGHT_HDC_GLOBAL_INHIB) * (self.YAW_HEIGHT_HDC - self.YAW_HEIGHT_HDC_GLOBAL_INHIB)
        
        total = (sum(self.YAW_HEIGHT_HDC))
        self.YAW_HEIGHT_HDC = self.YAW_HEIGHT_HDC/total

    def get_yaw(self):

        y = np.argmax(self.YAW_HEIGHT_HDC)

        tempYawHeightHdc = np.zeros(self.YAW_HEIGHT_HDC_Y_DIM)

        tempYawHeightHdc[self.YAW_HEIGHT_HDC_MAX_Y_WRAP[y : y + self.YAW_HEIGHT_HDC_PACKET_SIZE * 2]] = \
            self.YAW_HEIGHT_HDC[self.YAW_HEIGHT_HDC_MAX_Y_WRAP[y : y + self.YAW_HEIGHT_HDC_PACKET_SIZE * 2]]

        yawSumSin = np.sum(self.YAW_HEIGHT_HDC_Y_SUM_SIN_LOOKUP * tempYawHeightHdc)
        yawSumCos = np.sum(self.YAW_HEIGHT_HDC_Y_SUM_COS_LOOKUP * tempYawHeightHdc)


        outYawTheta = (np.arctan2(yawSumSin, yawSumCos) / self.YAW_HEIGHT_HDC_Y_TH_SIZE) % self.YAW_HEIGHT_HDC_Y_DIM

        return outYawTheta*self.YAW_HEIGHT_HDC_Y_TH_SIZE