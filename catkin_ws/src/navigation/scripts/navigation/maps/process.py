import cv2
import numpy as np
import os

def the_same_line(pointA,pointB,img):
    if pointA[0]==pointB[0] and pointA[1]!=pointB[1]:
        x=pointA[0]
        for y in range(pointA[1],pointB[1]):
            if(img[y,x,:].all()!=np.array([255,255,255]).all()):
                return False
        return True
    elif pointA[1]==pointB[1] and pointA[0]!=pointB[0]:
        y=pointA[1]
        for x in range(pointA[0],pointB[0]):
            if(img[y,x,:].all()!=np.array([255,255,255]).all()):
                return False
        return True
    else:
        return False

def read_room_lines(img_path):
    img=cv2.imread(img_path)

    width =img.shape[1]
    height=img.shape[0]
    
    corner_list=[]

    for x in range(width):
        for y in range(height):
            if(img[y,x,:].all()==np.array([255,255,255]).all()):
                up=0
                down=0
                left=0
                right=0
                if(img[y+1,x,:].all()==np.array([255,255,255]).all()):
                    down=1
                if(img[y-1,x,:].all()==np.array([255,255,255]).all()):
                    up=1
                if(img[y,x+1,:].all()==np.array([255,255,255]).all()):
                    right=1
                if(img[y,x-1,:].all()==np.array([255,255,255]).all()):
                    left=1
                if(up==1 and left==1):#is a corner point
                    corner_point=np.array([x,y])
                    corner_list.append(corner_point)
                elif(up==1 and right==1):
                    corner_point=np.array([x,y])
                    corner_list.append(corner_point)
                elif(down==1 and left==1):
                    corner_point=np.array([x,y])
                    corner_list.append(corner_point)
                elif(down==1 and right==1):
                    corner_point=np.array([x,y])
                    corner_list.append(corner_point)
    lines=[]
    corner_num=len(corner_list)
    for i in range(corner_num):
        for j in range(i+1,corner_num):
            corner_point_A=corner_list[i]
            corner_point_B=corner_list[j]

            if the_same_line(corner_point_A,corner_point_B,img=img):
                lines.append(np.array([corner_point_A,corner_point_B]))

    lines=np.stack(lines,axis=0)
    # print((lines))
    return lines

def read_semantic_lines(img_path):
    img=cv2.imread(img_path)

    width =img.shape[1]
    height=img.shape[0]
    
    corner_list=[]

    for x in range(width):
        for y in range(height):
            if(img[y,x,:].all()==np.array([255,255,255]).all()):
                up=0
                down=0
                left=0
                right=0
                if(img[y+1,x,:].all()==np.array([255,255,255]).all()):
                    down=1
                if(img[y-1,x,:].all()==np.array([255,255,255]).all()):
                    up=1
                if(img[y,x+1,:].all()==np.array([255,255,255]).all()):
                    right=1
                if(img[y,x-1,:].all()==np.array([255,255,255]).all()):
                    left=1
                if(up==1 and down==0):#is a end point
                    corner_point=np.array([x,y])
                    corner_list.append(corner_point)
                elif(down==1 and up==0):
                    corner_point=np.array([x,y])
                    corner_list.append(corner_point)
                elif(left==1 and right==0):
                    corner_point=np.array([x,y])
                    corner_list.append(corner_point)
                elif(right==1 and left==0):
                    corner_point=np.array([x,y])
                    corner_list.append(corner_point)
    lines=[]
    corner_num=len(corner_list)
    for i in range(corner_num):
        for j in range(i+1,corner_num):
            corner_point_A=corner_list[i]
            corner_point_B=corner_list[j]

            if the_same_line(corner_point_A,corner_point_B,img=img):
                lines.append(np.array([corner_point_A,corner_point_B]))

    lines=np.stack(lines,axis=0)
    # print((lines))

    return lines

if __name__ == '__main__':
    House_name="House_13"
    room_lines_img_path=os.path.join(House_name,'roomlines.png')
    door_lines_img_path=os.path.join(House_name,'doorlines.png')
    window_lines_img_path=os.path.join(House_name,'windowlines.png')

    room_lines=read_room_lines(room_lines_img_path)
    door_lines=read_semantic_lines(door_lines_img_path)
    window_lines=read_semantic_lines(window_lines_img_path)

    room_lines_npy_path=os.path.join("../unitydataset",House_name,'room_lines.npy')
    door_lines_npy_path=os.path.join("../unitydataset",House_name,'door_lines.npy')
    window_lines_npy_path=os.path.join("../unitydataset",House_name,'window_lines.npy')

    np.save(room_lines_npy_path,room_lines)
    np.save(door_lines_npy_path,door_lines)
    np.save(window_lines_npy_path,window_lines)

