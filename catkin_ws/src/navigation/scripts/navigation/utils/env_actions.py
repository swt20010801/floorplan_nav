import torch
import numpy as np
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose
from tf2_msgs.msg import TFMessage
import cv2
from scipy.spatial.transform import Rotation as R



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

    return image



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

def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler

def get_robot_position():
    robot_pos=np.zeros(2)

    msg = rospy.wait_for_message("/tf", TFMessage, timeout=None)

    robot_pos[0]=  msg.transforms[0].transform.translation.x+msg.transforms[1].transform.translation.x
    robot_pos[1]=-(msg.transforms[0].transform.translation.y+msg.transforms[1].transform.translation.y)

    quaternion_0=[msg.transforms[0].transform.rotation.x,msg.transforms[0].transform.rotation.y,msg.transforms[0].transform.rotation.z,msg.transforms[0].transform.rotation.w]
    quaternion_1=[msg.transforms[1].transform.rotation.x,msg.transforms[1].transform.rotation.y,msg.transforms[1].transform.rotation.z,msg.transforms[1].transform.rotation.w]

    theta_0=quaternion2euler(quaternion_0)
    theta_1=quaternion2euler(quaternion_1)
    
    rot=-(theta_0[2]+theta_1[2])/180*np.pi
    return robot_pos,rot