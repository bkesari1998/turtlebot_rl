
import rospy
from std_msgs.msg import Int16
from std_msgs.msg import String
import numpy as np
import scipy.ndimage as sp_img
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import gym
import matplotlib.pyplot as plt

FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
STOP = 4

NUM_STEPS = 60

class Env():
    def __init__(self) -> None:
        super(Env, self).__init__()

        self.single_action_space = gym.spaces.Discrete(4)
        self.single_observation_space = (3, 100, 100)
        self.single_action_space = np.array([FORWARD, BACKWARD, LEFT, RIGHT])
        self.img_flag = 0
        self.img = np.zeros((3, 100, 100))
        self.reward = 0
        self.robot_ready = 0
        self.env_fns = 1

        # init node
        rospy.init_node('env', anonymous=False)

        # Rosrate
        self.rate = rospy.Rate(1)
        # command publisher
        self.cmd_pub = rospy.Publisher('command', Int16, queue_size=10)
        # image array publisher
        self.img_array_sub = rospy.Subscriber('img', Floats, self.set_img_flag)

    def set_img_flag(self, data):
        observation= np.array(data.data).reshape((480,640,3))
        observation = np.transpose(observation[0:480,0:640,0:3],[2,0,1])
        observation = sp_img.zoom(observation, zoom = (1, 0.2083333333, 0.15625), order=1)
        observation = observation / 255
        self.img = observation

        self.img_flag = 1
        self.check_color()


    # detects color for the ball for the reward 
    # 15 is a good threshold, that and above are rewards
    def check_color(self):
        RED = 0
        GREEN = 1
        BLUE = 2

        limits = {}
        limits[RED] = (0.392,1)
        limits[GREEN] = (0.196, 0.78)
        limits[BLUE] = (0,0.196)

        # mask each rgb value by its respective range
        tmp_img = np.transpose(self.img,[1, 2, 0])
        mask = np.zeros(tmp_img.shape)
        for i in range(3):
            mask[:,:,i] = (limits[i][0] <= tmp_img[:,:,i]) & (tmp_img[:,:,i] <= limits[i][1])

        # mask by pixels and calculate percent
        total = np.sum(np.all(mask,2))
        percent = 100*total/np.prod(self.single_observation_space[0:2])
        
        if (percent >= 20):
            self.reward = 1000
        elif (percent >= 15):
            self.reward = 50
        elif (percent >= 10):
            self.reward = 10
        else:
            self.reward == 0

        return 

    def step(self, action, step):

        self.cmd_pub.publish(int(action))
        self.rate.sleep()

        while(self.img_flag == 0):
            continue
        
        done = 0
        if self.reward == 1000:
            done = 1
        
        self.img_flag == 0

        if step == NUM_STEPS - 1:
            done == 1

        img_ret = self.img.reshape((1,3,100,100))
        return img_ret, self.reward, done

    def reset(self):
        self.img_flag = 0
        self.reward = 0
        flag = 1

        while flag == 1:
            flag = input("Enter 0 when ready: ")
        

        self.cmd_pub.publish(STOP)
        self.rate.sleep()
        
        while(self.img_flag == 0):
            continue

        self.img_flag == 0

        img_ret = self.img.reshape((1,3,100,100))
        return img_ret
