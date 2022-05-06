#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16
from math import pi


class TurtleBot:
    def __init__(self):
        
        # initialize turtlebot node
        rospy.init_node('turtlebot', anonymous=False)
        rospy.loginfo('Press CTRL-C to stop')
        # rospy.on_shutdown(self.shutdown)

        # run at 1Hz
        self.rate = rospy.Rate(1)

        self.move_cmd = Twist()
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=1)
        self.flag_pub = rospy.Publisher('flag', Int16, queue_size=1)
        self.cmd_sub = rospy.Subscriber('command', Int16, self.process_cmd)

        self.stop = 0


        while not rospy.is_shutdown():
            rospy.spin()
    
    def process_cmd(self, data):
        if data.data == 0:
            self.forward()
        elif data.data == 1:
            self.backward()
        elif data.data == 2:
            self.counter_clockwise()
        elif data.data == 3:
            self.clockwise()

        flag = 1
        self.flag_pub.publish(flag)
        

    def shutdown(self):
        self.move_cmd = Twist()
        self.cmd_vel.publish(self.move_cmd)
        self.rate.sleep()
        rospy.loginfo('Stopping turtlebot')
        rospy.sleep(1)

    def forward(self):
        if not self.stop:
            self.move_cmd = Twist()
            self.move_cmd.linear.x = 0.2
            self.cmd_vel.publish(self.move_cmd)
            self.rate.sleep()
    
    def backward(self):
        if not self.stop:
            self.move_cmd = Twist()
            self.move_cmd.linear.x = -0.2
            self.cmd_vel.publish(self.move_cmd)
            self.rate.sleep()

    def clockwise(self):
        if not self.stop:
            self.move_cmd = Twist()
            self.move_cmd.angular.z = -pi / 6
            self.cmd_vel.publish(self.move_cmd)
            self.rate.sleep()

    def counter_clockwise(self):
        if not self.stop:
            self.move_cmd = Twist()
            self.move_cmd.angular.z = pi / 6
            self.cmd_vel.publish(self.move_cmd)
            self.rate.sleep()



if __name__ == '__main__':
    TurtleBot()
