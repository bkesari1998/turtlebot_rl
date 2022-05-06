#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int16

FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3

def publish_cmd():
    pub = rospy.Publisher('command', Int16, queue_size=10)
    rospy.init_node('cmd_publisher', anonymous=False)
    rate = rospy.Rate(1)
    cmds = [FORWARD, BACKWARD, LEFT, RIGHT]
    while not rospy.is_shutdown():
        cmd = input('Enter command: ')
        if int(cmd) not in cmds:
            continue
        pub.publish(int(cmd))
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_cmd()
    except rospy.ROSInterruptException:
        pass

