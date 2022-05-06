#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rospy_tutorials.msg import Floats
from std_msgs.msg import Int16
from rospy.numpy_msg import numpy_msg
import numpy as np

class ImgPub():
  def __init__(self):
      
    self.flag = 1
    self.bridge = CvBridge()

    rospy.init_node('image_pub')
    self.rate = rospy.Rate(1)

    self.img_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
    self.flag_sub = rospy.Subscriber('flag', Int16, self.set_flag)
    self.img_pub = rospy.Publisher('img', numpy_msg(Floats), queue_size=1)

    while not rospy.is_shutdown():
      rospy.spin()

  def set_flag(self, data):
    self.flag = data.data

  def image_callback(self, img_msg):
    
    if self.flag == 0:
      return
    
    try:
      cv_image = self.bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError as e:
      # rospy.logerr("CvBridge Error: {0}".format(e))
      return
    
    img = np.array(cv_image, dtype=np.float32).reshape(-1)
    self.img_pub.publish(img)
    self.flag = 0

if __name__ == '__main__':
  ImgPub()
