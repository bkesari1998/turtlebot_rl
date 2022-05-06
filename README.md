# probrobfinalproj

This repository contains the ROS nodes and edited main and ppo network code to run the navigation task on the physical robot. All three group members (Bharat, Zach, and Niko) contributed to this porition of the code.

## ROS Nodes

### turtlebot.py
This node contains code to publish Twist message actions to the turtlebot kobuki base. The actions are predetermined velocities that execute for 1 second.
The node recieves actions over the command topic. Once an action has been executed, a flag (1) is published on the flag topic.

### env.py
This node interfaces with the PPO algorithm. The node recieve an image over the img topic as numpy arrays. It calculates a reward based on the image. The reward
and image are passed to the PPO agent which determines the next action. The action from the PPO agent is published on the command topic

### img_pub.py
This node contains code to publish a camera image as a numpy array over the img topic. The node waits for a flag (1) on the flag topic before it publishes the image.

### cmd_pub.py
This node is used to manually publish commands over the command topic. This node is used for testing only.

## Python scipts

### main.py
The main python program. This script executes the trials based on command line inputs.

### ppo_agent.py
This program calls functions from env.py to advance the episodes. The data that is returned from these function is fed into the neural network. This script
returns an action to be published to the turtlebot back env.py.

### model_cnn.py
This script are the actual convolutional neural networks that calculate an action based on an image and reward.


## How To Run
Create a ros package containing all of the listed python scripts. On a desktop computer, set the ROS_MASTER_URI to the turtlebot's laptop IP address 
and port. On the desktop computer Create a conda environment with the package versions as listed in
conda_env.txt from the desktop computer. From the turtlebot's computer run:

`roslaunch turtlebot_bringup minimal.lauch`
and
`roslaunch astra_camera astra.launch`

Rosrun the img_pub.py and turtlebot.py nodes on the desktop computer. Run main.py within the conda environment with the `python` command on the host computer.
The command line arguments to main.py are listed below:

`python main.py --num-envs 1 --num-steps 120 --num-minibatches 3 --total-timesteps 12000 --train`

This will start a training session of 100 episodes.
