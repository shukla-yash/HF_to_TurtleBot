# HF_to_TurtleBot
Direct Policy Transfer between HF and TurtleBot

To run the TurtleBot, use the command : $ rosrun turtlebot_navigation test_curr.py

The gym environment can be found at turtlebot_navigation/src/TurtleBot_crafting/TurtleBot_v0/envs/turtlebot_vo_env.py
The step function (#142) in turtlebot_vo_env.py dictates the motion for the agent and the service call when the agent performs breaks action. 
