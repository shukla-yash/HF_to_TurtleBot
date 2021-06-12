#!/usr/bin/env python3.7

import os
import sys
import gym
import time
import numpy as np
import TurtleBot_v0
from SimpleDQN import SimpleDQN
import rospy
from geometry_msgs.msg import Twist
from movement_utils.srv import *

def CheckTrainingDoneCallback(reward_array, done_array, env):
    done_cond = False
    reward_cond = False
    if len(done_array) > 40:
        if np.mean(done_array[-10:]) > 0.85 and np.mean(done_array[-40:]) > 0.85:
            if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
                done_cond = True

        if done_cond == True:
            if env < 3:
                if np.mean(reward_array[-40:]) > 730:
                    reward_cond = True
            # else:
            #     if np.mean(reward_array[-10:]) > 950:
            #         reward_cond = True

        if done_cond == True and reward_cond == True:
            return 1
        else:
            return 0
    else:
        return 0

class GoForward():
    def __init__(self, action):
        # initiliaze
        rospy.init_node('GoForward', anonymous=False)

        # What function to call when you ctrl + c
        rospy.on_shutdown(self.shutdown)

        # Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

        #TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(10);

        # Twist is a datatype for velocity
        move_cmd = Twist()
        if action == 2:
            move_cmd.linear.x = 0.2
            move_cmd.angular.z = 0
        elif action == 1:
            move_cmd.linear.x = 0
            move_cmd.angular.z = -0.35
        elif action == 0:
            move_cmd.linear.x = 0
            move_cmd.angular.z = 0.35

        for j in range(1):
            for i in range(10):
                self.cmd_vel.publish(move_cmd)
                r.sleep()

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        self.cmd_vel.publish(Twist())

def main():
    no_of_environmets = 4

    width_array = [1.5,2.5,3,3]
    height_array = [1.5,2.5,3,3]
    no_trees_array = [1,1,3,4]
    no_rocks_array = [0,1,2,2]
    crafting_table_array = [0,0,1,1]
    starting_trees_array = [0,0,0,0]
    starting_rocks_array = [0,0,0,0]
    type_of_env_array = [0,1,2,2]

    total_timesteps_array = []
    total_reward_array = []
    avg_reward_array = []
    final_timesteps_array = []
    final_reward_array = []
    final_avg_reward_array = []
    task_completion_array = []

    actionCnt = 5
    D = 83 #90 beams x 4 items lidar + 3 inventory items
    NUM_HIDDEN = 16
    GAMMA = 0.995
    LEARNING_RATE = 1e-3
    DECAY_RATE = 0.99
    MAX_EPSILON = 0.1
    random_seed = 1

    # agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
    # agent.set_explore_epsilon(MAX_EPSILON)
    action_space = ['W','A','D','U','C']
    total_episodes_arr = []

    for k in range(1):
        i = 3
        print("Environment: ", i)

        width = width_array[i]
        height = height_array[i]
        no_trees = no_trees_array[i]
        no_rocks = no_rocks_array[i]
        crafting_table = crafting_table_array[i]
        starting_trees = starting_trees_array[i]
        starting_rocks = starting_rocks_array[i]
        type_of_env = type_of_env_array[i]

        final_status = False

        if i == 0:
            agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
            agent.set_explore_epsilon(MAX_EPSILON)
        else:
            agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
            agent.set_explore_epsilon(MAX_EPSILON)
            agent.load_model(0,0,i)
            agent.reset()
            print("loaded model")


        if i == no_of_environmets-1:
            final_status = True

        env_id = 'TurtleBot-v0'
        env = gym.make(env_id, map_width = width, map_height = height, items_quantity = {'tree': no_trees, 'rock': no_rocks, 'crafting_table': crafting_table, 'pogo_stick':0},
            initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks, 'crafting_table': 0, 'pogo_stick':0}, goal_env = type_of_env, is_final = final_status)

        t_step = 0
        episode = 0
        t_limit = 600
        reward_sum = 0
        reward_arr = []
        avg_reward = []
        done_arr = []
        env_flag = 0

        env.reset()

        while True:

            # get obseration from sensor
            obs = env.get_observation()

            # act
            a = agent.process_step(obs,True)
            if a == 0:
                print("Right")
                GoForward(0)
            elif a == 1:
                print("Left")
                GoForward(1)
            elif a == 2:
                print("Forward")
                GoForward(2)
            elif a == 3:
                print("Break")
            elif a == 4:
                print("Craft")

            new_obs, reward, done, info = env.step(a)

            # give reward
            agent.give_reward(reward)
            reward_sum += reward

            t_step += 1

            if t_step > t_limit or done == True:

                # finish agent
                if done == True:
                    done_arr.append(1)
                    task_completion_array.append(1)
                elif t_step > t_limit:
                    done_arr.append(0)
                    task_completion_array.append(0)

                print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")

                reward_arr.append(reward_sum)
                avg_reward.append(np.mean(reward_arr[-40:]))

                total_reward_array.append(reward_sum)
                avg_reward_array.append(np.mean(reward_arr[-40:]))
                total_timesteps_array.append(t_step)

                done = True
                t_step = 0
                agent.finish_episode()

                # update after every episode
                if episode % 10 == 0:
                    agent.update_parameters()

                # reset environment
                episode += 1

                env.reset()
                reward_sum = 0

                env_flag = 0
                if i< 3:
                    env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, i)

                # quit after some number of episodes
                if episode > 0 or env_flag == 1:

                    break
    print("done")

if __name__ == "__main__":
    main()
