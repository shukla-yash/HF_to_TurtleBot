#!/usr/bin/env python3.7

from enum import Enum
import os
import sys
import gym
import time
import numpy as np
import TurtleBot_v0
from SimpleDQN import SimpleDQN
import rospy
from geometry_msgs.msg import Twist

from movement_utils.srv import (
    ResetOdom,
    ResetOdomRequest,
    ResetOdomResponse,
    GetPosition,
    GetPositionRequest,
    GetPositionResponse,
    GoToRelative,
    GoToRelativeRequest,
    GoToRelativeResponse,
)

from qr_state_reader.srv import (
    ReadEnvironment,
    ReadEnvironmentRequest,
    ReadEnvironmentResponse,
)


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
            # 	if np.mean(reward_array[-10:]) > 950:
            # 		reward_cond = True

        if done_cond == True and reward_cond == True:
            return 1
        else:
            return 0
    else:
        return 0


class Action(Enum):
    STOP = 0
    FORWARD = 1
    CWISE = 2
    CCWISE = 3
    BREAK = 4
    CRAFT = 5


class TurtleBotRosNode(object):
    def __init__(self, timeout_seconds=15):
        rospy.init_node("TurtleBotCurriculumNav", anonymous=False)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Starting up turtlebot ROS node.")

        srv_str_get_position = "/movement_wrapper/get_position"
        srv_str_goto_relative = "/movement_wrapper/goto_relative"
        srv_str_reset_odom = "/movement_wrapper/reset_odom"
        srv_str_read_env = "/qr_state_reader/read_environment"

        try:
            rospy.loginfo("Attempting to connect to movement wrapper services.")
            rospy.wait_for_service(srv_str_get_position, timeout=timeout_seconds)
            self.service_get_position = rospy.ServiceProxy(
                srv_str_get_position, GetPosition
            )
            self.service_goto_position = rospy.ServiceProxy(
                srv_str_goto_relative, GoToRelative
            )
            self.service_reset_odom = rospy.ServiceProxy(srv_str_reset_odom, ResetOdom)
        except rospy.ROSException:
            rospy.logerr(
                "Tried accessing a movement_wrapper service but failed. Exiting."
            )
            sys.exit(1)
        rospy.loginfo("Connected to movement wrapper services successfully.")

        try:
            rospy.loginfo("Attempting to connect to qr service.")
            rospy.wait_for_service(srv_str_read_env, timeout=timeout_seconds)
            self.service_read_env = rospy.ServiceProxy(
                srv_str_read_env, ReadEnvironment
            )
        except rospy.ROSException:
            rospy.logerr(
                "Tried accessing the qr_state_reader service but failed. Exiting."
            )
            sys.exit(1)
        rospy.loginfo("Connected to qr service successfully. Ready to go!")

        self.reset_odom()

    def __del__(self):
        self.service_get_position.close()
        self.service_goto_position.close()
        self.service_reset_odom.close()
        self.service_read_env.close()

    def shutdown(self):
        self.halt()
        del self

    def get_position(self) -> GetPositionResponse:
        try:
            req = GetPositionRequest()
            return self.service_get_position(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))
            return GetPositionResponse()

    def goto_relative(self, req: GoToRelativeRequest) -> GoToRelativeResponse:
        resp = GoToRelativeResponse()
        try:
            resp = self.service_goto_position(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))

        return resp

    def act(self, action: Action):
        req = GoToRelativeRequest()
        if action == Action.STOP:
            req.movement = req.STOP
        if action == Action.CWISE:
            req.movement = req.CWISE
        if action == Action.CCWISE:
            req.movement = req.CCWISE
        if action == Action.FORWARD:
            req.movement = req.FORWARD
        if action == Action.CRAFT:
            time.sleep(2)
        if action == Action.BREAK:
            time.sleep(2)
        self.goto_relative(req)

    def halt(self):
        req = GoToRelativeRequest()
        req.movement = req.STOP
        self.goto_relative(req)

    def reset_odom(self) -> ResetOdomResponse:
        try:
            req = ResetOdomRequest()
            self.service_reset_odom(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))

        return ResetOdomResponse()

    def read_environment(self) -> ReadEnvironmentResponse:
        try:
            req = ReadEnvironmentRequest()
            return self.service_read_env(req)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:" + str(e))
            return ReadEnvironmentResponse()


def main():
    RosEnv = None
    no_of_environmets = 4

    width_array = [1.5, 2.5, 3, 3]
    height_array = [1.5, 2.5, 3, 3]
    no_trees_array = [1, 1, 3, 4]
    no_rocks_array = [0, 1, 2, 2]
    crafting_table_array = [0, 0, 1, 1]
    starting_trees_array = [0, 0, 0, 0]
    starting_rocks_array = [0, 0, 0, 0]
    type_of_env_array = [0, 1, 2, 2]

    total_timesteps_array = []
    total_reward_array = []
    avg_reward_array = []
    final_timesteps_array = []
    final_reward_array = []
    final_avg_reward_array = []
    task_completion_array = []

    actionCnt = 5
    D = 83  # 90 beams x 4 items lidar + 3 inventory items
    NUM_HIDDEN = 16
    GAMMA = 0.995
    LEARNING_RATE = 1e-3
    DECAY_RATE = 0.99
    MAX_EPSILON = 0.1
    random_seed = 1

    # agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
    # agent.set_explore_epsilon(MAX_EPSILON)
    action_space = ["W", "A", "D", "U", "C"]
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
            agent = SimpleDQN(
                actionCnt,
                D,
                NUM_HIDDEN,
                LEARNING_RATE,
                GAMMA,
                DECAY_RATE,
                MAX_EPSILON,
                random_seed,
            )
            agent.set_explore_epsilon(MAX_EPSILON)
        else:
            agent = SimpleDQN(
                actionCnt,
                D,
                NUM_HIDDEN,
                LEARNING_RATE,
                GAMMA,
                DECAY_RATE,
                MAX_EPSILON,
                random_seed,
            )
            agent.set_explore_epsilon(MAX_EPSILON)
            agent.load_model(0, 0, i)
            agent.reset()
            print("loaded model")

        if i == no_of_environmets - 1:
            final_status = True

        env_id = "TurtleBot-v0"
        env = gym.make(
            env_id,
            map_width=width,
            map_height=height,
            items_quantity={
                "tree": no_trees,
                "rock": no_rocks,
                "crafting_table": crafting_table,
                "pogo_stick": 0,
            },
            initial_inventory={
                "wall": 0,
                "tree": starting_trees,
                "rock": starting_rocks,
                "crafting_table": 0,
                "pogo_stick": 0,
            },
            goal_env=type_of_env,
            is_final=final_status,
        )

        t_step = 0
        episode = 0
        t_limit = 600
        reward_sum = 0
        reward_arr = []
        avg_reward = []
        done_arr = []
        env_flag = 0

        env.reset()
        if RosEnv is not None:
            del RosEnv
        RosEnv = TurtleBotRosNode()

        while True:

            # get obseration from sensor
            obs = env.get_observation()

            # act
            a = agent.process_step(obs, True)
            if a == 0:
                print("Right")
                RosEnv.act(Action.CWISE)
            elif a == 1:
                print("Left")
                RosEnv.act(Action.CCWISE)
            elif a == 2:
                print("Forward")
                RosEnv.act(Action.FORWARD)
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

                print(
                    "\n\nfinished episode = "
                    + str(episode)
                    + " with "
                    + str(reward_sum)
                    + "\n"
                )

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
                if i < 3:
                    env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, i)

                # quit after some number of episodes
                if episode > 0 or env_flag == 1:

                    break
    print("done")


if __name__ == "__main__":
    main()
