#!/usr/bin/env python3.7

import math
import time
import numpy as np
import os

# import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import copy
from qr_state_reader.srv import *
import rospy

# import pybullet as p

# REWARD_STEP = -1
# REWARD_DONE = 5000
# REWARD_BREAK = 300
# angle_increment = np.pi/60
# half_beams = 60
# number_of_episodes = 50000
# time_per_episode = 600


class TurtleBotV0Env(gym.Env):
    def __init__(
        self,
        map_width=None,
        map_height=None,
        items_id=None,
        items_quantity=None,
        initial_inventory=None,
        goal_env=None,
        is_final=False,
    ):
        # super(TurtleBotV0Env, self).__init__()

        self.width = map_width
        self.height = map_height
        self.object_types = [
            0,
            1,
            2,
            3,
        ]  # we have 4 objects: wall, tree, rock, and craft table

        self.reward_step = -1
        self.reward_done = 1000
        if is_final == True:
            self.reward_break = 0
        else:
            self.reward_break = 50

        self.reward_hit_wall = -10
        self.reward_extra_inventory = 0

        self.half_beams = 10
        self.angle_increment = np.pi / 10
        self.angle_increment_deg = 18

        self.time_per_episode = 300
        self.sense_range = 5.7

        low = np.zeros(self.half_beams * 2 * len(self.object_types) + 3)
        high = np.ones(self.half_beams * 2 * len(self.object_types) + 3)
        # inventory_array = np.array([5,2])
        # high = np.append(high_array, inventory_array)
        self.observation_space = spaces.Box(low, high, dtype=float)
        self.action_space = spaces.Discrete(5)
        self.num_envs = 1
        self.reset_time = 0

        self.n_trees_org = items_quantity["tree"]
        self.n_rocks_org = items_quantity["rock"]
        self.n_crafting_table = items_quantity["crafting_table"]
        self.starting_trees = initial_inventory["tree"]
        self.starting_rocks = initial_inventory["rock"]
        self.goal_env = goal_env

    def reset(self):

        # print("reset called: ", self.reset_time)
        self.reset_time += 1

        self.env_step_counter = 0
        offset = [0, 0, 0]

        self.agent_loc = [0, 0]
        self.agent_orn = np.pi / 2

        self.trees = []
        self.rocks = []
        self.table = []
        self.n_trees = self.n_trees_org
        self.n_rocks = self.n_rocks_org
        self.n_table = self.n_crafting_table
        # x_rand = np.random.rand(self.n_trees + self.n_rocks + self.n_table, 1)
        # y_rand = np.random.rand(self.n_trees + self.n_rocks + self.n_table, 1)

        #
        x_rand = np.array(
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.48]
        )  # First 4 are for trees, next 2 for rocks and the last for crafting table
        y_rand = np.array([0.3, 0.1, 0.62, 0.41, 0.9, 0.25, 0.1])
        self.x_pos = []
        self.y_pos = []
        self.map = np.zeros((int(self.width * 10), int(self.height * 10)))
        self.rocks_broken = []
        self.trees_broken = []

        for i in range(self.n_trees):  # Instantiate the trees
            self.x_pos.append(
                -self.width / 2 + self.width * x_rand[i]
            )  # (Tree 1 will be at absolute location: -1.5 + 3*0.1 = -1.2)
            self.y_pos.append(-self.height / 2 + self.height * y_rand[i])
            self.map[int(self.width * 10 * x_rand[i])][
                int(self.height * 10 * y_rand[i])
            ] = 1

        for i in range(self.n_rocks):  # Instantiate the rocks
            self.x_pos.append(-self.width / 2 + self.width * x_rand[i + self.n_trees])
            self.y_pos.append(-self.height / 2 + self.height * y_rand[i + self.n_trees])
            self.map[int(self.width * 10 * x_rand[i + self.n_trees])][
                int(self.height * 10 * y_rand[i + self.n_trees])
            ] = 2

        for i in range(self.n_table):
            if (
                abs(
                    -self.width / 2
                    + self.width * x_rand[i + self.n_trees + self.n_rocks]
                )
                < 0.3
                and abs(
                    -self.height / 2
                    + self.height * y_rand[i + self.n_trees + self.n_rocks]
                )
                < 0.3
            ):
                self.x_pos.append(self.width / 2 - 0.05)
                self.y_pos.append(self.height / 2 - 0.05)
                self.map[int(self.width * 5)][int(self.height * 5)] = 3
            else:
                self.x_pos.append(
                    -self.width / 2
                    + self.width * x_rand[i + self.n_trees + self.n_rocks]
                )
                self.y_pos.append(
                    -self.height / 2
                    + self.height * y_rand[i + self.n_trees + self.n_rocks]
                )
                self.map[
                    int(self.width * 10 * x_rand[i + self.n_trees + self.n_rocks])
                ][int(self.height * 10 * y_rand[i + self.n_trees + self.n_rocks])] = 3

        self.inventory = dict(
            [("wood", self.starting_trees), ("stone", self.starting_rocks), ("pogo", 0)]
        )
        self.x_low = [i - 0.15 for i in self.x_pos]
        self.x_high = [i + 0.15 for i in self.x_pos]
        self.y_low = [i - 0.15 for i in self.y_pos]
        self.y_high = [i + 0.15 for i in self.y_pos]

        obs = self.get_observation()
        self.x_pos_copy = copy.deepcopy(self.x_pos)
        self.y_pos_copy = copy.deepcopy(self.y_pos)
        print("X Pos: ", self.x_pos)
        print("Y Pos: ", self.y_pos)

        return obs

    def qr_reading_callback(self):
        rospy.wait_for_service("read_environment")
        try:
            read_env_func = rospy.ServiceProxy("read_environment", ReadEnvironment)
            resp1 = read_env_func()
            return resp1.reading
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def step(self, action):

        basePos = copy.deepcopy(self.agent_loc)
        baseOrn = copy.deepcopy(self.agent_orn)

        reward = self.reward_step
        done = False

        forward = 0
        object_removed = 0
        index_removed = 0

        self.map[int((self.agent_loc[0] + self.width / 2) * 10)][
            int((self.agent_loc[1] + self.height / 2) * 10)
        ] = 0

        if action == 0:  # Turn right
            baseOrn -= 20 * np.pi / 180

        elif action == 1:  # Turn left
            baseOrn += 20 * np.pi / 180

        elif action == 2:  # Move forward
            x_new = basePos[0] + 0.25 * np.cos(baseOrn)
            y_new = basePos[1] + 0.25 * np.sin(baseOrn)
            forward = 1
            for i in range(self.n_trees + self.n_rocks + self.n_table):
                if abs(self.x_pos[i] - x_new) < 0.15:
                    if abs(self.y_pos[i] - y_new) < 0.15:
                        forward = 0

            if (abs(abs(x_new) - abs(self.width / 2)) < 0.25) or (
                abs(abs(y_new) - abs(self.height / 2)) < 0.25
            ):
                reward = self.reward_hit_wall
                forward = 0

            if forward == 1:
                basePos[0] = x_new
                basePos[1] = y_new

        elif action == 3:  # Break
            x = basePos[0]
            y = basePos[1]
            index_removed = self.qr_reading_callback()
            if index_removed >= 0 and index_removed < 4:
                object_removed = 1
                print("Index Removed: ", index_removed)
                time.sleep(5.0)
                self.inventory["wood"] += 1
                if self.inventory["wood"] <= 2:
                    reward = self.reward_break
            if index_removed > 3 and index_removed < 6:
                object_removed = 2
                self.rocks_broken.append(index_removed)
                print("Index Removed: ", index_removed)
                time.sleep(5.0)
                self.inventory["stone"] += 1
                if self.inventory["stone"] <= 1:
                    reward = self.reward_break

            if object_removed == 1:
                flag = 0
                for i in range(len(self.trees_broken)):
                    if index_removed > self.trees_broken[i]:
                        flag += 1
                self.x_pos.pop(index_removed - flag)
                self.y_pos.pop(index_removed - flag)
                self.x_low.pop(index_removed - flag)
                self.x_high.pop(index_removed - flag)
                self.y_low.pop(index_removed - flag)
                self.y_high.pop(index_removed - flag)
                self.n_trees -= 1
                self.trees_broken.append(index_removed)
                print("Object Broken:", object_removed)
                print("Index Broken:", index_removed)

            if object_removed == 2:
                flag = 0
                for i in range(len(self.rocks_broken)):
                    if index_removed > self.rocks_broken[i]:
                        flag += 1
                flag += len(self.trees_broken)
                self.x_pos.pop(index_removed - flag)
                self.y_pos.pop(index_removed - flag)
                self.x_low.pop(index_removed - flag)
                self.x_high.pop(index_removed - flag)
                self.y_low.pop(index_removed - flag)
                self.y_high.pop(index_removed - flag)
                self.n_rocks -= 1
                print("Object Broken:", object_removed)
                print("Index Broken:", index_removed)

        elif action == 4:  # Craft
            x = basePos[0]
            y = basePos[1]
            index_removed = self.qr_reading_callback()
            if index_removed == 7:
                if self.inventory["wood"] >= 2 and self.inventory["stone"] >= 1:
                    self.inventory["pogo"] += 1
                    self.inventory["wood"] -= 2
                    self.inventory["stone"] -= 1
                    done = True
                    reward = self.reward_done

        self.agent_loc = basePos
        self.agent_orn = baseOrn

        if self.goal_env == 0:
            x = basePos[0]
            y = basePos[1]
            for i in range(self.n_trees_org + self.n_rocks_org + self.n_table):
                if abs(self.x_pos_copy[i] - x) < 0.3:
                    if abs(self.y_pos_copy[i] - y) < 0.3:
                        reward = self.reward_done
                        done = True

        elif self.goal_env == 1:
            if (
                self.inventory["wood"] >= self.n_trees_org + self.starting_trees
                or self.inventory["wood"] >= 2
            ) and (
                self.inventory["stone"] >= self.n_rocks_org + self.starting_rocks
                or self.inventory["stone"] >= 1
            ):
                reward = self.reward_done
                done = True
                print("Inventory: ", self.inventory)

        self.env_step_counter += 1

        obs = self.get_observation()
        self.map[int((self.agent_loc[0] + self.width / 2) * 10)][
            int((self.agent_loc[1] + self.height / 2) * 10)
        ] = 567
        return obs, reward, done, {}

    def get_observation(self):

        num_obj_types = len(self.object_types)

        basePos = copy.deepcopy(self.agent_loc)
        baseOrn = copy.deepcopy(self.agent_orn)

        base = baseOrn
        rot_degree = base * 57.2958
        current_angle_deg = rot_degree
        current_angle = base
        lidar_readings = []
        index_temp = 0
        angle_temp = 0

        while True:

            beam_i = np.zeros(num_obj_types)

            for r in np.arange(0, self.sense_range, 0.1):

                flag = 0
                x = basePos[0] + r * np.cos(np.deg2rad(current_angle_deg))
                y = basePos[1] + r * np.sin(np.deg2rad(current_angle_deg))

                for i in range(self.n_trees + self.n_rocks + self.n_table):
                    if x > self.x_low[i] and x < self.x_high[i]:
                        if y > self.y_low[i] and y < self.y_high[i]:
                            flag = 1
                            sensor_value = float(self.sense_range - r) / float(
                                self.sense_range
                            )
                            if i < self.n_trees:
                                obj_type = 1  # Update object as tree

                            elif (
                                i > self.n_trees - 1 and i < self.n_trees + self.n_rocks
                            ):
                                obj_type = 2  # Update object as rocks

                            else:
                                obj_type = 3  # Update object as table

                            index_temp += 1
                            beam_i[obj_type] = sensor_value

                            break

                if flag == 1:
                    break

                if (
                    abs(self.width / 2) - abs(x) < 0.05
                    or abs(self.height / 2) - abs(y) < 0.05
                ):
                    sensor_value = float(self.sense_range - r) / float(self.sense_range)
                    index_temp += 1
                    beam_i[0] = sensor_value
                    break

            for k in range(0, len(beam_i)):
                lidar_readings.append(beam_i[k])

            current_angle += self.angle_increment
            angle_temp += 1

            # Commented for degree
            # if current_angle >= 2*np.pi + base:
            # 	break

            current_angle_deg += self.angle_increment_deg

            if current_angle_deg >= 343 + rot_degree:
                break

        while len(lidar_readings) < self.half_beams * 2 * num_obj_types:
            print("lidar readings appended")
            lidar_readings.append(0)

        while len(lidar_readings) > self.half_beams * 2 * num_obj_types:
            print("lidar readings popped")
            lidar_readings.pop()

        lidar_readings.append(self.inventory["wood"])
        lidar_readings.append(self.inventory["stone"])
        lidar_readings.append(self.inventory["pogo"])

        observations = np.asarray(lidar_readings)

        return observations

    def close(self):
        return
