import copy

import numpy as np
from gym import spaces


# self._moves = ["NORTH", "SOUTH", "EAST", "WEST"]
#
# self._moves_as_coords = [ (-1, 0), (1, 0), (0, 1), (0, -1)]

class Action:
    def __init__(self):
        self.NORTH = (-1, 0)
        self.SOUTH = (1,0)
        self.EAST = (0,1)
        self.WEST = (0,-1)
        self.STAY = (0,0)



class Board():
    def __init__(self, with_reward_shaping=False):
        self.NORTH = (-1, 0)
        self.SOUTH = (1, 0)
        self.EAST = (0, 1)
        self.WEST = (0, -1)
        self.STAY = (0, 0)

        self.with_reward_shaping = with_reward_shaping
        # empty -- 0
        # player 0 -- 1
        # player 1 -- 2
        # apple -- 3

        self.action_idx_to_coord = {0: self.NORTH, 1: self.SOUTH, 2:self.EAST, 3:self.WEST, 4:self.STAY}

        self.random_seed = 0
        self.grid_w, self.grid_l = 10, 10
        self.grid_shape = (self.grid_w, self.grid_l)
        self.empty_grid = np.zeros((self.grid_w, self.grid_l))
        self.index_to_position = dict(enumerate([(i,j) for i in range(self.empty_grid.shape[0]) for j in range(self.empty_grid.shape[1])]))

        self.door_locations = []
        self.lock_locations = []
        self.gem_locations = []
        self.exit_locations = []

        self.num_players = 2
        self.player_positions = {i: None for i in range(self.num_players)}
        self.player_holding = {i: [] for i in range(self.num_players)}
        self.player_orientations = {i: self.NORTH for i in range(self.num_players)}

        self.current_timestep = 0
        self.max_timesteps = 600
        self.player_rewards = {i:0 for i in range(self.num_players)}

        self.reset(self.random_seed)

    def reset(self, seed=0):
        np.random.seed(seed)
        self.OPEN_FLOOR = 0
        self.WALL = 1
        self.DOOR = 5
        self.GEM = 4
        self.LOCK = 6
        self.P1 = 2
        self.P2 = 3
        self.EXIT = 7

        self.grid = np.array([[1, 1,1,1,1,1,1,1,1,1],
                            [1, 7,0,0,6,0,0,0,1,1],
                            [1, 0,0,0,0,0,0,1,4,1],
                            [1, 0,0,0,0,0,0,1,5,1],
                            [1, 0,0,0,0,0,0,0,0,1],
                            [1, 0,0,0,0,0,0,0,0,1],
                            [1, 5,1,0,0,0,0,0,0,1],
                            [1, 4,1,0,0,0,0,0,0,1],
                            [1, 1,0,0,0,0,6,0,7,1],
                            [1, 1,1,1,1,1,1,1,1,1]
                           ])

        self.door_locations = []
        self.lock_locations = []
        self.gem_locations = []
        self.exit_locations = []

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] == self.DOOR:
                    self.door_locations.append((i,j))
                elif self.grid[i][j] == self.LOCK:
                    self.lock_locations.append((i, j))
                elif self.grid[i][j] == self.GEM:
                    self.gem_locations.append((i, j))
                elif self.grid[i][j] == self.EXIT:
                    self.exit_locations.append((i, j))

        self.player_positions[0] = (2,3)
        self.player_positions[1] = (7,5)

        self.player_orientations = {i: self.NORTH for i in range(self.num_players)} # reinit orientations to north
        self.current_timestep = 0
        self.player_rewards = {i: 0 for i in range(self.num_players)}

        self.gems_remaining = copy.deepcopy(self.gem_locations)
        self.player_holding = {i: [] for i in range(self.num_players)}
        self.doors_open = False
        self.players_exited = False

        self.previous_actions = [self.STAY, self.STAY]


    def check_collision_with_doors(self, check_next_pos):
        if check_next_pos in self.door_locations:
            return True
        return False

    def check_collision_with_locks(self, check_next_pos):
        if check_next_pos in self.lock_locations:
            return True
        return False

    def check_collision_with_gems(self, check_next_pos):
        if check_next_pos in self.gem_locations:
            return True
        return False

    def check_collision_with_remaining_gems(self, check_next_pos):
        if check_next_pos in self.gems_remaining:
            return True
        return False

    def check_collision_with_exits(self, check_next_pos):
        if check_next_pos in self.exit_locations:
            return True
        return False

    def can_exit_game(self):
        if sum([len(self.player_holding[p_idx]) for p_idx in self.player_holding]) == 2:
            return True
        else:
            return False

    def l2_distance(self, loc1, loc2):
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

    def perform_direction_move(self, player_index, action):

        current_player_pos = self.player_positions[player_index]
        check_next_pos = (current_player_pos[0] + action[0], current_player_pos[1] + action[1])

        reward = -0.01
        if check_next_pos[0] >= 0 and check_next_pos[0] < self.grid_w:
            if check_next_pos[1] >= 0 and check_next_pos[1] < self.grid_l:
                if self.grid[check_next_pos] != self.WALL and check_next_pos != self.player_positions[1-player_index]:

                    if self.check_collision_with_exits(check_next_pos) is True:
                        self.player_positions[player_index] = check_next_pos

                    elif self.check_collision_with_gems(check_next_pos) is True:
                        self.player_positions[player_index] = check_next_pos
                        if check_next_pos in self.gems_remaining:
                            self.gems_remaining.remove(check_next_pos)
                            self.player_holding[player_index].append(check_next_pos)
                            if self.with_reward_shaping:
                                reward += 2

                        else:
                            if self.with_reward_shaping:
                                reward -= 1


                    elif self.check_collision_with_doors(check_next_pos) is True:
                        if self.doors_open is True:
                            self.player_positions[player_index] = check_next_pos


                    elif self.check_collision_with_locks(check_next_pos) is True:
                        distances = [self.l2_distance(self.player_positions[1-player_index],
                                                                           gem_loc) for gem_loc in self.gems_remaining]
                        if self.with_reward_shaping:
                            if len(distances) > 0:
                                min_dist_of_partner_to_gem = min(distances)
                                if self.doors_open is False:
                                    if min_dist_of_partner_to_gem < 2:

                                        reward += 0.01

                            else:
                                if self.check_collision_with_gems(self.player_positions[1-player_index]) is True:
                                    reward += 0.01


                        self.doors_open = True
                        self.player_positions[player_index] = check_next_pos

                    elif self.grid[check_next_pos] == self.OPEN_FLOOR:

                        if self.check_collision_with_locks(current_player_pos) is True:
                            self.doors_open = False

                        self.player_positions[player_index] = check_next_pos


        return reward



    def is_done(self):
        if self.current_timestep > self.max_timesteps or self.players_exited is True:
            return True
        else:
            return False

    def step_joint_action(self, joint_action):
        # print("joint_action = ", joint_action)
        p1_action, p2_action = joint_action
        p1_action = self.action_idx_to_coord[p1_action]
        p2_action = self.action_idx_to_coord[p2_action]
        collective_rew = 0

        # Check exit game
        p1_current_pos = self.player_positions[0]
        check_p1_next_pos = (p1_current_pos[0] + p1_action[0], p1_current_pos[1] + p1_action[1])
        p2_current_pos = self.player_positions[1]
        check_p2_next_pos = (p2_current_pos[0] + p2_action[0], p2_current_pos[1] + p2_action[1])

        if check_p1_next_pos != check_p2_next_pos:
            if self.check_collision_with_exits(check_p1_next_pos) is True and self.check_collision_with_exits(check_p2_next_pos) is True:
                if self.can_exit_game() is True:
                    collective_rew += 20
                    self.players_exited = True

            move_rew_1 = self.perform_direction_move(player_index=0, action=p1_action)
            move_rew_2 = self.perform_direction_move(player_index=1, action=p2_action)
            collective_rew += (move_rew_1 + move_rew_2)

        return collective_rew

    def set_previous_actions(self, previous_actions):

        self.previous_actions[0] = self.action_idx_to_coord[previous_actions['player_1']]
        self.previous_actions[1] = self.action_idx_to_coord[previous_actions['player_2']]

    def get_final_team_reward(self):
        return self.player_rewards[0] + self.player_rewards[1]


    def observation_as_vector(self, player_idx):
        # obs = spaces.Box(low=0, high=1, shape=(25, 1), dtype=np.int8)
        # print("player_idx = ", player_idx)

        ego_idx = player_idx
        partner_idx = 1 - player_idx
        num_actions = len(self.action_idx_to_coord)

        ego_prev_action = list(self.previous_actions[ego_idx])
        partner_prev_action = list(self.previous_actions[partner_idx])

        observation_list = []

        # get ego position
        ego_pos = self.player_positions[ego_idx]
        observation_list.append(ego_pos[0])
        observation_list.append(ego_pos[1])

        ego_or = self.player_orientations[ego_idx]
        observation_list.append(ego_or[0])
        observation_list.append(ego_or[1])

        # print("observation_list", observation_list)

        # get partner position
        partner_pos = self.player_positions[partner_idx]
        observation_list.append(partner_pos[0])
        observation_list.append(partner_pos[1])

        partner_or = self.player_orientations[partner_idx]
        observation_list.append(partner_or[0])
        observation_list.append(partner_or[1])

        # print("observation_list", observation_list)
        for list_objs in [self.door_locations, self.lock_locations, self.exit_locations]:
            for obj in list_objs:
                observation_list.extend(list(obj))

        for obj in self.gem_locations:
            if obj in self.gems_remaining:
                observation_list.extend(list(obj))
            else:
                observation_list.extend([-1, -1])

        doors_open_bool = 0
        if self.doors_open is True:
            doors_open_bool = 1

        observation_list.append(doors_open_bool)
        observation_list.extend(ego_prev_action)
        observation_list.extend(partner_prev_action)
            
        observation = np.array(observation_list).astype(np.int8)
        observation = np.expand_dims(observation, axis=1)

        # print("self.door_locations, ", len(self.door_locations))
        # print("self.lock_locations, ", len(self.lock_locations))
        # print("self.exit_locations, ", len(self.exit_locations))
        # print("self.gem_locations, ", len(self.gem_locations))
        # print("self.gems_remaining, ", len(self.gems_remaining))
        #
        # print("observation_list = ", len(observation_list))
        # print("observation_list", observation_list)
        # print("observation", observation.shape)

        return observation




