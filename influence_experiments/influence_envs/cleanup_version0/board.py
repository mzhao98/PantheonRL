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
        self.CONSUME = 'CONSUME'
        self.PLANT = 'PLANT'
        # self.PLANT = 'CONSUME'


class Board():
    def __init__(self):
        self.NORTH = (-1, 0)
        self.SOUTH = (1, 0)
        self.EAST = (0, 1)
        self.WEST = (0, -1)
        self.STAY = (0, 0)
        self.PICKUP = 'PICKUP'
        self.PLACE = 'PLACE'

        # empty -- 0
        # player 0 -- 1
        # player 1 -- 2
        # apple -- 3

        self.action_idx_to_coord = {0: self.NORTH, 1: self.SOUTH, 2:self.EAST, 3:self.WEST, 4:self.PICKUP, 6: self.PLACE, 5:self.STAY}

        self.random_seed = 0
        self.grid_w, self.grid_l = 10, 10
        self.grid_shape = (self.grid_w, self.grid_l)
        self.empty_grid = np.zeros((self.grid_w, self.grid_l))
        self.index_to_position = dict(enumerate([(i,j) for i in range(self.empty_grid.shape[0]) for j in range(self.empty_grid.shape[1])]))

        # red, green, blue
        self.color_to_start_num_blocks = {'red': 2, 'green':2, 'blue':2}



        self.num_blocks_starting = sum(self.color_to_start_num_blocks.values())

        self.block_number_to_color = {3: 'red',  4:'green', 5:'blue'}
        self.sink_number_to_color = {6: 'red', 7: 'green', 8: 'blue'}
        self.color_to_block_number = {'red': 3, 'green': 4, 'blue': 5}
        self.color_to_sink_number = {'red': 6, 'green': 7, 'blue': 8}




        self.num_players = 2
        self.player_positions = {i: None for i in range(self.num_players)}
        self.player_orientations = {i: self.NORTH for i in range(self.num_players)}
        self.player_holding = {i: None for i in range(self.num_players)}

        self.current_timestep = 0
        self.max_timesteps = 400
        self.player_rewards = {i:0 for i in range(self.num_players)}

        self.random_reset(self.random_seed)

    def random_reset(self, seed=0):
        np.random.seed(seed)

        self.grid = np.zeros((self.grid_w, self.grid_l))
        loc_indices = np.random.choice(len(self.index_to_position), replace=False, size=(2+self.num_blocks_starting + 3))
        self.player_positions[0] = self.index_to_position[loc_indices[0]]
        self.player_positions[1] = self.index_to_position[loc_indices[1]]

        self.color_to_block_locations = {}


        thresh = 2
        self.red_locations = [self.index_to_position[p] for p in loc_indices[thresh:thresh+self.color_to_start_num_blocks['red']]]
        self.color_to_block_locations['red'] = self.red_locations

        thresh = thresh+self.color_to_start_num_blocks['red']
        self.green_locations = [self.index_to_position[p] for p in loc_indices[thresh:thresh + self.color_to_start_num_blocks['green']]]
        self.color_to_block_locations['green'] = self.green_locations

        thresh = thresh + self.color_to_start_num_blocks['green']
        self.blue_locations = [self.index_to_position[p] for p in loc_indices[thresh:thresh + self.color_to_start_num_blocks['blue']]]
        self.color_to_block_locations['blue'] = self.blue_locations

        thresh = thresh + self.color_to_start_num_blocks['blue']
        self.sink_locations = {}
        self.sink_locations['red'] = self.index_to_position[loc_indices[thresh]]
        self.sink_locations['green'] = self.index_to_position[loc_indices[thresh+1]]
        self.sink_locations['blue'] = self.index_to_position[loc_indices[thresh+2]]


        self.grid[self.player_positions[0]] = 1
        self.grid[self.player_positions[1]] = 2
        for red_loc in self.red_locations:
            self.grid[red_loc] = 3
        for green_loc in self.green_locations:
            self.grid[green_loc] = 4
        for blue_loc in self.blue_locations:
            self.grid[blue_loc] = 5

        self.grid[self.sink_locations['red']] = 6
        self.grid[self.sink_locations['green']] = 7
        self.grid[self.sink_locations['blue']] = 8


        self.player_orientations = {i: self.NORTH for i in range(self.num_players)} # reinit orientations to north
        self.current_timestep = 0
        self.player_rewards = {i: 0 for i in range(self.num_players)}
        self.player_holding = {i: None for i in range(self.num_players)}

        self.current_color_to_num_blocks = copy.deepcopy(self.color_to_start_num_blocks)
        self.current_num_blocks_remaining = sum(self.current_color_to_num_blocks.values())


    def can_step(self, player_index, action):
        current_player_pos = self.player_positions[player_index]

        check_next_move = (current_player_pos[0] + action[0], current_player_pos[1] + action[1])

        can_move = False
        if check_next_move[0] >= 0 and check_next_move[0] < self.grid_w:
            if check_next_move[1] >= 0 and check_next_move[1] < self.grid_l:
                if self.grid[check_next_move] == 0:
                    can_move = True

        return can_move

    def is_valid_pickup(self, player_index):
        current_player_pos = self.player_positions[player_index]
        current_player_or = self.player_orientations[player_index]

        check_block_obj = (current_player_pos[0] + current_player_or[0], current_player_pos[1] + current_player_or[1])

        can_pickup = False
        if self.player_holding[player_index] is None:
            if check_block_obj[0] >= 0 and check_block_obj[0] < self.grid_w:
                if check_block_obj[1] >= 0 and check_block_obj[1] < self.grid_l:
                    if self.grid[check_block_obj] in [3, 4, 5]:
                        can_pickup = True

        return can_pickup

    def is_valid_place(self, player_index):
        current_player_pos = self.player_positions[player_index]
        current_player_or = self.player_orientations[player_index]

        check_block_obj = (current_player_pos[0] + current_player_or[0], current_player_pos[1] + current_player_or[1])

        can_place = False
        if self.player_holding[player_index] is not None:
            if check_block_obj[0] >= 0 and check_block_obj[0] < self.grid_w:
                if check_block_obj[1] >= 0 and check_block_obj[1] < self.grid_l:
                    # if self.grid[check_block_obj] == 0:
                    #     can_place = True


                    if self.player_holding[player_index] == 3 and self.grid[check_block_obj] == 6:
                        can_place = True

                    elif self.player_holding[player_index] == 4 and self.grid[check_block_obj] == 7:
                        can_place = True

                    elif self.player_holding[player_index] == 5 and self.grid[check_block_obj] == 8:
                        can_place = True

        return can_place

    def perform_direction_move(self, player_index, action):

        current_player_pos = self.player_positions[player_index]

        check_next_move = (current_player_pos[0] + action[0], current_player_pos[1] + action[1])

        can_perform_step = self.can_step(player_index, action)

        if can_perform_step:
            self.grid[current_player_pos] = 0
            self.grid[check_next_move] = player_index + 1
            self.player_positions[player_index] = check_next_move
            self.player_orientations[player_index] = action
        else:
            self.player_orientations[player_index] = action

        step_cost = -1
        return step_cost

    def perform_pickup(self, player_index):
        current_player_pos = self.player_positions[player_index]
        current_player_or = self.player_orientations[player_index]

        check_block_obj = (current_player_pos[0] + current_player_or[0], current_player_pos[1] + current_player_or[1])

        can_pickup_block = self.is_valid_pickup(player_index)
        reward = 0
        if can_pickup_block:
            block_type = self.grid[check_block_obj]
            self.player_holding[player_index] = block_type
            self.grid[check_block_obj] = 0
            self.color_to_block_locations[self.block_number_to_color[block_type]].remove(check_block_obj)


            reward = 5


        self.player_rewards[player_index] += reward

        return reward

    def perform_place(self, player_index):
        current_player_pos = self.player_positions[player_index]
        current_player_or = self.player_orientations[player_index]

        check_placement_location = (current_player_pos[0] + current_player_or[0], current_player_pos[1] + current_player_or[1])

        reward = 0
        can_place_block = self.is_valid_place(player_index)

        if can_place_block:
            block_type = self.player_holding[player_index]
            if self.grid[check_placement_location] in [6, 7, 8]:
                self.player_holding[player_index] = None
                self.current_color_to_num_blocks[self.block_number_to_color[block_type]] -= 1
                self.current_num_blocks_remaining -= 1
                reward = 50

            # elif self.grid[check_placement_location] == 0:
            #     self.grid[check_placement_location] = block_type
            #
            #     self.color_to_block_locations[self.block_number_to_color[block_type]].append(check_placement_location)
            #     self.player_holding[player_index] = None
            #     reward = -1


        self.player_rewards[player_index] += reward

        return reward

    def is_done(self):
        if self.current_timestep > self.max_timesteps or self.current_num_blocks_remaining == 0:
            return True
        else:
            return False

    def step_joint_action(self, joint_action):
        p1_action, p2_action = joint_action

        if p1_action == self.PICKUP:
            self.perform_pickup(player_index=0)
        elif p1_action == self.PLACE:
            self.perform_place(player_index=0)
        else:
            self.perform_direction_move(player_index=0, action=p1_action)

        if p2_action == self.PICKUP:
            self.perform_pickup(player_index=1)
        elif p2_action == self.PLACE:
            self.perform_place(player_index=1)
        else:
            self.perform_direction_move(player_index=1, action=p2_action)


    def step_single_action(self, p_idx, action):
        action_coord = self.action_idx_to_coord[action]
        if action_coord == self.PICKUP:
            r = self.perform_pickup(player_index=p_idx)
        elif action_coord == self.PLACE:
            r = self.perform_place(player_index=p_idx)
        else:
            r = self.perform_direction_move(player_index=p_idx, action=action_coord)
        return r

    def get_final_team_reward(self):
        return self.player_rewards[0] + self.player_rewards[1]


    def observation_as_vector(self):
        obs = spaces.Box(low=0, high=1, shape=(25, 1), dtype=np.int8)
        # observation = np.stack([grid_with_position, grid_with_prev_position, grid_with_lava, grid_with_destination], axis=0).astype(np.uint8)
        # obs = [self.current_position[0], self.current_position[1],
        #        self.prev_position[0], self.prev_position[1]]
        # for pos in self.lava_positions:
        #     obs.append(pos[0])
        #     obs.append(pos[1])
        # for pos in self.destinations:
        #     obs.append(pos[0])
        #     obs.append(pos[1])
        #
        # obs.append(self.previous_selected_actions[self.agents[1 - self.agent_name_mapping[agent]]])
        # observation = np.array(obs).astype(np.int8)
        # observation = np.expand_dims(observation, axis=1)


    def observation_as_img(self):
        obs = spaces.Box(low=0, high=255, shape=(self.grid_shape[0], self.grid_shape[1], 3), dtype=np.uint8)
        # grid_with_position = np.zeros(self.grid_shape)
        # grid_with_position[self.current_position] = 1
        #
        # grid_with_prev_position = np.zeros(self.grid_shape)
        # grid_with_prev_position[self.prev_position] = 1
        #
        # grid_with_lava = np.zeros(self.grid_shape)
        # for pos in self.lava_positions:
        #     grid_with_lava[pos] = 1
        #
        # grid_with_destination = np.zeros(self.grid_shape)
        # for pos in self.destinations:
        #     grid_with_destination[pos] = 1

        # observation = np.stack([grid_with_position, grid_with_lava, grid_with_destination], axis=2).astype(np.uint8)
        # observation = np.stack([grid_with_position, grid_with_prev_position, grid_with_lava, grid_with_destination], axis=0).astype(np.uint8)


    def observation_as_stacked_array(self, player_idx):
        # obs = spaces.Box(low=0, high=1, shape=(3, self.grid_shape[0], self.grid_shape[1]))

        grid_with_ego_position = np.zeros(self.grid_shape)
        grid_with_ego_position[self.player_positions[player_idx]] = 1

        grid_with_partner_position = np.zeros(self.grid_shape)
        grid_with_partner_position[self.player_positions[1-player_idx]] = 1

        stacked_grids = [grid_with_ego_position, grid_with_partner_position]

        for color in ['red', 'green', 'blue']:
            grid_with_color = np.zeros(self.grid_shape)
            for pos in self.color_to_block_locations[color]:
                grid_with_color[pos] = 1
            for p_idx in range(self.num_players):
                if self.player_holding[p_idx] == self.color_to_block_number[color]:
                    grid_with_color[self.player_positions[p_idx]] = 1

            grid_with_color[self.sink_locations[color]] = 2
            stacked_grids.append(grid_with_color)


        observation = np.stack(stacked_grids, axis=0)
        # print("observation", observation.shape)
        return observation













