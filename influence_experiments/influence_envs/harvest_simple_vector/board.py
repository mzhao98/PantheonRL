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
        # self.PLANT = 'CONSUME'


class Board():
    def __init__(self):
        self.NORTH = (-1, 0)
        self.SOUTH = (1, 0)
        self.EAST = (0, 1)
        self.WEST = (0, -1)
        self.STAY = (0, 0)
        self.CONSUME = 'CONSUME'

        # empty -- 0
        # player 0 -- 1
        # player 1 -- 2
        # apple -- 3

        self.action_idx_to_coord = {0: self.NORTH, 1: self.SOUTH, 2:self.EAST, 3:self.WEST, 4:self.CONSUME, 5:self.STAY}

        self.random_seed = 0
        self.grid_w, self.grid_l = 10, 10
        self.grid_shape = (self.grid_w, self.grid_l)
        self.empty_grid = np.zeros((self.grid_w, self.grid_l))
        self.index_to_position = dict(enumerate([(i,j) for i in range(self.empty_grid.shape[0]) for j in range(self.empty_grid.shape[1])]))

        self.start_num_apples = 5
        self.num_apples_remaining = 5



        self.apple_locations = []

        self.num_players = 2
        self.player_positions = {i: None for i in range(self.num_players)}
        self.player_orientations = {i: self.NORTH for i in range(self.num_players)}

        self.current_timestep = 0
        self.max_timesteps = 400
        self.player_rewards = {i:0 for i in range(self.num_players)}

        self.random_reset(self.random_seed)

    def random_reset(self, seed=0):
        np.random.seed(seed)

        self.grid = np.zeros((self.grid_w, self.grid_l))
        loc_indices = np.random.choice(len(self.index_to_position), replace=False, size=(2+self.start_num_apples))
        self.apple_locations = [self.index_to_position[p] for p in loc_indices[2:]]
        self.player_positions[0] = self.index_to_position[loc_indices[0]]
        self.player_positions[1] = self.index_to_position[loc_indices[1]]

        self.grid[self.player_positions[0]] = 1
        self.grid[self.player_positions[1]] = 2
        for apple_loc in self.apple_locations:
            self.grid[apple_loc] = 3

        self.player_orientations = {i: self.NORTH for i in range(self.num_players)} # reinit orientations to north
        self.current_timestep = 0
        self.player_rewards = {i: 0 for i in range(self.num_players)}

        self.num_apples_remaining = len(self.apple_locations)

    def can_step(self, player_index, action):
        current_player_pos = self.player_positions[player_index]

        check_next_move = (current_player_pos[0] + action[0], current_player_pos[1] + action[1])

        can_move = False
        if check_next_move[0] >= 0 and check_next_move[0] < self.grid_w:
            if check_next_move[1] >= 0 and check_next_move[1] < self.grid_l:
                if self.grid[check_next_move] == 0:
                    can_move = True

        return can_move

    def is_valid_consume(self, player_index):
        current_player_pos = self.player_positions[player_index]
        current_player_or = self.player_orientations[player_index]

        check_apple_obj = (current_player_pos[0] + current_player_or[0], current_player_pos[1] + current_player_or[1])

        can_consume = False
        if check_apple_obj[0] >= 0 and check_apple_obj[0] < self.grid_w:
            if check_apple_obj[1] >= 0 and check_apple_obj[1] < self.grid_l:
                if self.grid[check_apple_obj] == 3:
                    can_consume = True

        return can_consume

    def perform_direction_move(self, player_index, action):

        current_player_pos = self.player_positions[player_index]

        check_next_move = (current_player_pos[0] + action[0], current_player_pos[1] + action[1])

        can_step = False
        if check_next_move[0] >= 0 and check_next_move[0] < self.grid_w:
            if check_next_move[1] >= 0 and check_next_move[1] < self.grid_l:
                if self.grid[check_next_move] == 0:
                    can_step = True

        if can_step:
            self.grid[current_player_pos] = 0
            self.grid[check_next_move] = player_index + 1
            self.player_positions[player_index] = check_next_move
            self.player_orientations[player_index] = action
        else:
            self.player_orientations[player_index] = action

        return 0

    def perform_consume(self, player_index):
        current_player_pos = self.player_positions[player_index]
        current_player_or = self.player_orientations[player_index]

        check_apple_obj = (current_player_pos[0] + current_player_or[0], current_player_pos[1] + current_player_or[1])

        can_consume = False
        reward = 0
        if check_apple_obj[0] >= 0 and check_apple_obj[0] < self.grid_w:
            if check_apple_obj[1] >= 0 and check_apple_obj[1] < self.grid_l:
                if self.grid[check_apple_obj] == 3:
                    self.grid[check_apple_obj] = 0
                    self.apple_locations.remove(check_apple_obj)
                    self.num_apples_remaining -=1
                    reward = 5
                    can_consume = True


        self.player_rewards[player_index] += reward

        return reward

    def is_done(self):
        if self.current_timestep > self.max_timesteps:
            return True
        else:
            return False

    def step_joint_action(self, joint_action):
        p1_action, p2_action = joint_action

        if p1_action == self.CONSUME:
            self.perform_consume(player_index=0)
        else:
            self.perform_direction_move(player_index=0, action=p1_action)

        if p2_action == self.CONSUME:
            self.perform_consume(player_index=1)
        else:
            self.perform_direction_move(player_index=1, action=p2_action)


    def step_single_action(self, p_idx, action):
        action_coord = self.action_idx_to_coord[action]
        if action_coord == self.CONSUME:
            r = self.perform_consume(player_index=p_idx)
        else:
            r = self.perform_direction_move(player_index=p_idx, action=action_coord)
        return r

    def get_final_team_reward(self):
        return self.player_rewards[0] + self.player_rewards[1]


    def observation_as_vector(self, player_idx):
        obs = spaces.Box(low=0, high=1, shape=(25, 1), dtype=np.int8)

        observation_list = []
        observation_list.extend(self.remaining_colors)

        partner_agent = self.agents[1 - self.agent_name_mapping[agent]]
        observation_list.extend(self.agent_past_selections[partner_agent])
        observation_list.extend(self.previous_agent_state_action[partner_agent]['state'])
        observation_list.extend(self.previous_agent_state_action[partner_agent]['action'])

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
        observation = np.array(observation_list).astype(np.int8)
        observation = np.expand_dims(observation, axis=1)

        return observation


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

        grid_with_apples = np.zeros(self.grid_shape)
        for pos in self.apple_locations:
            grid_with_apples[pos] = 1


        observation = np.stack([grid_with_ego_position, grid_with_partner_position, grid_with_apples], axis=0)
        # print("observation", observation.shape)
        return observation


    def respawn_apples(self):
        to_spawn = False
        if self.current_timestep % 10 == 0:
            if self.num_apples_remaining == 0:
                to_spawn = False
            elif 1 <= self.num_apples_remaining <= 5:
                p = 0.05
                rand = np.random.uniform(0,1)
                if rand < p:
                    to_spawn = True

            else:
                p = 0.1
                rand = np.random.uniform(0, 1)
                if rand < p:
                    to_spawn = True

            if to_spawn:
                loc_index = np.random.choice(len(self.index_to_position), replace=False)
                loc = self.index_to_position[loc_index]
                if self.grid[loc] == 0:
                    self.grid[loc] = 3
                    self.apple_locations.append(loc)

        return











