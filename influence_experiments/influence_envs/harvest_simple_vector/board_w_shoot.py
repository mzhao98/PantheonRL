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
        self.SHOOT = 'SHOOT'
        self.CONSUME = 'CONSUME'
        # self.PLANT = 'CONSUME'


class Board():
    def __init__(self):
        self.NORTH = (-1, 0)
        self.SOUTH = (1, 0)
        self.EAST = (0, 1)
        self.WEST = (0, -1)
        self.SHOOT = 'SHOOT'
        self.CONSUME = 'CONSUME'

        # empty -- 0
        # player 0 -- 1
        # player 1 -- 2
        # apple -- 3

        self.action_idx_to_coord = {0: self.NORTH, 1: self.SOUTH, 2:self.EAST, 3:self.WEST, 4:self.CONSUME, 5:self.SHOOT}

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


        self.previous_actions = [None, None]

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
        self.number_of_apples_respawned = 0
        self.player_rewards = {i: 0 for i in range(self.num_players)}

        self.num_apples_remaining = len(self.apple_locations)
        self.previous_actions = [None, None]
        self.apples_consumed = [0, 0]

    def set_previous_actions(self, previous_actions):
        self.previous_actions[0] = previous_actions['player_1']
        self.previous_actions[1] = previous_actions['player_2']

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
        # if self.num_apples_remaining < 7:
        #     reward = -1
        if check_apple_obj[0] >= 0 and check_apple_obj[0] < self.grid_w:
            if check_apple_obj[1] >= 0 and check_apple_obj[1] < self.grid_l:
                if self.grid[check_apple_obj] == 3:
                    self.grid[check_apple_obj] = 0
                    assert self.grid[check_apple_obj] != 3 and self.grid[check_apple_obj] == 0
                    self.apple_locations.remove(check_apple_obj)
                    self.num_apples_remaining -= 1
                    reward = 1
                    # if self.num_apples_remaining < 6 and player_index == 0:
                    #     reward = -1
                    self.apples_consumed[player_index] += 1


        # self.player_rewards[player_index] += reward

        return reward

    def perform_shoot(self, player_index):
        current_player_pos = self.player_positions[player_index]
        current_player_or = self.player_orientations[player_index]

        self.shoot_range = 3

        ego_reward = -1
        partner_reward = 0
        for i in range(self.shoot_range):
            check_shoot_loc = (current_player_pos[0] + i * current_player_or[0], current_player_pos[1] + i * current_player_or[1])

            if 0 <= check_shoot_loc[0] < self.grid_w:
                if 0 <= check_shoot_loc[1] < self.grid_l:
                    if check_shoot_loc == self.player_positions[1-player_index]:
                        partner_reward = -50


        # self.player_rewards[player_index] += ego_reward
        # self.player_rewards[1-player_index] += partner_reward


        return ego_reward, partner_reward

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
        p_r = 0
        if action_coord == self.CONSUME:
            # if self.num_apples_remaining > 10:


            r = self.perform_consume(player_index=p_idx)
            # print(
            #     f"player {p_idx} takes action {action}: # apples_left = {self.num_apples_remaining}, "
            #     f"respawned = {self.number_of_apples_respawned}, consumed = {self.apples_consumed}: rew={r}")
        elif action_coord == self.SHOOT:
            r, p_r = self.perform_shoot(player_index=p_idx)
        else:
            r = self.perform_direction_move(player_index=p_idx, action=action_coord)
        return r, p_r


    def check_successful(self, p_idx, action):
        action_coord = self.action_idx_to_coord[action]

        if action_coord == self.CONSUME:

            can = self.is_valid_consume(player_index=p_idx)

        elif action_coord == self.SHOOT:
            can = True
        else:
            can = self.can_step(player_index=p_idx, action=action_coord)

        success = 0
        if can:
            success = 1
        return success



    def get_final_team_reward(self):
        return self.player_rewards[0] + self.player_rewards[1]



    def observation_as_vector(self, player_idx):
        # obs = spaces.Box(low=0, high=1, shape=(27, 1), dtype=np.int8)
        ego_idx = player_idx
        partner_idx = 1-player_idx
        num_actions = len(self.action_idx_to_coord)


        observation_list = []

        # Get last partner actions
        partner_last_action = [0]*num_actions
        if self.previous_actions[partner_idx] is not None:
            partner_last_action[self.previous_actions[partner_idx]] = 1
        observation_list.extend(partner_last_action)

        # print("observation_list", observation_list)

        # Get last ego action
        ego_last_action = [0] * num_actions
        if self.previous_actions[partner_idx] is not None:
            ego_last_action[self.previous_actions[ego_idx]] = 1
        observation_list.extend(ego_last_action)

        # print("observation_list", observation_list)

        # get ego position
        ego_pos = self.player_positions[ego_idx]
        observation_list.append(ego_pos[0])
        observation_list.append(ego_pos[1])

        # print("observation_list", observation_list)

        # get partner position
        partner_pos = self.player_positions[partner_idx]
        observation_list.append(partner_pos[0])
        observation_list.append(partner_pos[1])

        # print("observation_list", observation_list)

        # number of apples remaining
        num_apples_left = self.num_apples_remaining
        observation_list.append(num_apples_left)

        # print("num_apples_left", num_apples_left)
        # print("self.apple_locations", self.apple_locations)

        # five closest apple locations
        closest_apple_locs = []
        check_apple_positions = []
        for i in range(len(self.apple_locations)):
            apple_loc = self.apple_locations[i]
            distance_to_player = np.sqrt((ego_pos[0]-apple_loc[0])**2 + (ego_pos[1]-apple_loc[1])**2)
            check_apple_positions.append((i, distance_to_player))

        check_apple_positions = sorted(check_apple_positions, key=lambda x: x[1])
        for i in range(5):

            if i >= len(check_apple_positions):
                closest_apple_locs.extend([0, 0])
            else:
                apple_idx = check_apple_positions[i][0]
                closest_apple_locs.extend([self.apple_locations[apple_idx][0], self.apple_locations[apple_idx][1]])

        observation_list.extend(closest_apple_locs)
        observation = np.array(observation_list).astype(np.int8)
        observation = np.expand_dims(observation, axis=1)
        # print("observation_list", observation_list)
        # print("observation", observation.shape)

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


    def respawn_apples_old(self):
        spawn_prob = 0
        if 1 <= self.num_apples_remaining <= 5:
            spawn_prob = 0.005

        elif self.num_apples_remaining > 5:
            spawn_prob = 0.001

        apple_loc_surroundings = []
        for apple_loc in self.apple_locations:
            for i in [-2,0,2]:
                for j in [-2,0,2]:
                    new_apple_loc = (apple_loc[0]+i, apple_loc[1]+j)
                    if 0 <= new_apple_loc[0] < self.grid_w:
                        if 0 <= new_apple_loc[1] < self.grid_l:
                            apple_loc_surroundings.append(new_apple_loc)


        if spawn_prob > 0:
            for new_apple_loc in apple_loc_surroundings:
                rand = np.random.uniform(0, 1)
                if rand < spawn_prob:

                    if self.grid[new_apple_loc] == 0:
                        self.grid[new_apple_loc] = 3
                        self.apple_locations.append(new_apple_loc)
                        self.number_of_apples_respawned += 1



        return

    def respawn_apples(self):
        spawn_prob = 0
        if self.num_apples_remaining < 5:
            spawn_prob = 0

        elif self.num_apples_remaining == 5:
            spawn_prob = 0.5

        elif self.num_apples_remaining > 5:
            spawn_prob = 1

        if spawn_prob > 0 and self.current_timestep % 5 == 0:
            loc_index = np.random.choice(len(self.index_to_position), replace=False)
            new_apple_loc = self.index_to_position[loc_index]

            rand = np.random.uniform(0, 1)
            if rand < spawn_prob:
                if self.grid[new_apple_loc] == 0:
                    self.grid[new_apple_loc] = 3
                    self.apple_locations.append(new_apple_loc)
                    self.number_of_apples_respawned += 1
                    self.num_apples_remaining += 1
                    # print(f"RESPAWN: number_of_apples_respawned = {self.number_of_apples_respawned}, num consumed = {self.apples_consumed}")


        return












