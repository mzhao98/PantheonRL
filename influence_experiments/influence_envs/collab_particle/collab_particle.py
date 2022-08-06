import numpy as np
from gym import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import copy
# from .board import Board

#
# def env():
#     env = raw_env()
#     env = wrappers.CaptureStdoutWrapper(env)
#     env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
#     env = wrappers.AssertOutOfBoundsWrapper(env)
#     env = wrappers.OrderEnforcingWrapper(env)
#     return env
def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "collaborative_particle_v0",
        "is_parallelizable": True,
        "render_fps": 1,
    }

    def __init__(self, max_cycles=50):
        super().__init__()

        self.max_cycles = max_cycles
        # Set grid
        self.grid_txt = "0000" \
                        "0X00" \
                        "0000"

        # self.initial_grid = np.array([[0, 0, 0, 3], [0, 1, 0, 0], [2, 0, 0, 0]])
        # self.current_grid = np.array([[0, 0, 0, 3], [0, 1, 0, 0], [2, 0, 0, 0]])

        self.initial_grid = np.array([[0, 0, 0, 0, 0, 0, 0, 3, 3],
                                      [0, 0, 0, 0, 0, 0, 0, 3, 3],
                                      [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [2, 0, 0, 0, 0, 0, 0, 0, 0]])

        # self._moves = ["NORTH", "SOUTH", "EAST", "WEST"]



        # self.initial_grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                               [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.current_grid = copy.deepcopy(self.initial_grid)

        # self.initial_grid = np.array([[2, 0, 0, 3, 3]])
        # self.current_grid = np.array([[2, 0, 0, 3, 3]])

        self.grid_shape = self.initial_grid.shape
        # self.start_position = (2, 0)
        # self.destination = (0, 3)
        # self.lava_position = (1, 1)
        # self.current_position = (2,0)
        self.start_position = None
        self.destinations = []
        self.lava_positions = []
        self.current_position = None

        for i in range(self.initial_grid.shape[0]):
            for j in range(self.initial_grid.shape[1]):
                if self.initial_grid[i,j] == 3:
                    self.destinations.append((i,j))
                if self.initial_grid[i,j] == 2:
                    self.start_position = (i,j)
                if self.initial_grid[i, j] == 1:
                    self.lava_positions.append((i,j))

        self.current_position = copy.deepcopy(self.start_position)
        self.prev_position = copy.deepcopy(self.start_position)
        # print("self.destination", self.destinations)
        # print("self.start_position", self.start_position)
        # print("lava_positions", self.lava_positions)
        # 0 = free space, 1 = lava, 2 = start, 3 = destination

        # Set moves
        self._moves = ["NORTH", "SOUTH", "EAST", "WEST"]
        # self._moves = ["EAST", "WEST", "SOUTH", "NORTH"]
        # self._moves_as_coords = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self._moves_as_coords = [ (-1, 0), (1, 0), (0, 1), (0, -1)]

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]
        self.num_actions = 4
        # self.num_agents = 2

        self.action_spaces = {i: spaces.Discrete(self.num_actions) for i in self.agents}
        # self.observation_spaces = {
        #     i: spaces.Dict(
        #         {
        #             "observation": spaces.Box(
        #                 low=0, high=1, shape=(3, self.grid_shape[0], self.grid_shape[1]), dtype=np.int8
        #             ),
        #             "action_mask": spaces.Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8),
        #         }
        #     )
        #     for i in self.agents
        # }
        self.observation_spaces = {i: spaces.Box(low=0, high=1, shape=(25, 1), dtype=np.int8) for i in self.agents}
        # self.observation_spaces = {i: spaces.Box(low=0, high=255, shape=(self.grid_shape[0], self.grid_shape[1], 3), dtype=np.uint8) for i in self.agents}
        # self.observation_spaces = {
        #     i: spaces.Box(low=0, high=1, shape=(4, self.grid_shape[0], self.grid_shape[1]), dtype=np.int8) for i in
        #     self.agents}


        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {"legal_moves": list(range(0, self.num_actions))} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._none = self.num_actions
        self.selected_actions = {agent: self._none for agent in self.agents}
        self.previous_selected_actions = {agent: self._none for agent in self.agents}
        self.num_moves = 0
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        self.observations = {}
        for i in self.agents:
            self.observations[i] = self.observe(i)

        # Key
    # ----
    # grid_with_position = 0
    # grid with lava  = 1
    # grid with destination = 2
    # An observation is list of lists, where each list represents a row
    #
    # [[0,0,2]
    #  [1,2,1]
    #  [2,1,0]]
    def observe(self, agent):
        grid_with_position = np.zeros(self.grid_shape)
        grid_with_position[self.current_position] = 1

        grid_with_prev_position = np.zeros(self.grid_shape)
        grid_with_prev_position[self.prev_position] = 1

        grid_with_lava = np.zeros(self.grid_shape)
        for pos in self.lava_positions:
            grid_with_lava[pos] = 1

        grid_with_destination = np.zeros(self.grid_shape)
        for pos in self.destinations:
            grid_with_destination[pos] = 1

        # observation = np.stack([grid_with_position, grid_with_lava, grid_with_destination], axis=2).astype(np.uint8)
        # observation = np.stack([grid_with_position, grid_with_prev_position, grid_with_lava, grid_with_destination], axis=0).astype(np.uint8)
        obs = [self.current_position[0], self.current_position[1],
                                self.prev_position[0], self.prev_position[1]]
        for pos in self.lava_positions:
            obs.append(pos[0])
            obs.append(pos[1])
        for pos in self.destinations:
            obs.append(pos[0])
            obs.append(pos[1])

        obs.append(self.previous_selected_actions[self.agents[1 - self.agent_name_mapping[agent]]])
        observation = np.array(obs).astype(np.int8)
        observation = np.expand_dims(observation, axis=1)
        # print("observation", observation.shape)

        # action_mask = np.zeros(self.num_actions, "int8")
        # for i in range(action_mask.shape[0]):
        #     action_mask[i] = 1
        # return {"observation": observation, 'action_mask': action_mask}
        return observation

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]


    # action in this case is a value from 0 to 8 indicating position to move on tictactoe board
    def old_step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        # check if input action is a valid move (0 == empty spot)
        assert self.board.squares[action] == 0, "played illegal move"
        # play turn
        self.board.play_turn(self.agents.index(self.agent_selection), action)

        # update infos
        # list of valid actions (indexes in board)
        # next_agent = self.agents[(self.agents.index(self.agent_selection) + 1) % len(self.agents)]
        next_agent = self._agent_selector.next()

        if self.board.check_game_over():
            winner = self.board.check_for_winner()

            if winner == -1:
                # tie
                pass
            elif winner == 1:
                # agent 0 won
                self.rewards[self.agents[0]] += 1
                self.rewards[self.agents[1]] -= 1
            else:
                # agent 1 won
                self.rewards[self.agents[1]] += 1
                self.rewards[self.agents[0]] -= 1

            # once either play wins or there is a draw, game over, both players are done
            self.dones = {i: True for i in self.agents}

        # Switch selection to next agents
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_agent

        self._accumulate_rewards()

    def step(self, action, display=False):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection

        self.selected_actions[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # print(f"game pos: {self.current_position}, selected actions: {self.selected_actions}, prev actions: {self.previous_selected_actions}")


            self.previous_selected_actions = copy.deepcopy(self.selected_actions)
            # # Check in agents can move
            # can_move = False
            # if self.selected_actions[self.agents[0]] == self.selected_actions[self.agents[1]]:
            #     can_move = True
            #
            # # If cannot move
            # rewards = (0, 0)
            # if can_move is False:
            #     rewards = (-1, -1)
            can_move = True
            rewards = (-1, -1)

            # If can move
            if can_move is True:


                move_p0 = self._moves_as_coords[self.selected_actions[self.agents[0]]]
                move_p1 = self._moves_as_coords[self.selected_actions[self.agents[1]]]
                move = (move_p0[0]+move_p1[0], move_p0[1]+move_p1[1])
                new_x = self.current_position[0] + move[0]
                new_y = self.current_position[1] + move[1]

                # print("move_p0", move_p0)
                # print("move_p1", move_p1)
                # print("move", move)
                # print("self.current_position", self.current_position)
                # print("new_x", new_x)
                # print("new_y", new_y)


                if new_x >= 0 and new_x < self.grid_shape[0]:
                    if new_y >= 0 and new_y < self.grid_shape[1]:
                        self.prev_position = (self.current_position[0], self.current_position[1])
                        if self.current_position in self.lava_positions:
                            self.current_grid[self.current_position] = 1
                        else:
                            self.current_grid[self.current_position] = 0
                        self.current_position = (new_x, new_y)
                        self.current_grid[self.current_position] = 2

                        if display:
                            if move != (0, 0):
                                print("can move: \n", self.current_grid)

                # print()
            # If moved to trap, start over
            if self.current_position in self.lava_positions:
                rewards = (-10, -10)

                # self.current_grid[self.current_position] = 0
                # self.current_position = self.start_position
                # self.current_grid[self.current_position] = 2


            # If moved to destination, get reward, start over
            if self.current_position in self.destinations:
                rewards = (100, 100)
                # print("reached destination!!!", (self.current_position, self.destinations))

                # self.current_grid[self.current_position] = 0
                # self.current_position = self.start_position
                # self.current_grid[self.current_position] = 2

            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = rewards

            self.num_moves += 1
            # observe the current state
            for i in self.agents:
                self.observations[i] = self.observe(i)

            # self.dones = {
            #     agent: self.num_moves >= self.max_cycles for agent in self.agents
            # }
            # print("self.current_position and self.destination", (self.current_position, self.destination))
            if self.current_position in self.destinations:
                # print("self.current_position == self.destination", (self.current_position , self.destinations))
                self.dones = {
                    agent: True for agent in self.agents
                }
            else:
                self.dones = {
                    agent: self.num_moves >= self.max_cycles for agent in self.agents
                }



        else:
            self.selected_actions[self.agents[1 - self.agent_name_mapping[agent]]] = self._none

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def reset(self, seed=None, return_info=False, options=None):
        # reset environment
        self.current_position = copy.deepcopy(self.start_position)
        self.current_grid = copy.deepcopy(self.initial_grid)
        self.prev_position = copy.deepcopy(self.start_position)
        # print("self.initial_grid", self.initial_grid)
        # print("self.current_grid", self.current_grid)
        self.selected_actions = {agent: self._none for agent in self.agents}
        self.num_moves = 0

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

    def render(self, mode="human"):
        pass

    def close(self):
        pass