import numpy as np
from gym import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import copy
from .board import Board

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
        "name": "harvest_v2",
        "is_parallelizable": True,
        "render_fps": 1,
    }

    def __init__(self, max_cycles=400):
        super().__init__()

        self.max_cycles = max_cycles
        # Set grid
        self.board = Board()


        self.grid_shape = self.board.grid_shape



        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]
        self.num_actions = 7
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
        # self.observation_spaces = {i: spaces.Box(low=0, high=1, shape=(25, 1), dtype=np.int8) for i in self.agents}
        # self.observation_spaces = {i: spaces.Box(low=0, high=255, shape=(self.grid_shape[0], self.grid_shape[1], 3), dtype=np.uint8) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Box(low=0, high=1, shape=(3, self.grid_shape[0], self.grid_shape[1]), dtype=np.int8) for i in
            self.agents}


        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}


        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._none = self.num_actions
        self.selected_actions = {agent: self._none for agent in self.agents}
        self.previous_selected_actions = {agent: self._none for agent in self.agents}
        self.num_moves = 0
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        self.infos = {i: {"legal_moves": list(range(0, self.num_actions)),
                          "partner_action": self._none} for i in self.agents}

        self.observations = {}
        for i in self.agents:
            self.observations[i] = self.observe(i)

    def observe(self, agent):
        observation = self.board.observation_as_stacked_array(self.agents.index(self.agent_selection))
        # print("previous_selected_actions", self.previous_selected_actions)

        return observation

    def get_partner_action(self, agent):
        partner_action = self.previous_selected_actions[self.agents[1 - self.agents.index(self.agent_selection)]]
        return partner_action

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]



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
            rewards = [0, 0]

            # If can move
            # rew = self.board.step_single_action(self.agents.index(agent), action)
            # rewards[self.agents.index(agent)] = rew
            for acting_agent in self.agents:
                selected_action = self.selected_actions[acting_agent]
                rew = self.board.step_single_action(self.agents.index(acting_agent), selected_action)
                rewards[self.agents.index(acting_agent)] = rew

            # Increment timestep
            self.board.current_timestep += 1

            # Respawn apples - Apples dont respawn, need to be planted
            # self.board.respawn_apples()


            # If last timestep, add team reward
            if self.board.current_timestep > self.board.max_timesteps:
                team_rew = self.board.get_final_team_reward()
                rewards[0] = rewards[0] + team_rew
                rewards[1] = rewards[1] + team_rew


            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = tuple(rewards)

            self.num_moves += 1
            # observe the current state
            for i in self.agents:
                self.observations[i] = self.observe(i)

            self.dones = {
                agent: self.board.is_done() for agent in self.agents
            }
            for agent in self.agents:
                self.infos[agent]['partner_action'] = self.selected_actions


        else:
            self.selected_actions[self.agents[1 - self.agent_name_mapping[agent]]] = self._none

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def reset(self, seed=None, return_info=False, options=None):
        # reset environment
        self.board.random_reset()

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