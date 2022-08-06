import functools
from gym.spaces import Discrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import numpy as np
import os


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class joint_particle_env(AECEnv):
    """Two-player environment for repeated prisoners dilemma
    The observation is simply the last opponent action."""

    metadata = {
        "render_modes": ["human"],
        "name": "joint_particle_v0",
        "is_parallelizable": True,
        "render_fps": 2,
    }

    def __init__(self, num_actions=4, max_cycles=15):
        self.max_cycles = max_cycles

        # number of actions must be odd and greater than 3
        assert num_actions > 1, "The number of actions must be equal or greater than 3."
        # assert num_actions % 2 == 0, "The number of actions must be an odd number."
        self._moves = ["NORTH", "SOUTH", "EAST", "WEST"]
        self._moves_as_coords = [(0,1), (0,-1), (1,0), (-1,0)]

        # Optionally add actions

        # none is last possible action, to satisfy discrete action space
        self._moves.append("None")
        self._none = num_actions

        self.agents = ["player_" + str(r) for r in range(2)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.action_spaces = {agent: Discrete(num_actions) for agent in self.agents}
        # self.observation_spaces = {
        #     agent: Discrete(8*5) for agent in self.agents
        # }
        self.observation_spaces = {
            agent: Discrete(1 + num_actions) for agent in self.agents
        }

        self.grid_txt = "000" \
                        "0X0" \
                        "000"

        self.initial_grid = np.array([[0,0,0], [0,1,0], [2,0,0]])
        self.grid = np.array([[0, 0, 0], [0, 1, 0], [2, 0, 0]])

        self.coords = {(0,0):0,
                           (1,0):1,
                           (2,0):2,
                           (0,1):3,
                           (2,1):4,
                           (0,2):5,
                           (1,2):6,
                           (2,2):7,}

        self.action_coord_pairs = []
        for action in range(5):
            for coord in self.coords:
                keyname = (action, coord[0], coord[1])
                self.action_coord_pairs.append(keyname)

        self.num_to_action_coord_pairs  = dict(enumerate(self.action_coord_pairs))
        self.action_coord_pairs_to_num = {v: k for k, v in self.num_to_action_coord_pairs.items()}


        self.screen = None
        self.history = [0] * (2 * 5)

        self.reinit()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reinit(self):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.current_position = (2, 0)
        self.start_position = (2, 0)
        self.destination = (0, 2)
        self.trap = (1, 1)

        self.state = {agent: self.initial_grid for agent in self.agents}
        self.actions_for_state = {agent: self._none for agent in self.agents}
        self.observations = {agent: self._none for agent in self.agents}
        # self.observations = {agent: self.action_coord_pairs_to_num[(4, self.current_position[0],self.current_position[1])] for agent in self.agents}
        self.grid = self.initial_grid

        # print("action_coord_pairs_to_num", self.action_coord_pairs_to_num)




        self.num_moves = 0

    def render(self, mode="human"):
        """
                Renders the environment. In human mode, it can print to terminal, open
                up a graphical window, or open up some other display that a human can see and understand.
                """
        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
            )
        else:
            string = "Game over"
        print(string)

    def observe(self, agent):
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        pass

    def reset(self, seed=None, return_info=False, options=None):
        self.reinit()

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection

        self.actions_for_state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():

            # Check in agents can move
            can_move = False
            if self.actions_for_state[self.agents[0]] == self.actions_for_state[self.agents[1]]:
                can_move = True

            # If cannot move
            rewards = (0, 0)
            if can_move is False:
                rewards = (-1, -1)

            # If can move
            if can_move is True:
                move = self._moves_as_coords[self.actions_for_state[self.agents[0]]]
                new_x = self.current_position[0] + move[0]
                new_y = self.current_position[1] + move[1]

                if new_x >= 0 and new_x < 3:
                    if new_y >= 0 and new_y < 3:
                        self.grid[self.current_position] = 0
                        self.current_position = (new_x, new_y)
                        self.grid[self.current_position] = 2

            # If moved to trap, start over
            if self.current_position == self.trap:
                rewards = (-100,-100)

                self.grid[self.current_position] = 0
                self.current_position = self.start_position
                self.grid[self.current_position] = 2

            # If moved to destination, get reward, start over
            if self.current_position == self.destination:
                rewards = (100, 100)

                self.grid[self.current_position] = 0
                self.current_position = self.start_position
                self.grid[self.current_position] = 2


            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = rewards

            self.num_moves += 1

            self.dones = {
                agent: self.num_moves >= self.max_cycles for agent in self.agents
            }

        #     # observe the current state
        #     for i in self.agents:
        #         self.observations[i] = self.action_coord_pairs_to_num[(self.actions_for_state[self.agents[1 - self.agent_name_mapping[agent]]], self.current_position[0],self.current_position[1])]
        # else:
        #     self.state[self.agents[1 - self.agent_name_mapping[agent]]] = self.action_coord_pairs_to_num[(4, self.current_position[0],self.current_position[1])]
        #
        #     self._clear_rewards()
            # observe the current state
            for i in self.agents:
                self.observations[i] = self.actions_for_state[self.agents[1 - self.agent_name_mapping[agent]]]
        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = self._none

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

