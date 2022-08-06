import functools
from gym import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import numpy as np
import os

COOPERATE = 0
DEFECT = 1
NONE = 2
MOVES = ["COOPERATE", "DEFECT",  "None"]
NUM_ITERS = 100
REWARD_MAP = {
    (COOPERATE, COOPERATE): (1, 1),
    (COOPERATE, DEFECT): (-1, 1),
    (DEFECT, COOPERATE): (1, -1),
    (DEFECT, DEFECT): (1, 1),
}


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv):
    """Two-player environment for repeated prisoners dilemma
    The observation is simply the last opponent action."""

    metadata = {
        "render_modes": ["human"],
        "name": "rpd_v2",
        "is_parallelizable": True,
        "render_fps": 2,
    }

    def __init__(self, num_actions=2, max_cycles=15):
        self.max_cycles = max_cycles

        # number of actions must be odd and greater than 3
        assert num_actions > 1, "The number of actions must be equal or greater than 3."
        # assert num_actions % 2 == 0, "The number of actions must be an odd number."
        self._moves = ["COOPERATE", "DEFECT"]

        # Optionally add actions

        # none is last possible action, to satisfy discrete action space
        self._moves.append("None")
        self._none = num_actions

        self.agents = ["player_" + str(r) for r in [1,2]]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.action_spaces = {agent: spaces.Discrete(num_actions) for agent in self.agents}
        # self.observation_spaces = {
        #     agent: Discrete(1 + num_actions) for agent in self.agents
        # }
        # self.action_spaces = {i: spaces.Discrete(self.num_actions) for i in self.agents}
        self.observation_spaces = {i: spaces.Box(low=0, high=1, shape=(2, 1), dtype=np.int8) for i in self.agents}

        self.screen = None
        self.history = {agent: [] for agent in self.agents}

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

        self.state = {agent: self._none for agent in self.agents}
        self.observations = {agent: self._none for agent in self.agents}

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
        observation = np.array([self.observations[self.agents[self.agent_name_mapping[agent]]], self.observations[self.agents[self.agent_name_mapping[agent]]]])
        observation = np.expand_dims(observation, axis=1)
        return observation

    def close(self):
        pass

    def reset(self, seed=None, return_info=False, options=None):
        self.reinit()

    def step(self, action, display=False):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection

        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            if display:
                print(f"(ego, partner): {self.state[self.agents[0]], self.state[self.agents[1]]}")

            # same action => 1 reward each agent
            if self.state[self.agents[0]] == self.state[self.agents[1]]:
                # if self.state[self.agents[0]] == 0:
                rewards = (1, 1)

            else:
                # Higher action parity wins: Defect (1) > Cooperate (0)
                if self.state[self.agents[0]] > self.state[self.agents[1]]:
                    rewards = (-1, -1)
                else:
                    rewards = (-1, -1)

            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = rewards

            self.num_moves += 1

            self.dones = {
                agent: self.num_moves >= self.max_cycles for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
                # self.history[self.agent_name_mapping[i]].append(self.state[
                #     self.agents[1 - self.agent_name_mapping[i]]
                # ])
        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = self._none

            self._clear_rewards()

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

