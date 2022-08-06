from functools import lru_cache

from gym.utils import EzPickle
from pettingzoo.utils import wrappers
# from pettingzoo.utils.conversions import from_parallel_wrapper
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import ParallelEnv

from .env_creator import get_env_creator

MAX_CYCLES = 1000


# def parallel_env(max_cycles=MAX_CYCLES, **ssd_args):
#     return _parallel_env(max_cycles, **ssd_args)
#
#
# # def raw_env(max_cycles=MAX_CYCLES, **ssd_args):
# #     return parallel_wrapper_fn(parallel_env(max_cycles, **ssd_args))
#
# def raw_env(max_cycles=MAX_CYCLES, **ssd_args):
#     return parallel_env(max_cycles, **ssd_args)
#
#
# def env(max_cycles=MAX_CYCLES, **ssd_args):
#     aec_env = raw_env(max_cycles, **ssd_args)
#     aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
#     aec_env = wrappers.OrderEnforcingWrapper(aec_env)
#     return aec_env


def env(max_cycles=MAX_CYCLES, **ssd_args):
    env = get_env_creator(**ssd_args)(ssd_args["num_agents"])
    env = ssd_parallel_env(env, max_cycles)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

class ssd_parallel_env(ParallelEnv):
    def __init__(self, env, max_cycles):
        self.ssd_env = env
        self.max_cycles = max_cycles
        self.possible_agents = list(self.ssd_env.agents.keys())
        self.ssd_env.reset()
        self.observation_space = lru_cache(maxsize=None)(lambda agent_id: env.observation_space)
        self.observation_spaces = {agent: env.observation_space for agent in self.possible_agents}
        self.action_space = lru_cache(maxsize=None)(lambda agent_id: env.action_space)
        self.action_spaces = {agent: env.action_space for agent in self.possible_agents}

    def reset(self):
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        self.dones = {agent: False for agent in self.agents}
        return self.ssd_env.reset()

    def reinit(self):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        # self.rewards = {agent: 0 for agent in self.agents}
        # self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # self.dones = {agent: False for agent in self.agents}
        # self.infos = {agent: {} for agent in self.agents}
        #
        # self.state = {agent: self._none for agent in self.agents}
        # self.observations = {agent: self._none for agent in self.agents}
        #
        # self.num_moves = 0

    def seed(self, seed=None):
        return self.ssd_env.seed()

    def render(self, mode="human"):
        return self.ssd_env.render(mode=mode)

    def close(self):
        self.ssd_env.close()

    def step(self, actions):
        obss, rews, self.dones, infos = self.ssd_env.step(actions)
        del self.dones["__all__"]
        self.num_cycles += 1
        if self.num_cycles >= self.max_cycles:
            self.dones = {agent: True for agent in self.agents}
        self.agents = [agent for agent in self.agents if not self.dones[agent]]
        return obss, rews, self.dones, infos


class _parallel_env(ssd_parallel_env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, max_cycles, **ssd_args):
        EzPickle.__init__(self, max_cycles, **ssd_args)
        env = get_env_creator(**ssd_args)(ssd_args["num_agents"])
        super().__init__(env, max_cycles)