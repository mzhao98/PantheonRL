
import multiprocessing
multiprocessing.set_start_method("fork")

from pettingzoo.mpe import simple_v2 as e

from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent
from pantheonrl.envs.pettingzoo import PettingZooAECWrapper
import gym

import numpy as np


def run_test(ego, env, num_episodes, render=False):
    rewards = []
    for game in range(num_episodes):
        obs = env.reset()
        done = False
        reward = 0
        if render:
            env.render()
        while not done:
            action = ego.get_action(obs, False)
            obs, newreward, done, _ = env.step(action)
            reward += newreward

            if render:
                env.render()
                sleep(1/60)

        rewards.append(reward)

    env.close()
    print(f"Average Reward: {sum(rewards)/num_episodes}")
    print(f"Standard Deviation: {np.std(rewards)}")


# We have a simple wrapper class that converts PettingZoo environments to
# work with our framework.
#
# WARNING: PettingZoo environments with complex spaces may not be directly
# compatible with our agents.

# env, altenv = generate_env(args)
env = PettingZooAECWrapper(e.env(max_cycles=25, continuous_actions=False))


# ego = OnPolicyAgent(PPO('MlpPolicy', env.getDummyEnv(i), verbose=1))

ego = PPO('MlpPolicy', env.getDummyEnv(0), verbose=1)
output_of_learn = ego.learn(total_timesteps=1000, tb_log_name='simple_v2-PPO-1')

print("output_of_learn", output_of_learn)

def gen_load(policy_type, location):
    if policy_type == 'PPO':
        agent = PPO.load(location)
    elif policy_type == 'ModularAlgorithm':
        agent = ModularAlgorithm.load(location)
    elif policy_type == 'BC':
        agent = BCShell(reconstruct_policy(location))
    else:
        raise EnvException("Not a valid FIXED/LOAD policy")

    return agent

def gen_fixed(policy_type, location):
    agent = gen_load(policy_type, location)
    return StaticPolicyAgent(agent.policy)



ego.save('saved_models/simple_v2_ego')

test_agent = gen_fixed("PPO", 'saved_models/simple_v2_ego')

run_test(test_agent, env, 5, render=False)



transition = env.get_transitions()
transition.write_transition('saved_models/simple_v2_transitions')
















