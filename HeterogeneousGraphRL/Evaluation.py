from collections import namedtuple

import numpy as np

from HeterogeneousGraphRL.GymRL.GraphSpeedEnv import GraphSpeedEnv


TestMetrics = namedtuple('TestMetrics', ['total_runs', 'failed_runs', 'succeeded_runs', 'total_steps',
                                         'overspeed_steps', 'avg_reward', 'terminal_transitions',
                                         'avg_successful_episode_steps', 'avg_episode_steps'])


def test_metrics(net, env: GraphSpeedEnv, runs=100, return_terminals=False):
    from tqdm import tqdm
    import torch
    from math import isnan

    failed_runs = 0
    succeeded_runs = 0
    total_steps = 0
    overspeed_steps = 0

    episode_rewards = []
    successful_episode_steps = []
    episode_lengths = []
    term_buf = []

    # Reset the seed for the environment, this will reset the RNG to a fixed state but will still generate different
    # (but predictably fixed across multiple runs) seeds
    env.seed(0)

    for _ in tqdm(range(runs)):
        obs = env.reset()
        done = False
        r = 0
        episode_r = 0

        while not done:
            act = torch.argmax(net([obs])).item()
            obs, r, done, *_ = env.step(act)

            if not isnan(r):
                episode_r += r
                if done and return_terminals:
                    term_buf.append((obs, act, r))
            else:
                # A NaN reward must always terminate the episode
                assert done

        episode_rewards.append(episode_r)
        episode_lengths.append(env.steps)
        total_steps += env.steps
        overspeed_steps += env.steps_with_overspeed

        if r < -.8:
            failed_runs += 1
        elif env.steps < env.max_steps:
            succeeded_runs += 1
            successful_episode_steps.append(env.steps)

    se_steps = np.nan if len(successful_episode_steps) == 0 else np.mean(successful_episode_steps)
    avg_steps = np.mean(episode_lengths)
    return TestMetrics(runs, failed_runs, succeeded_runs, total_steps, overspeed_steps,
                       sum(episode_rewards) / len(episode_rewards), term_buf, se_steps, avg_steps)
