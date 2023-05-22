import torch
import numpy as np
from typing import Tuple, Any
from collections import OrderedDict

import wandb

from HeterogeneousGraphRL.ExtendedModule import ExtendedModule
from HeterogeneousGraphRL.Util.Timer import Timer
from HeterogeneousGraphRL.QLearning.Memory import GraphMultiStepBuffer


def uncuda(tensor):
    return tensor.cpu() if tensor.is_cuda else tensor


class DqnNetProvider:
    def __call__(self, n_actions, **kwargs):
        raise NotImplementedError


class BaseSteppedDQN(ExtendedModule):
    def __init__(self, net_provider: DqnNetProvider = None, net_provider_args=None, drop_nan_rewards: bool = False):
        super(BaseSteppedDQN, self).__init__()
        self.net_provider = net_provider
        self.net_provider_args = net_provider_args if net_provider_args is not None else {}

        self.net = None
        self.target_net = None
        self.drop_nan_rewards = drop_nan_rewards
        self.env_pool = None

        # Metrics

        self.env_steps = 0

    def build_networks(self):
        # from pl_bolts.models.rl.common.networks import MLP
        self.net = self.net_provider(self.n_actions, **self.net_provider_args)
        self.target_net = self.net_provider(self.n_actions, **self.net_provider_args)

    def populate(self, warm_start: int) -> None:
        from HeterogeneousGraphRL.QLearning.Memory import Experience
        from HeterogeneousGraphRL.Util.Progress import maybe_tqdm

        """Populates the buffer with initial experience"""
        if warm_start > 0:
            self.env_pool.async_reset()

            current_len = 0

            with maybe_tqdm(total=warm_start, desc="Warm start") as pbar:
                while current_len < warm_start:
                    states, transitions = self.env_pool.spool(sync=True)

                    self.agent.epsilon = 1.0
                    actions = self.agent(states, self.device)
                    assert len(actions) == len(states)
                    self.env_pool.async_step(actions)

                    for state, action, next_state, reward, done, info in transitions:
                        if np.isnan(reward) and not self.drop_nan_rewards:
                            raise ValueError("Reward is NaN, which is not allowed")

                        self.env_steps += 1

                        if not np.isnan(reward):
                            exp = Experience(state=state, action=action, reward=reward, done=done, new_state=next_state, info=info)
                            self.buffer.append(exp)
                            current_len += 1
                            pbar.update(1)

    def run_greedy_episodes(self, count=1):

        with self.timer.time('greedy'):
            total_r = 0.
            non_failures = 0

            for i in range(count):
                # Only render first greedy episode
                if i == 0:
                    self.greedy_env.request_render_next_episode(name="greedy")

                state = self.greedy_env.reset()

                done = False

                while not done:
                    action = self.agent.get_action([state], self.device)

                    state, r, done, *_ = self.greedy_env.step(action[0])

                    if np.isnan(r) and not self.drop_nan_rewards:
                        raise ValueError("Reward is NaN, which is not allowed")

                    if not np.isnan(r):
                        total_r += r

                # Have -1 as terminal failure but may be slightly higher due to step rewards
                # Comparison is inverted to correctly handle NaN positive or neutral termination as "not a failure"
                if not r < -.8:
                    non_failures += 1

        self.log_dict({
            "greedy_reward": total_r / count,
            "greedy_non_failure_rate": non_failures / count
        })

    @staticmethod
    def buffer_collate(batch):
        from torch.utils.data.dataloader import default_collate

        states, actions, rewards, dones, new_states, infos = zip(*batch)
        return states, default_collate(actions), default_collate(rewards), \
               default_collate(dones), new_states, infos

    def _make_dataloader(self):
        raise NotImplementedError

    def train_dataloader(self):
        return self._make_dataloader()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]


def _safe_mean(arr):
    if len(arr) == 0:
        return .0
    else:
        return float(np.mean(arr))


# from pytorch_lightning import LightningModule
class SteppedDQN(BaseSteppedDQN):
    def __init__(self, greedy_env=None, train_env_pool=None, net_provider: DqnNetProvider = None, net_provider_args=None,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 replay_size: int = 100000,
                 warm_start_size: int = 50000,
                 gamma: float = 0.99,
                 eps_start: float = 1.0,
                 eps_end: float = 0.02,
                 eps_last_frame: int = 1000000,
                 sync_rate: int = 2000,
                 batches_per_epoch: int = 1000,
                 avg_reward_len: int = 100,
                 render_interval: int = 100,
                 greedy_count: int = 20,
                 min_episode_reward: int = -1.,
                 n_steps: int = 1,
                 use_double: bool = False,
                 drop_nan_rewards: bool = False,
                 verbose_logging: bool = True,
                 last_step: int = 1000 * 1000,
                 n_actions = None):
        from HeterogeneousGraphRL.QLearning.Agents import ValueAgent
        from HeterogeneousGraphRL.QLearning.Losses import DqnLoss, DoubleDqnLoss

        super(SteppedDQN, self).__init__(net_provider, net_provider_args=net_provider_args, drop_nan_rewards=drop_nan_rewards)
        self.save_hyperparameters(ignore=['greedy_env', 'train_env_pool', 'net_provider'])

        self.timer = Timer()

        self.lr = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size
        self.gamma = gamma
        self.sync_rate = sync_rate
        self.batches_per_epoch = batches_per_epoch
        self.n_steps = n_steps
        self.render_interval = render_interval
        self.greedy_count = greedy_count
        self.min_episode_reward = min_episode_reward
        self.use_double = use_double
        self.verbose_logging = verbose_logging
        self.last_step = last_step

        self.greedy_env = greedy_env
        self.env_pool = train_env_pool
        self.n_actions = self.greedy_env.action_space.n if n_actions is None else n_actions

        self.buffer = None
        self.net = None
        self.target_net = None
        self.build_networks()

        # Metrics
        self.total_episode_steps = [0]
        self.total_rewards = []
        self.done_episodes = 0
        self.total_steps = 0
        self.ready_envs = 0

        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_frames=eps_last_frame,
            collate_fn=lambda state, device: state
        )

        self.dqn_loss = DoubleDqnLoss(gamma) if self.use_double else DqnLoss(gamma)

        # Average Rewards
        self.avg_reward_len = avg_reward_len

        # for _ in range(avg_reward_len):
        #     self.total_rewards.append(torch.tensor(min_episode_reward, device=self.device))

        self.avg_rewards = _safe_mean(self.total_rewards[-self.avg_reward_len:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def sample_from_buffer(self):
        raise NotImplementedError("Check non PER impl")

        states, actions, rewards, dones, new_states, infos = self.buffer.sample(self.batch_size)

        for idx, _ in enumerate(dones):
            yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx], infos[idx]

    def train_batch(self, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader

        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """

        from HeterogeneousGraphRL.QLearning.Memory import Experience
        import numpy as np

        episode_reward = 0
        episode_steps = 0

        while True:
            self.total_steps += 1

            with self.timer.time('env_spool'):
                states, transitions = self.env_pool.spool()

            self.ready_envs = len(states)

            if len(states) > 0:
                with self.timer.time('agent'):
                    actions = self.agent(states, self.device)
            else:
                # State machine of the env pool may require stepping even with no states, just do so
                actions = []

            with self.timer.time('env_async_step'):
                self.env_pool.async_step(actions)

            for state, action, next_state, reward, done, info in transitions:
                if np.isnan(reward) and not self.drop_nan_rewards:
                    raise ValueError("Reward is NaN, which is not allowed")

                self.env_steps += 1
                episode_steps += 1

                if not np.isnan(reward):
                    episode_reward += reward

                    veh_rel, _, _, wrapped_obs, *_ = state

                    # Catch "illegal" terminal observations
                    if done and reward < -.8 and len(veh_rel) == 0 and wrapped_obs is not None:
                        wrapped_obs.pgraph.to_dot("tmp/bad-terminal.dot")
                        raise ValueError("Empty other vehicle observation for terminal for {}".format(wrapped_obs.ego_id))
                    elif wrapped_obs is not None:
                        # Clear the wrapped obs to save memory
                        wrapped_obs.pgraph = None
                        wrapped_obs.cache = None

                    exp = Experience(state=state, action=action, reward=reward, done=done, new_state=next_state, info=info)

                    self.buffer.append(exp)

                if done:
                    self.done_episodes += 1
                    # TODO: Incorrect
                    self.total_rewards.append(episode_reward)
                    # TODO: Incorrect
                    self.total_episode_steps.append(episode_steps)
                    # TODO: Incorrect
                    self.avg_rewards = _safe_mean(self.total_rewards[-self.avg_reward_len:])

                    if self.render_interval > 0 and self.done_episodes % self.render_interval == 0:
                        # Run a greedy episode and render, don't render non-greedy episodes
                        self.run_greedy_episodes(count=self.greedy_count)
                        # self.env.request_render_next_episode()

                    # TODO: Incorrect
                    episode_steps = 0
                    episode_reward = 0

            self.agent.update_epsilon(self.global_step)

            yield self.sample_from_buffer()

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break

    def training_step_epilog(self, samples, loss, extra_train_logs):
        # Soft update of target network
        with self.timer.time('update_target'):
            if self.global_step % self.sync_rate == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        mean_loss = torch.mean(loss)

        with self.timer.time('logging'):
            if self.log_this_step:
                _, _, rewards, dones, _, infos, *_ = samples

                log_dict = {
                    "total_reward": self.total_rewards[-1] if len(self.total_rewards) > 0 else .0,
                    "avg_reward": self.avg_rewards,
                    "train_loss": mean_loss.detach(),
                    "episodes": self.done_episodes,
                    "episode_steps": self.total_episode_steps[-1],
                    "epsilon": self.agent.epsilon,
                    "ready_envs": self.ready_envs,
                    "env_steps": self.env_steps
                }

                # Only log histograms every few steps for better performance
                if self.verbose_logging and self.global_step % 200 == 0:
                    wandb_hists = {
                        "train_distribution_reward": wandb.Histogram(uncuda(rewards))
                    }

                    # Below only applies if terminal_rewards is 'nan'
                    # assert torch.all(dones == (rewards < -.8))
                    if torch.sum(dones) > 0:
                        wandb_hists['train_distribution_terminal_rewards'] = wandb.Histogram(uncuda(rewards[dones]))
                        wandb_hists['train_distribution_terminal_action_values'] = wandb.Histogram(uncuda(self.dqn_loss.selected_action_values_[dones].detach()))
                        assert rewards[dones].shape == self.dqn_loss.selected_action_values_[dones].shape

                    if self.dqn_loss.has_duelling_:
                        wandb_hists['train_distribution_adv'] = wandb.Histogram(uncuda(self.dqn_loss.advantages_.detach()))
                        wandb_hists['train_distribution_val'] = wandb.Histogram(uncuda(self.dqn_loss.values_.detach()))
                        wandb_hists['train_distribution_target_val'] = wandb.Histogram(uncuda(self.dqn_loss.target_values_))

                    # Histograms not supported by self.log()
                    self.log_dict(wandb_hists)

                log_dict.update({'train_' + k: v for k, v in extra_train_logs.items()})

                # Only log time if verbose
                if self.verbose_logging:
                    log_dict.update({'time/' + k: v for k, v in self.timer.durations.items()})

                self.log_dict(log_dict)

            return OrderedDict({
                "loss": mean_loss
            })

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """

        # calculates training loss
        loss, extra = self.dqn_loss(batch, self.net, self.target_net)

        return self.training_step_epilog(batch, loss, extra)

    def _make_dataloader(self):
        from pl_bolts.datamodules.experience_source import ExperienceSourceDataset
        from torch.utils.data import DataLoader

        self.buffer = GraphMultiStepBuffer(self.replay_size, self.n_steps)
        self.populate(self.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, collate_fn=self.buffer_collate)


class DummyDataLoader:
    def __init__(self, generator_func):
        self.generator_func = generator_func

    def __iter__(self):
        return self.generator_func()


class PERSteppedDQN(SteppedDQN):
    def __init__(self, *args, **kwargs):
        super(PERSteppedDQN, self).__init__(*args, **kwargs)

    def _make_dataloader(self):
        from HeterogeneousGraphRL.QLearning.Memory import GraphPERBuffer

        self.buffer = GraphPERBuffer(self.replay_size, beta_frames=self.last_step)
        self.populate(self.warm_start_size)

        return DummyDataLoader(self.train_batch)

    def sample_from_buffer(self):
        return self.buffer.sample(self.batch_size)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        from ..Training import pinned_transfer

        # Do not move states and next_states to the device as pytorch lightning wants, as we already moved to the
        # device and this slows down training. Also don't move *extra.

        samples, indices, weights = batch
        states, actions, rewards, dones, next_states, *extra = samples

        states = self.net.collate(states)
        actions = pinned_transfer(torch.from_numpy(actions), device)
        rewards = pinned_transfer(torch.from_numpy(rewards), device)
        next_states = self.net.collate(next_states)
        dones = pinned_transfer(torch.from_numpy(dones), device)

        # Don't need to move indices
        weights = pinned_transfer(torch.from_numpy(weights), device)

        # Reconstruct tuples and return
        samples = (states, actions, rewards, dones, next_states, *extra)
        return samples, indices, weights

    def training_step(self, batch, _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """
        from HeterogeneousGraphRL.QLearning.Losses import per_dqn_loss

        samples, indices, weights = batch
        assert len(indices) == self.batch_size
        # fini = torch.isfinite(weights)
        # torch.testing.assert_allclose(fini, torch.ones_like(fini))

        # calculates training loss
        with self.timer.time('base_loss'):
            base_loss, extra = self.dqn_loss(samples, self.net, self.target_net, report_extra=self.log_this_step)
            # fini = torch.isfinite(base_loss)
            # torch.testing.assert_allclose(fini, torch.ones_like(fini))

        with self.timer.time('per_loss'):
            loss, batch_weights = per_dqn_loss(base_loss, weights)

            # if self._use_dp_or_ddp2(self.trainer):
            #     loss = loss.unsqueeze(0)

            # update priorities in buffer
            self.buffer.update_priorities(indices, batch_weights)
            new_beta = self.buffer.update_beta(self.global_step)

            if self.log_this_step:
                self.log_dict({"per_beta": new_beta})

        return self.training_step_epilog(samples, loss, extra)
