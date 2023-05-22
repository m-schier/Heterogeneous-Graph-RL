from typing import Tuple, List, Union, Dict
import torch
from torch import Tensor, nn
import numpy as np


ASSERT_LOSSES = False


def _maybe_add_terminal_correlation(log_dict, dones, predicted, rewards):
    if torch.sum(dones) > 0:
        obs = torch.stack([predicted[dones], rewards[dones]], dim=-1)
        assert obs.shape[1] == 2
        cc = torch.corrcoef(obs)

        if ASSERT_LOSSES:
            assert torch.isfinite(cc[0, 1])
        log_dict['terminal_correlation'] = cc[0, 1]


def _maybe_add_terminal_mse(log_dict, dones, predicted, rewards):
    if torch.sum(dones) > 0:
        with torch.no_grad():
            # Suggested linear correlation does not work!
            preds = predicted[dones]
            rews = rewards[dones]
            log_dict['terminal_reward_mse'] = torch.nn.MSELoss()(preds, rews)

            failures = rews < -0.8
            non_failures = torch.logical_not(failures)

            if torch.sum(failures) > 0:
                loss = torch.nn.MSELoss()(preds[failures], rews[failures])
                log_dict['terminal_reward_mse_failures'] = loss
            if torch.sum(non_failures) > 0:
                log_dict['terminal_reward_mse_non_failures'] = torch.nn.MSELoss()(preds[non_failures], rews[non_failures])


def relative_policy_entropy(pred_rewards):
    policy = torch.argmax(pred_rewards, dim=-1)
    n = pred_rewards.shape[-1]
    max_entropy = -torch.log(torch.tensor(1 / n, device=pred_rewards.device))
    probs = torch.sum(policy[:, None] == torch.arange(n, device=policy.device), dim=0) / pred_rewards.shape[0]
    assert probs.shape == (n,)

    if ASSERT_LOSSES:
        torch.testing.assert_close(torch.sum(probs), torch.tensor(1., device=probs.device, dtype=probs.dtype))

    non_zero_probs = probs[probs > 0.]
    entropy = -torch.sum(non_zero_probs * torch.log(non_zero_probs))
    rel_entropy = entropy / max_entropy

    if ASSERT_LOSSES:
        assert torch.isfinite(rel_entropy)
    return rel_entropy


class DqnLoss:
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.loss = torch.nn.MSELoss(reduction='none')
        self.has_duelling = False
        self.selected_action_values_ = None

    def __call__(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], net: nn.Module, target_net: nn.Module,
                 report_extra: bool = True) -> Tuple[Tensor, Dict[str, float]]:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data
            net: main training network
            target_net: target network of the main training network

        Returns:
            loss
        """
        extra = {}
        states, actions, rewards, dones, next_states, *_ = batch

        self.has_duelling_ = hasattr(net, 'adv')

        actions = actions.long().squeeze(-1)

        pred_rewards = net(states)

        if self.has_duelling_:
            self.values_ = net.val
            self.advantages_ = net.adv

        if report_extra:
            extra['relative_policy_entropy'] = relative_policy_entropy(pred_rewards)

        state_action_values = pred_rewards.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        self.selected_action_values_ = state_action_values

        with torch.no_grad():
            target_rewards = target_net(next_states)

            if self.has_duelling_:
                self.target_values_ = target_net.val

            next_state_values = target_rewards.max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        if report_extra:
            extra['residual_variance'] = (torch.var(state_action_values - expected_state_action_values) / torch.var(expected_state_action_values)).detach()
            _maybe_add_terminal_mse(extra, dones, state_action_values, rewards)

        loss = self.loss(state_action_values, expected_state_action_values)

        return loss, extra


class DoubleDqnLoss:
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.loss = torch.nn.MSELoss(reduction='none')
        self.has_duelling_ = False
        self.target_values_ = None
        self.values_ = None
        self.advantages_ = None
        self.selected_action_values_ = None

    def __call__(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        net: nn.Module,
        target_net: nn.Module,
        report_extra: bool = True,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Calculates the mse loss using a mini batch from the replay buffer. This uses an improvement to the original
        DQN loss by using the double dqn. This is shown by using the actions of the train network to pick the value
        from the target network. This code is heavily commented in order to explain the process clearly.

        Args:
            batch: current mini batch of replay data
            net: main training network
            target_net: target network of the main training network
            gamma: discount factor

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states, *_ = batch  # batch of experiences, batch_size = 16

        self.has_duelling_ = hasattr(net, 'adv')

        extra = {}

        actions = actions.long().squeeze(-1)

        pred_rewards = net(states)

        if self.has_duelling_:
            self.values_ = net.val
            self.advantages_ = net.adv

        if report_extra:
            extra['relative_policy_entropy'] = relative_policy_entropy(pred_rewards)

        state_action_values = pred_rewards.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        self.selected_action_values_ = state_action_values

        # dont want to mess with gradients when using the target network
        with torch.no_grad():
            next_outputs = net(next_states)  # [16, 2], [batch, action_space]

            next_state_acts = next_outputs.max(1)[1].unsqueeze(-1)  # take action at the index with the highest value
            next_tgt_out = target_net(next_states)

            if self.has_duelling_:
                self.target_values_ = target_net.val

            # Take the value of the action chosen by the train network
            next_state_values = next_tgt_out.gather(1, next_state_acts).squeeze(-1)
            next_state_values[dones] = 0.0  # any steps flagged as done get a 0 value
            next_state_values = next_state_values.detach()  # remove values from the graph, no grads needed

        # calc expected discounted return of next_state_values
        expected_state_action_values = next_state_values * self.gamma + rewards

        if report_extra:
            extra['residual_variance'] = (torch.var(state_action_values - expected_state_action_values) / torch.var(expected_state_action_values)).detach()

        # TODO Remove debug check
        if ASSERT_LOSSES:
            assert torch.all(rewards[dones] == expected_state_action_values[dones])

        if report_extra:
            _maybe_add_terminal_mse(extra, dones, state_action_values, rewards)

        # Standard MSE loss between the state action values of the current state and the
        # expected state action values of the next state
        loss = self.loss(state_action_values, expected_state_action_values)

        return loss, extra


def per_dqn_loss(
    batch_loss: Tensor,
    batch_weights: List,
) -> Tuple[Tensor, np.ndarray]:
    """Calculates the mse loss with the priority weights of the batch from the PER buffer.

    Args:
        batch_loss: previous loss per sample in batch either calculated using dueling or double DQN
        batch_weights: how each of these samples are weighted in terms of priority

    Returns:
        loss and batch_weights
    """

    assert batch_loss.shape == (len(batch_weights),)

    losses_v = batch_weights * batch_loss
    # This line was likely incorrect in framework, originally using (batch_weights * batch_loss)
    new_weights = (batch_loss + 1e-5).detach().cpu().numpy()

    if ASSERT_LOSSES:
        assert np.all(np.isfinite(new_weights))
    return losses_v, new_weights
