from pl_bolts.models.rl.common.agents import Agent
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data.data import BaseData
from torch_geometric.data import Batch, Data, HeteroData
from typing import Union, List


class ValueAgent(Agent):
    """Value based agent that returns an action based on the Q values from the network."""

    def __init__(
        self,
        net: nn.Module,
        action_space: int,
        eps_start: float = 1.0,
        eps_end: float = 0.2,
        eps_frames: float = 1000,
        collate_fn = None
    ):
        super().__init__(net)
        self.action_space = action_space
        self.eps_start = eps_start
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_frames = eps_frames

        def default_collate(state, device):
            if not issubclass(type(state), BaseData):
                return Batch.from_data_list(state).to(device)
            else:
                return state

        self.collate_fn = default_collate if collate_fn is None else collate_fn

    @torch.no_grad()
    def __call__(self, state: Tensor, device: str) -> List[int]:
        """Takes in the current state and returns the action based on the agents policy.

        Args:
            state: current state of the environment
            device: the device used for the current batch

        Returns:
            action defined by policy
        """
        if not isinstance(state, list):
            state = [state]

        if np.random.random() < self.epsilon:
            action = self.get_random_action(state)
        else:
            action = self.get_action(state, device)

        return action

    def get_random_action(self, state: Tensor) -> int:
        """returns a random action."""
        actions = []

        for i in range(len(state)):
            action = np.random.randint(0, self.action_space)
            actions.append(action)

        return actions

    def get_action(self, state: Union[List[Data], List[HeteroData], Batch], device: torch.device):
        """Returns the best action based on the Q values of the network.

        Args:
            state: current state of the environment
            device: the device used for the current batch

        Returns:
            action defined by Q values
        """
        # DONT
        # if not isinstance(state, Tensor):
        #     state = torch.tensor(state, device=device)
        state = self.collate_fn(state, device)
        q_values = self.net(state)
        _, actions = torch.max(q_values, dim=1)
        return actions.detach().cpu().numpy()

    def update_epsilon(self, step: int) -> None:
        """Updates the epsilon value based on the current step.

        Args:
            step: current global step
        """
        m = (self.eps_end - self.eps_start) / self.eps_frames
        y = m * step + self.eps_start
        self.epsilon = max(self.eps_end, min(self.eps_start, y))
        # self.epsilon = max(self.eps_end, self.eps_start - (step + 1) / self.eps_frames)
