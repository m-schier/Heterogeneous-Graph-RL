import collections
from collections import deque, namedtuple
from typing import List, Tuple, Union

import numpy as np

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "new_state", "info"], defaults=(None,) * 6)


# Copy of the pl_bolts implementation, but don't try to put graphs into a numpy array


class GraphBuffer:
    """Basic Buffer for storing a single experience at a time."""

    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity: size of the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    # pylint: disable=unused-argument
    def sample(self, *args) -> Union[Tuple, List[Tuple]]:
        """
        returns everything in the buffer so far it is then reset
        Returns:
            a batch of tuple np arrays of state, action, reward, done, next_state
        """
        states, actions, rewards, auxs, dones, next_states, infos = zip(*(self.buffer[idx] for idx in range(self.__len__())))

        self.buffer.clear()

        return (
            states,
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            auxs,
            np.array(dones, dtype=np.bool),
            next_states,
            infos,
        )


class GraphReplayBuffer(GraphBuffer):
    """Replay Buffer for storing past experiences allowing the agent to learn from them."""

    def sample(self, batch_size: int) -> Tuple:
        """Takes a sample of the buffer.

        Args:
            batch_size: current batch_size

        Returns:
            a batch of tuple np arrays of state, action, reward, done, next_state
        """

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states, infos = zip(*(self.buffer[idx] for idx in indices))

        return (
            states,
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            next_states,
            infos,
        )


class GraphMultiStepBuffer(GraphReplayBuffer):
    """N Step Replay Buffer."""

    def __init__(self, capacity: int, n_steps: int = 1, gamma: float = 0.99) -> None:
        """
        Args:
            capacity: max number of experiences that will be stored in the buffer
            n_steps: number of steps used for calculating discounted reward/experience
            gamma: discount factor when calculating n_step discounted reward of the experience being stored in buffer
        """
        super().__init__(capacity)

        self.n_steps = n_steps
        self.gamma = gamma
        self.history = deque(maxlen=self.n_steps)
        self.exp_history_queue = deque()

    def append(self, exp: Experience) -> None:
        """Add experience to the buffer.

        Args:
            exp: tuple (state, action, reward, done, new_state)
        """
        self.update_history_queue(exp)  # add single step experience to history
        while self.exp_history_queue:  # go through all the n_steps that have been queued
            experiences = self.exp_history_queue.popleft()  # get the latest n_step experience from queue

            last_exp_state, tail_experiences = self.split_head_tail_exp(experiences)

            total_reward = self.discount_rewards(tail_experiences)

            n_step_exp = Experience(
                state=experiences[0].state,
                action=experiences[0].action,
                reward=total_reward,
                done=experiences[0].done,
                new_state=last_exp_state,
                info=experiences[0].info
            )

            self.buffer.append(n_step_exp)  # add n_step experience to buffer

    def update_history_queue(self, exp) -> None:
        """Updates the experience history queue with the lastest experiences. In the event of an experience step is
        in the done state, the history will be incrementally appended to the queue, removing the tail of the
        history each time.

        Args:
            env_idx: index of the environment
            exp: the current experience
            history: history of experience steps for this environment
        """
        self.history.append(exp)

        # If there is a full history of step, append history to queue
        if len(self.history) == self.n_steps:
            self.exp_history_queue.append(list(self.history))

        if exp.done:
            if 0 < len(self.history) < self.n_steps:
                self.exp_history_queue.append(list(self.history))

            # generate tail of history, incrementally append history to queue
            while len(self.history) > 2:
                self.history.popleft()
                self.exp_history_queue.append(list(self.history))

            # when there are only 2 experiences left in the history,
            # append to the queue then update the env stats and reset the environment
            if len(self.history) > 1:
                self.history.popleft()
                self.exp_history_queue.append(list(self.history))

            # Clear that last tail in the history once all others have been added to the queue
            self.history.clear()

    def split_head_tail_exp(self, experiences: Tuple[Experience]) -> Tuple[List, Tuple[Experience]]:
        """Takes in a tuple of experiences and returns the last state and tail experiences based on if the last
        state is the end of an episode.

        Args:
            experiences: Tuple of N Experience

        Returns:
            last state (Array or None) and remaining Experience
        """
        last_exp_state = experiences[-1].new_state
        tail_experiences = experiences

        if experiences[-1].done and len(experiences) <= self.n_steps:
            tail_experiences = experiences

        return last_exp_state, tail_experiences

    def discount_rewards(self, experiences: Tuple[Experience]) -> float:
        """Calculates the discounted reward over N experiences.

        Args:
            experiences: Tuple of Experience

        Returns:
            total discounted reward
        """
        total_reward = 0.0
        for exp in reversed(experiences):
            total_reward = (self.gamma * total_reward) + exp.reward
        return total_reward


class GraphPERBuffer(GraphReplayBuffer):
    """simple list based Prioritized Experience Replay Buffer Based on implementation found here:

    https://github.com/Shmuma/ptan/blob/master/ptan/experience.py#L371
    """

    def __init__(self, buffer_size, prob_alpha=0.6, beta_start=0.4, beta_frames=100000):
        super().__init__(capacity=buffer_size)
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.__prob_alpha = prob_alpha
        self.capacity = buffer_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.priorities_with_alpha = np.zeros((buffer_size,), dtype=np.float32)
        self.__current_max_prio = None

    @property
    def prob_alpha(self):
        return self.__prob_alpha

    def update_beta(self, step) -> float:
        """Update the beta value which accounts for the bias in the PER.

        Args:
            step: current global step

        Returns:
            beta value for this indexed experience
        """
        beta_val = self.beta_start + step * (1.0 - self.beta_start) / self.beta_frames
        self.beta = min(1.0, beta_val)

        return self.beta

    def append(self, exp) -> None:
        """Adds experiences from exp_source to the PER buffer.

        Args:
            exp: experience tuple being added to the buffer
        """
        # Cache max priority because costly and possibly multiple calls to append before an update
        if self.__current_max_prio is None:
            self.__current_max_prio = self.priorities_with_alpha.max() if self.buffer else 1.0 ** self.__prob_alpha
        max_prio_with_alpha = self.__current_max_prio

        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp

        # the priority for the latest sample is set to max priority so it will be resampled soon
        self.priorities_with_alpha[self.pos] = max_prio_with_alpha

        # update position, loop back if it reaches the end
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=32) -> Tuple:
        """Takes a prioritized sample from the buffer.

        Args:
            batch_size: size of sample

        Returns:
            sample of experiences chosen with ranked probability
        """
        # get list of priority rankings
        if len(self.buffer) == self.capacity:
            prios_with_alpha = self.priorities_with_alpha
        else:
            prios_with_alpha = self.priorities_with_alpha[:self.pos]

        # probability to the power of alpha to weight how important that probability it, 0 = normal distirbution
        probs = prios_with_alpha / prios_with_alpha.sum()

        # choise sample of indices based on the priority prob distribution
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        # samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, dones, next_states, infos = zip(*(self.buffer[idx] for idx in indices))

        samples = (
            states,
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            next_states,
            infos
        )
        total = len(self.buffer)

        # weight of each sample datum to compensate for the bias added in with prioritising samples
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # return the samples, the indices chosen and the weight of each datum in the sample
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices: np.ndarray, batch_priorities: np.ndarray) -> None:
        """Update the priorities from the last batch, this should be called after the loss for this batch has been
        calculated.

        Args:
            batch_indices: index of each datum in the batch
            batch_priorities: priority of each datum in the batch
        """
        assert batch_priorities.shape == batch_indices.shape
        assert len(batch_priorities.shape) == 1

        self.priorities_with_alpha[batch_indices] = batch_priorities ** self.__prob_alpha

        # Invalidate cached maximum priority
        self.__current_max_prio = None
