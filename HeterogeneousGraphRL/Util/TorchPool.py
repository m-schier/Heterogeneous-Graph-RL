import multiprocessing
import queue

import torch
import torch.multiprocessing as mp
import numpy as np

from .AbstractPool import AbstractEnvironmentPool
from warnings import warn

try:
    # Not available on Windows thus optional
    from faster_fifo import Queue
    _HAVE_FIFO = True
except ImportError:
    Queue = mp.Queue
    _HAVE_FIFO = False
    warn("Failed to import faster-fifo")


class TransmissionWrapper:
    def encode(self, x):
        return x

    def decode(self, x):
        return x


def _mp_worker_do_work(env_provider, act_queue: Queue, obs_queue: Queue, idx: int, seed: int, state_encoder):
    from gym import Env

    # Disable computation multi-threading in worker to prevent resource contention
    torch.set_num_threads(1)

    env: Env = env_provider()
    env.seed(seed)

    reset = False

    last_obs = None

    while True:
        cmd, *extra = act_queue.recv()

        if cmd == 'stop':
            print("Received stop in {}, goodbye".format(idx))
            break
        elif cmd == 'reset':
            assert not reset
            reset = True
            last_obs = env.reset()
            obs_queue.put((idx, state_encoder(last_obs), None))  # No transition on hard reset
        elif cmd == 'step':
            assert reset
            state = last_obs
            action = extra[0]
            next_state, reward, done, *_ = env.step(action)

            if done:
                next_state = env.reset()  # On done next_state is garbage anyway and not used, immediately reset

            # Drop the info when multiprocessing, everything else is super slow
            response = (idx, None, (state_encoder(state), action, state_encoder(next_state), reward, done, None))
            obs_queue.put(response)
            last_obs = next_state
        else:
            raise ValueError(cmd)


def container_remap(obj, type_filter, fn):
    if isinstance(obj, tuple):
        return tuple([container_remap(x, type_filter, fn) for x in obj])
    elif isinstance(obj, list):
        return [container_remap(x, type_filter, fn) for x in obj]
    elif isinstance(obj, dict):
        return {k: container_remap(v, type_filter, fn) for k, v in obj.items()}
    elif isinstance(obj, type_filter):
        return fn(obj)
    else:
        return obj


class NumpyTransmissionWrapper(TransmissionWrapper):
    def encode(self, x):
        return container_remap(x, (torch.Tensor,), lambda el: el.numpy())

    def decode(self, x):
        return container_remap(x, (np.ndarray,), lambda el: torch.from_numpy(el))


class TorchPool(AbstractEnvironmentPool):
    def __init__(self, env_provider, n_workers, state_wrapper=None):
        self.manager = mp.Manager()

        # TODO: manager.Queue appears to be slower, but works without dead locks, maybe review
        q_func = Queue if _HAVE_FIFO else self.manager.Queue
        p_func = mp.Process

        self.state_wrapper = state_wrapper or TransmissionWrapper()

        self.obs_queues = q_func()
        pipes = [multiprocessing.Pipe() for _ in range(n_workers)]
        act_recvs, self.act_queues = zip(*pipes)

        self.workers = [p_func(daemon=True, target=_mp_worker_do_work, args=(env_provider, aq, self.obs_queues, i,
                                                                             np.random.randint(0, 1 << 31),
                                                                             self.state_wrapper.encode))
                        for i, aq in enumerate(act_recvs)]

        for w in self.workers:
            w.start()

        self._last_idxs = []
        self._is_reset = False
        self._obs_ready = False  # Doesn't actually indicate obs really ready, but state machine is ready to query

    def async_reset(self):
        assert not self._obs_ready
        assert not self._is_reset

        for aq in self.act_queues:
            aq.send(('reset',))

        self._is_reset = True
        self._obs_ready = True

    def async_step(self, actions):
        assert self._is_reset
        assert not self._obs_ready

        assert len(actions) == len(self._last_idxs)

        for i, a in zip(self._last_idxs, actions):
            self.act_queues[i].send(('step', int(a)))

        self._obs_ready = True

    def close(self):
        for a in self.act_queues:
            a.send(('stop',))

        for w in self.workers:
            w.join()

        self.workers, self.act_queues, self.obs_queues = [], [], []

    def spool(self, sync=False):
        assert self._obs_ready

        states, transitions = [], []
        self._last_idxs = []

        while True:
            try:
                i, s, t = self.obs_queues.get(block=sync)
                s = self.state_wrapper.decode(s)
                t = self.state_wrapper.decode(t)

                if s is None:
                    assert t is not None
                    # Indicates that s is equal to next state of t, used by worker to reduce transmission size
                    s = t[2]

                states.append(s)

                if t is not None:
                    transitions.append(t)

                self._last_idxs.append(i)

                if (len(self._last_idxs) == len(self.workers)) and sync:
                    # Break if syncing and we have one from each worker
                    break
            except queue.Empty:
                break

        self._obs_ready = False
        return states, transitions
