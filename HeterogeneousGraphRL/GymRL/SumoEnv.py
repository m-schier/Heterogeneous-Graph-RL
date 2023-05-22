import warnings

import gym
import numpy as np
from ..SumoInterop import Context
from typing import Optional
from torch_geometric.data.data import BaseData


class SumoEnvInfo:
    def __init__(self, msg, history=None, ego_collision_veh=None):
        self.msg = msg
        self.history = history
        self.ego_collision_veh = ego_collision_veh


class SumoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, default_sumo_mode='libsumo'):
        self._sumo_ctx: Optional[Context] = None
        self._sumo_mode = default_sumo_mode
        self._is_done = False
        self._last_obs = None
        self._sumo_seeder: Optional[np.random.Generator] = None
        self._tracked_vehicles = None
        self._render_next_episode = False
        self._render_name = None
        self._render_recorder = None
        self._render_recorder_args = None
        self._movie_render_callback = None

    def seed(self, seed=None):
        self._sumo_seeder = np.random.default_rng(seed=seed)

    def get_rng(self) -> np.random.Generator:
        return self._sumo_seeder

    def get_tracked_vehicles(self):
        return None if self._tracked_vehicles is None else self._tracked_vehicles[:]

    def set_sumo_mode(self, mode):
        self._sumo_mode = mode

    def sumo_init(self, mode='libsumo'):
        """Initialize a sumo context for the given step and bring the simulation to the desired point"""
        raise NotImplementedError

    def sumo_leadin(self):
        raise NotImplementedError

    def sumo_step(self, action):
        raise NotImplementedError

    def sumo_make_observation(self):
        raise NotImplementedError

    def safe_observation(self):
        obs = self.sumo_make_observation()

        if issubclass(type(self.observation_space), gym.spaces.Box):
            if not issubclass(type(obs), np.ndarray):
                raise ValueError("The implementation returned an observation which is not ndarray")
            elif obs.shape != self.observation_space.shape:
                raise ValueError("The implementation returned an observation of incorrect shape: was {}, expected {}".format(obs.shape, self.observation_space.shape))
        elif issubclass(type(obs), BaseData):
            pass  # Graph data always acceptable
        elif self.observation_space is None:
            pass  # Accept everything if not adhering to standard
        else:
            raise ValueError("Observation space type not understood: {}, observation type: {}".format(type(self.observation_space), type(obs)))

        self._last_obs = obs

        return obs

    def step(self, action):
        if self._is_done:
            raise ValueError("Calling step() on a done environment is not legal for SumoEnv")

        if self._render_recorder is not None:
            self._render_recorder.prestep()

        reward, done, info = self.sumo_step(action)

        if self._render_recorder is not None and info is not None and info.msg is not None:
            self._render_recorder.poststep(info.msg)

        if done:
            # We expect that no valid observation may be formed anyways if the environment is done
            obs = self._last_obs if self._last_obs is not None else self.observation_space.sample()
            self._is_done = True
        else:
            obs = self.safe_observation()
        return obs, reward, done, info

    def request_render_next_episode(self, name=None, **kwargs):
        self._render_next_episode = True
        self._render_name = name
        self._render_recorder_args = kwargs

    def set_movie_render_callback(self, cb):
        self._movie_render_callback = cb

    def reset(self):
        if self._render_recorder is not None:
            # TODO: Cleanup
            movie = self._render_recorder.get_result(**self._render_recorder_args)

            if self._movie_render_callback is None:
                import wandb
                if wandb.run is None:
                    warnings.warn("Cannot log recording because W&B has not been initialized")
                elif movie is not None:
                    movie = np.moveaxis(movie, -1, 1)  # Move channel to second axis
                    wandb.log({self._render_recorder.name: wandb.data_types.Video(movie, fps=8)})
            else:
                self._movie_render_callback(movie)
            self._render_recorder.close()
            self._render_recorder = None

        # Seed if not done
        if self._sumo_seeder is None:
            self.seed()

        if self._sumo_ctx is not None:
            self._sumo_ctx.close()
            self._sumo_ctx = None
        self._is_done = False
        self._sumo_ctx = self.sumo_init(mode=self._sumo_mode if not self._render_next_episode else 'sumo-gui')
        self._sumo_ctx.set_seed(self._sumo_seeder.integers(2 ** 31))
        self._sumo_ctx.open()
        self._tracked_vehicles = self.sumo_leadin()

        if self._render_next_episode:
            from ..SumoInterop.Recorder import Recorder
            video_name = 'video' if self._render_name is None else 'video_{}'.format(self._render_name)
            self._render_recorder = Recorder(self._tracked_vehicles[0], width=1280, height=720, name=video_name)
            self._render_next_episode = False
            self._render_name = None

        obs = self.safe_observation()
        return obs

    def close(self):
        if self._sumo_ctx is not None:
            self._sumo_ctx.close()
            self._sumo_ctx = None
