from .SumoEnv import SumoEnv, SumoEnvInfo
from .Spawner import ManualTripSpawner, NoopSpawner, SingleVehicleRouteSpawner
from enum import IntEnum
from ..SumoInterop import Context, Vehicle
from typing import Optional
import gym
import numpy as np
import os


class SpeedAction(IntEnum):
    MAINTAIN = 0  # Maintain current speed
    ACCEL = 1  # Accelerate with 2 m/s^2
    DECEL = 2  # Decelerate with 2 m/s^2


# GraphSpeedEnvObs = namedtuple("GraphSpeedEnvObs", ["state", "aux"])
class GraphSpeedEnvObs:
    def __init__(self, state, aux):
        self.state = state
        self.aux = aux

    # Legacy support for tuple unpacking
    def __iter__(self):
        return iter((self.state, self.aux))


# GraphSpeedHistory = namedtuple("GraphSpeedHistory", ["scenario", "seed", "ego_id", "actions", "step"])
class GraphSpeedHistory:
    def __init__(self, scenario, seed, ego_id, actions, step):
        self.scenario = scenario
        self.seed = seed
        self.ego_id = ego_id
        self.actions = actions
        self.step = step

    # Legacy support for tuple unpacking
    def __iter__(self):
        return iter((self.scenario, self.seed, self.ego_id, self.actions, self.step))


class GraphSpeedEnvEncoder:
    def __call__(self, ego_id, ctx):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


SINGLE_SPAWN_SCENARIOS = {'S1', 'S1b', 'S2', 'S2b', 'S3', 'S3b', 'S4', 'S4b', 'S5'}


class GraphSpeedEnv(SumoEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, scenario, encoder: GraphSpeedEnvEncoder, max_steps=600, terminal_reward=1., timeout_reward=0.,
                 terminal_penalty=1., overspeed_factor=.003, underspeed_factor=.001, acceleration_factor=.0002,
                 unrelated_collision_handling='abort', punish_standing_collision=False, scenario_selection='random',
                 allow_overspeed=True, auto_start_gui=True, super_steps=1, micro_step_length=.4, randomize_speeds=False,
                 accel=4.5,
                 **kwargs):
        super(GraphSpeedEnv, self).__init__(**kwargs)
        self.action_space = gym.spaces.Discrete(len(SpeedAction))
        self.scenario: str = scenario
        self.encoder = encoder
        self.ego_id: Optional[str] = None
        self.max_steps = max_steps
        self.steps = 0
        self.steps_with_overspeed = 0
        self.__last_action = None
        self.terminal_reward = terminal_reward
        self.timeout_reward = timeout_reward
        self.terminal_penalty = terminal_penalty
        self.overspeed_factor = overspeed_factor
        self.underspeed_factor = underspeed_factor
        self.acceleration_factor = acceleration_factor
        self.unrelated_collision_handling = unrelated_collision_handling
        self.punish_standing_collision = punish_standing_collision
        self.scenario_selection = scenario_selection
        self.allow_overspeed = allow_overspeed
        self.auto_start_gui = auto_start_gui
        self.super_steps = super_steps
        self.micro_step_length = micro_step_length
        self.randomize_speeds = randomize_speeds
        self.accel = accel

        # The history to replay the episode
        self._history: Optional[None, GraphSpeedHistory] = None

        self._last_scenario_idx = None

    # noinspection PyMethodMayBeStatic
    def get_action_meanings(self):
        return [e.name for e in SpeedAction]

    @staticmethod
    def create_simulation_context(scenario, mode, auto_start_gui=True, step_length=.4):
        net_file = 'Simulations/{}.net.xml'.format(scenario)
        route_file = 'Simulations/{}.rou.xml'.format(scenario)

        if scenario in SINGLE_SPAWN_SCENARIOS:
            spawner = SingleVehicleRouteSpawner()
        elif not os.path.isfile(route_file):
            route_file = None
            spawner = ManualTripSpawner()
        else:
            spawner = NoopSpawner()

        extra_args = dict()
        if mode == 'sumo-gui':
            extra_args['config_file'] = 'Simulations/render-view.sumocfg'

        return Context(net_file, route_file, mode=mode, step_length=step_length, auto_start_gui=auto_start_gui, **extra_args), spawner

    def simulate_initial_flow(self, module, spawner):
        while True:
            result = spawner.major_step(module)
            for _ in range(self.super_steps):
                spawner.micro_step(module)
                module.simulationStep()

            if result is not None:
                break

        return result

    @staticmethod
    def configure_ego_vehicle(module, ego_id):
        # Paint the ego vehicle and the usual stuff
        module.vehicle.deactivateGapControl(ego_id)
        module.vehicle.setSpeedMode(ego_id, 6)
        module.vehicle.setColor(ego_id, (200, 211, 23))
        # Important to always set the speed factor to 1.0 for the ego vehicle, agent cannot observe this factor
        # and with <1 for overspeeding disallowed may make collisions unavoidable
        module.vehicle.setSpeedFactor(ego_id, 1.0)

    def simulation_apply_action(self, module, ego_id, action):
        ego_veh = Vehicle(ego_id, module)

        if action == SpeedAction.MAINTAIN:
            applied_accel = 0
        elif action == SpeedAction.ACCEL:
            applied_accel = self.accel
        elif action == SpeedAction.DECEL:
            applied_accel = -self.accel
        else:
            raise ValueError

        delta_v = applied_accel * module.simulation.getDeltaT()
        curr_v = ego_veh.get_speed()

        set_speed = curr_v + delta_v

        if not self.allow_overspeed:
            v_allowed = ego_veh.get_allowed_speed()
            set_speed = min(v_allowed, set_speed)
        set_speed = max(0, set_speed)
        ego_veh.set_target_speed(set_speed)

        return set_speed, applied_accel, curr_v

    def sumo_init(self, mode='libsumo'):
        if type(self.scenario) == list:
            if self.scenario_selection == 'random':
                scenario = self.get_rng().choice(self.scenario)
            elif self.scenario_selection == 'sequential':
                # Select scenario sequentially, 0-indexed, only valid on 10 scenarios of which 8 with variations
                self._last_scenario_idx = 0 if self._last_scenario_idx is None else (self._last_scenario_idx + 1) % len(self.scenario)
                scenario = self.scenario[self._last_scenario_idx]
            else:
                raise ValueError(self.scenario_selection)
        else:
            scenario = self.scenario

        self._history = GraphSpeedHistory(scenario, None, None, None, None)  # Reset episode history
        ctx, self._spawner = GraphSpeedEnv.create_simulation_context(scenario, mode, auto_start_gui=self.auto_start_gui,
                                                                     step_length=self.micro_step_length)
        return ctx

    def sumo_leadin(self):
        module = self._sumo_ctx.get_traci_module()

        if self.randomize_speeds:
            edges = [e for e in module.edge.getIDList() if e[0] != ':']
            new_speeds = self.get_rng().uniform(30. / 3.6, 50. / 3.6, len(edges))
            for e, ns in zip(edges, new_speeds):
                module.edge.setMaxSpeed(e, ns)

        self._spawner.reset(module, self._sumo_ctx.get_seed(), self._history.scenario)

        tv = self.simulate_initial_flow(module, self._spawner)

        GraphSpeedEnv.configure_ego_vehicle(module, tv)

        self._history = GraphSpeedHistory(self._history.scenario, self._sumo_ctx.get_seed(), tv, [], -1)

        # Inform the encoder about reset
        self.encoder.reset()
        
        self.steps = 0
        self.steps_with_overspeed = 0
        self.__last_action = None

        return [tv]

    def sumo_step(self, action: SpeedAction):
        reward, done = 0, False

        # Update action history in place
        hist_list = self._history.actions
        hist_list.append(action)
        self._history = GraphSpeedHistory(self._history.scenario, self._history.seed, self._history.ego_id, hist_list,
                                          self._history.step + 1)

        module = self._sumo_ctx.get_traci_module()
        tv, = self._tracked_vehicles
        ego_veh = Vehicle(tv, module)

        # Only step the spawner once per real step
        self._spawner.major_step(module)

        colliding_list = []
        arrived_list = []
        v_ego_prev = ego_veh.get_speed()
        v_ego_new = v_ego_prev  # Initially set to previous to have value in case of crash

        for micro_step in range(self.super_steps):
            _, curr_acc, _ = self.simulation_apply_action(module, tv, action)
            self._spawner.micro_step(module)
            if micro_step != 0 and self._render_recorder is not None:
                self._render_recorder.prestep()
            module.simulationStep()
            colliding_list += module.simulation.getCollidingVehiclesIDList()
            arrived_list += module.simulation.getArrivedIDList()

            # Break immediately if ego vehicle no longer in vehicle list
            if tv not in module.vehicle.getIDList():
                break
            else:
                v_ego_new = ego_veh.get_speed()

        self.__last_action = action

        self.steps += 1

        ego_collision_veh = None

        if len(colliding_list) != 0:
            done = True
            if tv in colliding_list:
                if not self.punish_standing_collision and v_ego_prev <= 0 and v_ego_new <= 0:
                    # If the ego vehicle clearly is not at fault because it is not moving, handle by aborting with
                    # invalid reward (Note the vehicle cannot drive backwards)
                    # TODO: Could be problematic if stopped by end of lane maybe?
                    reward = np.nan
                    # print("Ego vehicle collided while not moving, which will not be punished", file=sys.stderr)
                else:
                    reward -= self.terminal_penalty

                    # If we can be sure who we collided with, because only 2 vehicles collided, add to info
                    if len(colliding_list) == 2:
                        ego_collision_veh = colliding_list[1 - colliding_list.index(tv)]
                        # print("Ego vehicle collided with {}".format(ego_collision_veh), file=sys.stderr)
            else:
                # It may happen that the ego vehicle is not part of the collision. Handle by aborting with invalid reward
                # print("Collision of unrelated vehicles: {}, ego {}".format(colliding_list, tv), file=sys.stderr)

                if self.unrelated_collision_handling == 'abort':
                    reward = np.nan
                elif self.unrelated_collision_handling == 'ignore':
                    done = False  # Unset done flag again
                else:
                    raise ValueError("GraphSpeedEnv.unrelated_collision_handling = {}".format(self.unrelated_collision_handling))

        if not done:
            if tv in arrived_list:
                # print("Ego vehicle arrived safely", file=sys.stderr)
                reward += self.terminal_reward
                done = True
            else:
                # Regular post action rewards
                # Penalty for not driving at the speed limit
                # TODO
                v_allowed = ego_veh.get_allowed_speed()
                reward -= abs(curr_acc) * self.acceleration_factor

                if v_allowed < v_ego_new:
                    # Speeding by more than 10 km/h
                    reward -= (v_ego_new - v_allowed) * self.overspeed_factor
                    self.steps_with_overspeed += 1
                else:
                    # Only offer distance reward penalty if not speeding, otherwise speeding reward might be exceeded with
                    # very high speed
                    reward -= (v_allowed - v_ego_new) * self.underspeed_factor

        if self.steps >= self.max_steps:
            # print("Maximum number of steps reached", file=sys.stderr)
            done = True
            reward += self.timeout_reward

        # info_msg = "{}\n{:.2f} m/s\nr = {:.2f}\n{}".format(SpeedAction(action).name, ego_veh.get_speed() if ego_veh.exists() else 0, reward, module.vehicle.getLeader(tv) if ego_veh.exists() else 0)
        info_msg = None
        info = SumoEnvInfo(info_msg, history=self._history, ego_collision_veh=ego_collision_veh)

        return reward, done, info

    def sumo_make_observation(self):
        tv, = self._tracked_vehicles
        return self.encoder(tv, self._sumo_ctx)
