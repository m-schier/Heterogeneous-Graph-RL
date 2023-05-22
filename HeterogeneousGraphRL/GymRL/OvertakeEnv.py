from .SumoEnv import SumoEnv
from ..SumoInterop import Context, Vehicle
import gym
import numpy as np
from enum import IntEnum


class OvertakeAction(IntEnum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    ACCEL = 3
    DECEL = 4


class OvertakeEnv(SumoEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, scenario='HighwayStraight', traffic='low'):
        super(OvertakeEnv, self).__init__()
        self.max_dist = 200
        self.max_tti = 20
        self.scenario = scenario
        self.traffic = traffic
        self.action_space = gym.spaces.Discrete(len(OvertakeAction))
        # Actions: left, straight, right, accelerate, deccelerate
        self.observation_space = gym.spaces.Box(low=np.array([.0, .0] + [.0] * 6 + [.0, .0] * 6),
                                                high=np.array([100., 100.] + [.1] * 6 + [self.max_dist, self.max_tti] * 6),
                                                shape=(2 + 6 + 2 * 2 * 3,))
        # Observations:
        # v_current, v_max, per lane and direction: (distance next car, time to impact next car)
        self._current_target_speed = None

    def get_action_meanings(self):
        return [e.name for e in OvertakeAction]

    def sumo_init(self, mode='libsumo'):
        net_file = 'Simulations/{}.net.xml'.format(self.scenario)
        route_file = 'Simulations/{}.rou.xml'.format(self.scenario)
        return Context(net_file, route_file, mode=mode, step_length=.4)

    def sumo_leadin(self):
        from ..SumoInterop.misc import populate
        import random

        traffic_params = {
        	'none': {'before': 0, 'after': 0, 'step_wait': 10},
            'low': {'before': 5, 'step_wait': 20},
            'high': {'before': 30, 'step_wait': 3},
        }
        
        traffic_mode = self.traffic if self.traffic != 'random' else random.choice(list(traffic_params.keys()))

        populate(**traffic_params[traffic_mode])
        ego = Vehicle('ego_0')
        ego.disable_simulation_control()
        self._current_target_speed = ego.get_speed()
        return ['ego_0']

    def sumo_step(self, action):
        from ..SumoInterop.misc import module

        traci = module()

        ego = Vehicle('ego_0')

        reward = 0.
        speed_change = 0
        done = False

        if action == OvertakeAction.LEFT:
            ego.change_left()
            reward -= 1
        elif action == OvertakeAction.RIGHT:
            ego.change_right()
            reward -= 1
        elif action == OvertakeAction.ACCEL:
            speed_change = 2 / traci.simulation.getDeltaT()
        elif action == OvertakeAction.DECEL:
            speed_change = -2 / traci.simulation.getDeltaT()

        if ego.get_speed_mode() != 6:
            # Ignore if speed mode changed, used if i.e. testing original SUMO performance
            print("WARNING: Ego speed changed by controller, will not set speed")
        else:
            ego.set_target_speed(ego.get_speed() + speed_change)

        traci.simulationStep()

        if len(traci.simulation.getCollidingVehiclesIDList()) != 0:
            reward -= 1000
            done = True
        elif not ego.exists():
            reward += 1000
            done = True
        else:
            # Regular post action rewards
            # Penalty for not driving at the speed limit
            reward -= abs((ego.get_speed() - ego.get_allowed_speed()) / ego.get_allowed_speed())

        return reward, done, None

    def sumo_make_observation(self):
        ego = Vehicle('ego_0')
        lc_encoding = [float(x) for x in ego.encode_lane_change_vector()]
        return np.array([ego.get_speed(), ego.get_allowed_speed()] + lc_encoding
                        + ego.encode_neighbors(max_dist=self.max_dist, max_tti=self.max_tti))

