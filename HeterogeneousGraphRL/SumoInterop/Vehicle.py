from enum import Enum

from .Direction import Direction


class ChangeState(Enum):
    OKAY = 0
    INVALID = 1
    WOULD_COLLIDE = 2
    INSUFFICIENT_SPACE = 3


_signal_map = {
    'right': 0,
    'left': 1,
    'emergency': 2,
    'brake': 3,
    'front': 4,
    'fog': 5,
    'highbeam': 6,
    'reverse': 7,
    'wiper': 8,
    'door_left': 9,
    'door_right': 10,
    'emergency_blue': 11,
    'emergency_red': 12,
    'emergency_yellow': 13
}


class Vehicle:
    def __init__(self, veh_id, module):
        self.module = module
        self.id = veh_id

    @staticmethod
    def iterate(module):
        for i in module.vehicle.getIDList():
            yield Vehicle(i, module)

    def change_dir(self, direction):
        self.module.vehicle.changeLaneRelative(self.id, int(direction), .0)

    def change_left(self):
        self.module.vehicle.changeLaneRelative(self.id, 1, .0)

    def change_right(self):
        self.module.vehicle.changeLaneRelative(self.id, -1, .0)

    def can_change_dir(self, direction):
        state, state_tra_ci = self.module.vehicle.getLaneChangeState(self.id, int(direction))
        if self.module.vehicle.wantsAndCouldChangeLane(self.id, int(direction), state_tra_ci):
            # vehicle changed in the last step. state is no longer applicable
            return False
        elif state & 0b1111111111111111 == 0:
            return ChangeState.INVALID
        elif state & (1 << 13):
            return ChangeState.WOULD_COLLIDE
        elif state & (0b1111 << 9):
            return ChangeState.INSUFFICIENT_SPACE
        elif state & (0b1111 << 3):
            return ChangeState.OKAY
        else:
            raise ValueError("Failed to interpret change state: {}".format(state))

    def can_change_left(self):
        return self.can_change_dir(1)

    def can_change_right(self):
        return self.can_change_dir(-1)

    def exists(self):
        return self.id in self.module.vehicle.getIDList()

    def disable_simulation_control(self):
        self.module.vehicle.deactivateGapControl(self.id)
        self.module.vehicle.setSpeedMode(self.id, 6)  # Honor physical vehicle parameters with no safeties
        #                                             xx            No sublane changes
        #                                               xx          Ignore other drivers when fulfilling traci request
        #                                                 xx        No right drive change
        #                                                   xx      No speed gain change
        #                                                     xx    No cooperative changes
        #                                                       xx  No strategic changes
        self.module.vehicle.setLaneChangeMode(self.id, 0b000000000000)

    def get_previous_acceleration(self):
        return self.module.vehicle.getAcceleration(self.id)

    def get_best_lanes(self):
        return self.module.vehicle.getBestLanes(self.id)

    def get_lane(self):
        return self.module.vehicle.getLaneIndex(self.id)

    def get_signals(self):
        sigs = self.module.vehicle.getSignals(self.id)
        return {k: bool(sigs & (1 << v)) for k, v in _signal_map.items()}

    def get_allowed_speed(self):
        speed = self.module.vehicle.getAllowedSpeed(self.id)

        if speed == -2 ** 30:
            raise ValueError("Failed to query allowed speed for \"{}\"".format(self.id))

        return speed

    def get_acceleration_potential(self):
        return self.module.vehicle.getAccel()

    def get_deceleration_potential(self):
        return self.module.vehicle.getDecel()

    def get_speed(self):
        speed = self.module.vehicle.getSpeed(self.id)

        if speed == -2 ** 30:
            raise ValueError("Failed to query speed for \"{}\"".format(self.id))

        return speed

    def get_speed_mode(self):
        return self.module.vehicle.getSpeedMode(self.id)

    def get_speed_factor(self):
        factor = self.module.vehicle.getSpeedFactor(self.id)
        return factor

    def set_signals(self, **kwargs):
        sigs = self.module.vehicle.getSignals(self.id)
        or_mask = 0
        and_mask = 2 ** 32 - 1

        for k, v in kwargs.items():
            if v:
                or_mask = or_mask | (1 << _signal_map[k])
            else:
                and_mask = and_mask - (1 << _signal_map[k])

        new_sigs = (sigs | or_mask) & and_mask
        print(self.id, sigs, new_sigs)
        self.module.vehicle.setSignals(self.id, new_sigs)

    def set_target_speed(self, v):
        self.module.vehicle.setSpeed(self.id, v)

    def set_immediate_speed(self, v):
        traci = self.module
        old_speed_mode = traci.vehicle.getSpeedMode(self.id)
        traci.vehicle.setSpeedMode(self.id, 0b100000)  # Disregard all restrictions
        traci.vehicle.setSpeed(self.id, v)
        traci.vehicle.setSpeedMode(self.id, old_speed_mode)

    def encode_neighbors(self, max_dist=200, max_tti=20):
        traci = self.module

        v_own = traci.vehicle.getSpeed(self.id)

        def encode(others, is_forwards):
            best_id = None
            best_distance = None

            if len(others) > 0 and others[0] is not None:
                for o_id, distance in others:
                    # TODO: TraCI sometimes returns an undocumented error value of ('', -1.0) for getFollower()
                    if o_id != '' and (best_distance is None or best_distance > distance):
                        best_distance = distance
                        best_id = o_id

            if best_id is None:
                return [max_dist, max_tti]
            dist = min(max_dist, max(0, best_distance))
            v_other = traci.vehicle.getSpeed(best_id)
            v_approach = v_own - v_other if is_forwards else v_other - v_own
            if v_approach <= 0:
                tti = max_tti
            else:
                tti = min(max_tti, dist / v_approach)
            return [dist, tti]

        encoding = []
        encoding += encode(traci.vehicle.getLeftLeaders(self.id), True)
        encoding += encode([traci.vehicle.getLeader(self.id)], True)
        encoding += encode(traci.vehicle.getRightLeaders(self.id), True)
        encoding += encode(traci.vehicle.getLeftFollowers(self.id), False)
        encoding += encode([traci.vehicle.getFollower(self.id)], False)
        encoding += encode(traci.vehicle.getRightFollowers(self.id), False)
        return encoding

    def encode_lane_change_vector(self):
        """
        Query the lane change information as vector lp, sp, rp, ld, sd, rd, where l, s, r = left, straight, right;
        p, d = possible, desirable (to be on correct route)
        """
        idx = self.module.vehicle.getLaneIndex(self.id)

        if idx == -2 ** 30:
            raise ValueError("Bad lane index returned by TraCI")

        info = self.module.vehicle.getBestLanes(self.id)

        available_idxs = list(range(len(info)))

        lp = idx + 1 in available_idxs
        sp = idx in available_idxs
        rp = idx - 1 in available_idxs

        best_idx = min([abs(info[i][3]) for i in range(idx - 1, idx + 2) if i in available_idxs], default=-1)

        if best_idx == -1:
            raise ValueError

        ld = lp and info[idx + 1][3] == best_idx
        sd = sp and info[idx][3] == best_idx
        rd = rp and info[idx - 1][3] == best_idx
        return lp, sp, rp, ld, sd, rd

    def want_change(self):
        idx = self.module.vehicle.getLaneIndex(self.id)

        info = self.module.vehicle.getBestLanes(self.id)
        if info[idx][3] == 0:
            return Direction.STAY
        elif info[idx][3] > 0:
            return Direction.RIGHT
        else:
            return Direction.LEFT
