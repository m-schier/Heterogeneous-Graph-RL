import numpy as np
from typing import Optional


class Spawner:
    """
    A stateful vehicle spawner for reproducible fast route spawning in the TRACI simulation
    """

    def reset(self, module, seed, token):
        """
        Reset the spawner state at the beginning of an episode
        :param module: Traci module to be used
        :param seed: RNG seed to be used for any random operations
        :param token: Any hashable token uniquely identifying the network, i.e. file path. May be used by the
        implementation to cache state for an efficient reset.
        """
        raise NotImplementedError

    def major_step(self, module) -> Optional[str]:
        """
        Carry out a major simulation step. To be called once per environment step. Will be called before the first
        micro step call for the major step.
        :param module: Traci module to be used
        :returns: Normally `None`, a vehicle id if the "lead-in" of the simulation is completed and the agent should
        take control of the given vehicle id
        """
        raise NotImplementedError

    def micro_step(self, module):
        """
        Carry out a minor simulation step. To be called before each module.simulationStep() call.
        """
        pass


class NoopSpawner(Spawner):
    """
    Useful if the simulation comes with adequate routing
    """
    def __init__(self, steps_until_start=100):
        self.steps_until_start = steps_until_start
        self.steps = 0
        self.departed = []
        self._rng = None

    def reset(self, module, seed, token):
        self.steps = 0
        self.departed = []
        self._rng = np.random.default_rng(seed)

    def major_step(self, module):
        result = None

        if self.steps < self.steps_until_start:
            self.departed += [d for d in module.simulation.getDepartedIDList() if
                              module.vehicle.getTypeID(d) == 'DEFAULT_VEHTYPE']
            self.departed = self.departed[-10:]
        elif self.steps == self.steps_until_start:
            # Check still exists in simulation
            curr_vehs = set(module.vehicle.getIDList())
            candidates = [d for d in self.departed if d in curr_vehs]
            result = candidates[self._rng.integers(len(candidates))]
        self.steps += 1

        return result


class SingleVehicleRouteSpawner(Spawner):
    def __init__(self, steps_until_spawn=100, spawn_any_route=False):
        super(SingleVehicleRouteSpawner, self).__init__()
        self.steps_until_spawn = steps_until_spawn
        self.spawn_any_route = spawn_any_route
        self.steps = 0
        self.has_found_ego = False
        self._rng = None

    def reset(self, module, seed, token):
        self.steps = 0
        self.has_found_ego = False
        self._rng = np.random.default_rng(seed)

    def micro_step(self, module):
        vehs = module.vehicle.getIDList()
        for v in vehs:
            if module.vehicle.getTypeID(v) != 'DEFAULT_VEHTYPE':
                # Always set the speed to full allowed speed for any other vehicles, otherwise SUMO may drive slightly
                # slower, which may be picked up by an agent to deduce vehicle behavior
                module.vehicle.setSpeed(v, module.vehicle.getAllowedSpeed(v))

                # For vehicles without gap control, also disable gap control and set speed mode fixed
                if 'WITH_GAP' not in module.vehicle.getRouteID(v):
                    module.vehicle.deactivateGapControl(v)
                    module.vehicle.setSpeedMode(v, 6)

    def major_step(self, module):
        result = None
        vehs = module.vehicle.getIDList()

        if self.steps == self.steps_until_spawn:
            # Check that the simulation has not spawned any default vehicles, i.e. there is no default vehicle flow
            assert len([d for d in module.vehicle.getIDList() if
                        module.vehicle.getTypeID(d) == 'DEFAULT_VEHTYPE']) == 0

            routes = module.route.getIDList()

            if 'route_EGO' in routes:
                route = 'route_EGO'
            elif self.spawn_any_route:
                route = routes[self._rng.choice(np.arange(len(routes)))]
            else:
                raise ValueError("No EGO route found and will not spawn on any route")

            module.vehicle.addFull('VEH', route, departSpeed='max')
        elif self.steps > self.steps_until_spawn and not self.has_found_ego:
            if 'VEH' in vehs:
                result = 'VEH'
                self.has_found_ego = True
            elif self.steps > self.steps_until_spawn * 2:
                raise ValueError("Took too long to find ego vehicle")

        self.steps += 1
        return result


def _is_dead(module, edge_id):
    lane_count = module.edge.getLaneNumber(edge_id)

    for i in range(lane_count):
        if module.lane.getLinkNumber('{}_{}'.format(edge_id, i)) > 0:
            return False
    return True


class ManualTripSpawner(Spawner):
    def __init__(self):
        self._token_params = dict()
        self._time_to_spawn = None
        self._no_routes = None
        self._probabilities = None
        self._timer = None
        self._next_veh_id = None
        self._rng = None

    def reset(self, module, seed, token):
        try:
            tts, no_routes, probs = self._token_params[token]
        except KeyError:
            edges = module.edge.getIDList()
            external_edges = [e for e in edges if e[0] != ':']  # Filter out likely junction lanes

            total_road_len = .0
            for e in external_edges:
                for i in range(module.edge.getLaneNumber(e)):
                    total_road_len += module.lane.getLength('{}_{}'.format(e, i))

            # Spawn 1 car per second per 25 km road, thus invert for duration between spawns
            tts = 25000 / total_road_len

            dst_edges = [e for e in external_edges if _is_dead(module, e)]

            reachable_edges = set()

            for e in edges:
                lane_count = module.edge.getLaneNumber(e)
                for i in range(lane_count):
                    lane_id = '{}_{}'.format(e, i)
                    for linked_id, *_ in module.lane.getLinks(lane_id):
                        linked_edge = '_'.join(linked_id.split('_')[:-1])
                        reachable_edges.add(linked_edge)

            src_edges = [e for e in external_edges if e not in reachable_edges]

            # print("Have {} source and {} destination edges".format(len(src_edges), len(dst_edges)))

            # Add all imaginable routes#
            rq = _ReachabilityQuery(module)
            no_routes = 0
            weights = []
            for e1 in src_edges:
                for e2 in dst_edges:
                    if e1 == e2 or not rq.is_reachable(e1, e2):
                        continue
                    r_id = 'ROUTE_{}'.format(no_routes)
                    module.route.add(r_id, [e1, e2])
                    weights.append(
                        module.edge.getLaneNumber(e1) * module.edge.getLaneNumber(e2) * module.lane.getMaxSpeed(
                            '{}_0'.format(e1)) * module.lane.getMaxSpeed('{}_0'.format(e2)))
                    # print("Added route from {} to {} as {}".format(e1, e2, r_id))
                    no_routes += 1

            # print("Added {} routes".format(no_routes))

            probs = np.array(weights) * 1.
            probs /= np.sum(probs)

            self._token_params[token] = tts, no_routes, probs

        self._time_to_spawn, self._no_routes, self._probabilities = tts, no_routes, probs

        self._timer = 0.
        self._next_veh_id = 0
        self._rng = np.random.default_rng(seed)

    def major_step(self, module):
        raise NotImplementedError("Class has not been updated to reflect Spawner changes")
        self._timer += module.simulation.getDeltaT()

        while self._timer > self._time_to_spawn:
            v_id = 'VEH_{}'.format(self._next_veh_id)
            idx = self._rng.choice(np.arange(self._no_routes), p=self._probabilities)
            route_id = 'ROUTE_{}'.format(idx)
            # print("Add vehicle", v_id, "on route", route_id)
            module.vehicle.add(v_id, route_id, departSpeed="max")
            self._next_veh_id += 1
            self._timer -= self._time_to_spawn


class _ReachabilityQuery:
    def __init__(self, module):
        self._outlinks = dict()
        self._module = module
        self._checked = dict()

    def is_reachable(self, start_edge, stop_edge):
        module = self._module

        if start_edge in self._checked:
            return stop_edge in self._checked[start_edge]

        frontier = [start_edge]
        visited = set()

        while len(frontier) > 0:
            current = frontier.pop(0)

            if current in visited:
                continue

            visited.add(current)

            try:
                outlinks = self._outlinks[current]
            except KeyError:
                outlinks = set()
                for i in range(module.edge.getLaneNumber(current)):
                    outlinks.update(
                        ['_'.join(t.split('_')[:-1]) for t, *_ in module.lane.getLinks('{}_{}'.format(current, i))])
                outlinks = list(outlinks)
                self._outlinks[current] = outlinks

            frontier += outlinks

        self._checked[start_edge] = visited
        return stop_edge in visited
