from HeterogeneousGraphRL.SumoInterop import Context
from collections import defaultdict
from HeterogeneousGraphRL.Representation.NetworkQuery import NetworkQuery
from enum import IntEnum
from random import random
from typing import Tuple, Dict, Set
import numpy as np


class PgRoadNode:
    def __init__(self, speed_limit, is_internal=False, lane_length=None, xy=None):
        self.speed_limit = speed_limit
        self.is_internal = is_internal
        self.lane_length = lane_length  # Lane length attribute more required internally
        self.xy = xy  # XY position should only be used for visualization

    def __repr__(self):
        return "PgRoadNode(is_internal = {}, speed_limit = {}, lane_length = {})".format(self.is_internal,
                                                                                         self.speed_limit,
                                                                                         self.lane_length)


class PgVehicleNode:
    def __init__(self, speed, max_speed, prev_speed, xy=None, angle=None, signaling_left=False, signaling_right=False, length=0):
        self.speed = speed
        self.max_speed = max_speed
        self.xy = xy
        self.angle = angle
        self.signaling_left = signaling_left
        self.signaling_right = signaling_right
        self.prev_speed = prev_speed
        self.length = length
        self.random_token = random()  # Random token in range [0, 1). Can be used for identification tasks

    @staticmethod
    def from_id(vid, prev_node=None, module=None):
        module = Context.get_current_context().get_traci_module() if module is None else module
        signals = module.vehicle.getSignals(vid)
        speed = module.vehicle.getSpeed(vid)
        length = module.vehicle.getLength(vid)
        prev_speed = prev_node.speed if prev_node is not None else speed
        return PgVehicleNode(speed=speed, max_speed=module.vehicle.getMaxSpeed(vid),
                             xy=module.vehicle.getPosition(vid), angle=module.vehicle.getAngle(vid),
                             signaling_left=signals & 2 == 2, signaling_right=signals & 1 == 1, prev_speed=prev_speed,
                             length=length)


class PgVehicleRoadEdge:
    def __init__(self, ratio, distance, is_front):
        """
        Create new edge from vehicle node to road node. Each vehicle has two such edges, one to the next road node and
        one to the previous one.
        :param ratio: The ratio in range [0, 1] to the node
        :param is_front: True if connected to the next road node, false if connected to the previous road node
        """
        self.ratio = ratio
        self.distance = distance
        self.is_front = is_front

    def __repr__(self):
        return "PgVehicleRoadEdge(ratio = {}, distance = {}, is_front = {})".format(self.ratio, self.distance,
                                                                                    self.is_front)


class PgVehicleVehicleEdge:
    def __init__(self, distance):
        self.distance = distance


class PgRoadRoadType(IntEnum):
    LeftNeighbor = 0
    RightNeighbor = 1
    Continuation = 2  # Use if a single lane was split into multiple nodes
    CrossingWithRow = 3  # The from node has priority over the to node
    CrossingWithYield = 4  # The to node has priority over the from node
    LinkLeft = 5
    LinkStraight = 6
    LinkRight = 7

    def is_link(self):
        """
        Return true if this type marks any forward link a vehicle may traverse, that is both entering traversing or
        exiting internal junction lanes, but also the "continuation" from an alpha to an omega lane node
        """
        return self == PgRoadRoadType.LinkLeft or self == PgRoadRoadType.LinkStraight or \
               self == PgRoadRoadType.LinkRight or self == PgRoadRoadType.Continuation

    def is_junction_link(self):
        """
        Return true if this type marks a forward link through a junction, but not a forward link on a continuation edge
        """
        return self == PgRoadRoadType.LinkLeft or self == PgRoadRoadType.LinkRight or \
               self == PgRoadRoadType.LinkStraight


class PgRoadRoadEdge:
    def __init__(self, distance, ctype: PgRoadRoadType, own_crossing_distance=0., other_crossing_distance=0.):
        if type(ctype) != PgRoadRoadType:
            raise TypeError

        self.distance = distance
        self.own_crossing_distance = own_crossing_distance
        self.other_crossing_distance = other_crossing_distance
        self.type = ctype

    def __repr__(self):
        return "{}(distance = {}, type = {})".format(self.__class__.__name__, self.distance, self.type)


_PG_INTERNAL_ALLOWED_TYPES = {PgRoadRoadType.LinkRight, PgRoadRoadType.LinkLeft, PgRoadRoadType.LinkStraight}


def _pg_defaultdict_init():
    return dict()


class PointGraph:
    def __init__(self, time_stamp=None, sim_id=None):
        # Caution: When modifying members, must modify all `with_*` functions!
        self.time_stamp = time_stamp  # Simulation time stamp, may be used to assert synchrone
        self.sim_id = sim_id  # Simulation identifier, may be used to assert synchrone
        self.road_nodes: Dict[str, PgRoadNode] = dict()
        self.veh_nodes: Dict[str, PgVehicleNode] = dict()
        self.road_road_forward = dict()
        self.road_road_backward = dict()
        self.veh_road_forward = dict()
        self.veh_road_backward = defaultdict(_pg_defaultdict_init)  # Be lazy here for easy copying without vehicles

        self.route_reachable_set = set()
        self.route_continuable_set = set()
        self.route_goal_set = set()

    def copy_share_road_and_route(self):
        """
        Create a copy of this point graph which shares the road and route information but discards vehicle information
        """
        copy = PointGraph()
        copy.sim_id = self.sim_id
        copy.road_nodes = self.road_nodes
        copy.road_road_forward = self.road_road_forward
        copy.road_road_backward = self.road_road_backward
        copy.route_reachable_set = self.route_reachable_set
        copy.route_continuable_set = self.route_continuable_set
        copy.route_goal_set = self.route_goal_set
        return copy

    def add_road_node(self, key, value):
        assert key not in self.road_nodes
        assert type(value) == PgRoadNode
        self.road_nodes[key] = value
        self.road_road_forward[key] = dict()
        self.road_road_backward[key] = dict()
        self.veh_road_backward[key] = dict()

    def add_veh_node(self, key, value):
        assert key not in self.veh_nodes
        assert type(value) == PgVehicleNode
        self.veh_nodes[key] = value
        self.veh_road_forward[key] = dict()

    def add_road_road_edge(self, rfrom, rto, value):
        assert rto not in self.road_road_forward[rfrom]
        assert type(value) == PgRoadRoadEdge
        self.road_road_forward[rfrom][rto] = value
        self.road_road_backward[rto][rfrom] = value

    def add_veh_road_edge(self, vfrom, rto, value):
        assert rto not in self.veh_road_forward[vfrom]
        assert type(value) == PgVehicleRoadEdge
        self.veh_road_forward[vfrom][rto] = value
        self.veh_road_backward[rto][vfrom] = value

    @staticmethod
    def from_query(nq: NetworkQuery):
        pg = PointGraph()

        # Add all lanes as points
        for lane_id, lane_val in nq.lanes.items():
            edge = nq.edges[lane_val.parent]

            if edge.internal:
                x = (lane_val.shape[0][0] + lane_val.shape[-1][0]) / 2
                y = (lane_val.shape[0][1] + lane_val.shape[-1][1]) / 2
                value = PgRoadNode(speed_limit=lane_val.speed, is_internal=True, lane_length=lane_val.length, xy=(x, y))
                pg.add_road_node(lane_id, value)
            else:
                v1 = PgRoadNode(speed_limit=lane_val.speed, is_internal=False, lane_length=lane_val.length,
                                xy=lane_val.shape[0])
                v2 = PgRoadNode(speed_limit=lane_val.speed, is_internal=False, lane_length=lane_val.length,
                                xy=lane_val.shape[-1])
                pg.add_road_node(lane_id + '_α', v1)
                pg.add_road_node(lane_id + '_ω', v2)

                # Also add edge
                edge = PgRoadRoadEdge(distance=lane_val.length, ctype=PgRoadRoadType.Continuation)
                pg.add_road_road_edge(lane_id + '_α', lane_id + '_ω', edge)

        # Connect according to left/right lane changing
        for edge_id, edge_val in nq.edges.items():
            if edge_val.internal:
                continue  # Cannot change lane on internals

            i = 0
            while i in edge_val.lanes and i + 1 in edge_val.lanes:
                pg.add_road_road_edge(edge_val.lanes[i] + '_α', edge_val.lanes[i+1] + '_α', PgRoadRoadEdge(distance=0, ctype=PgRoadRoadType.LeftNeighbor))
                pg.add_road_road_edge(edge_val.lanes[i] + '_ω', edge_val.lanes[i+1] + '_ω', PgRoadRoadEdge(distance=0, ctype=PgRoadRoadType.LeftNeighbor))
                pg.add_road_road_edge(edge_val.lanes[i+1] + '_α', edge_val.lanes[i] + '_α', PgRoadRoadEdge(distance=0, ctype=PgRoadRoadType.RightNeighbor))
                pg.add_road_road_edge(edge_val.lanes[i+1] + '_ω', edge_val.lanes[i] + '_ω', PgRoadRoadEdge(distance=0, ctype=PgRoadRoadType.RightNeighbor))
                i += 1

        # Connect direct links
        for from_id, from_val in nq.lanes.items():
            from_edge = nq.edges[from_val.parent]

            from_node = from_id if from_edge.internal else from_id + '_ω'
            from_dist = from_val.length / 2 if from_edge.internal else 0

            for to_id, direction in from_val.next_lanes:
                to_val = nq.lanes[to_id]
                to_edge = nq.edges[to_val.parent]
                to_node = to_id if to_edge.internal else to_id + '_α'
                to_dist = to_val.length / 2 if to_edge.internal else 0

                # TODO: What's the difference in capital letters?
                if direction == 'l' or direction == 'L':
                    ctype = PgRoadRoadType.LinkLeft
                elif direction == 'r' or direction == 'R':
                    ctype = PgRoadRoadType.LinkRight
                elif direction == 's':
                    ctype = PgRoadRoadType.LinkStraight
                else:
                    raise ValueError("Unsupported direction: {}".format(direction))
                pg.add_road_road_edge(from_node, to_node, PgRoadRoadEdge(from_dist + to_dist, ctype=ctype))

        return pg

    def add_crossing(self, n_from, n_to, row, own_dist=0., other_dist=0.):
        ctype = PgRoadRoadType.CrossingWithRow if row else PgRoadRoadType.CrossingWithYield
        self.add_road_road_edge(n_from, n_to, PgRoadRoadEdge(distance=0, ctype=ctype,
                                                             own_crossing_distance=own_dist,
                                                             other_crossing_distance=other_dist))

    def get_external_upcoming(self, external_node):
        """
        Get the list of upcoming internal nodes of an external omega node
        """
        changes = {PgRoadRoadType.LeftNeighbor, PgRoadRoadType.RightNeighbor}
        nodes = [k for k, v in self.road_road_forward[external_node].items() if v.type not in changes]

        assert all((self.road_nodes[n].is_internal for n in nodes))
        return nodes

    def get_internal_row_internals(self, internal_node: str):
        """
        Get the internals that have RoW over the given internal
        """
        return [k for k, v in self.road_road_forward[internal_node].items() if v.type == PgRoadRoadType.CrossingWithYield]

    def get_internal_yield_internals(self, internal_node: str):
        """
        Get the internals that the given internal has RoW over
        """
        return [k for k, v in self.road_road_forward[internal_node].items() if v.type == PgRoadRoadType.CrossingWithRow]

    def get_internal_incoming(self, internal_edge):
        """
        Get the incoming previous road node of an internal lane
        """
        nodes = [(k, v) for k, v in self.road_road_backward[internal_edge].items() if v.type in _PG_INTERNAL_ALLOWED_TYPES]
        assert len(nodes) == 1
        return nodes[0][0]

    def get_internal_outgoing(self, internal_edge):
        """
        Get the outgoing next road node of an internal lane
        """
        nodes = [n for n in self.road_road_forward[internal_edge].items() if n[1].type in _PG_INTERNAL_ALLOWED_TYPES]
        assert len(nodes) == 1
        return nodes[0][0]

    def get_vehicles_on_road(self, rid):
        """
        Get a list of (veh_id, veh_road_edge) for all vehicles linked (either front or back) to the given road
        """
        return list(self.veh_road_backward[rid].items())

    def get_vehicle_roads(self, vid: str) -> Tuple[str, PgVehicleRoadEdge, str, PgVehicleRoadEdge]:
        l = list(self.veh_road_forward[vid].items())

        # TODO: Fails with inter vehicle relations
        assert len(l) == 2
        assert l[0][1].is_front != l[1][1].is_front

        front_idx = 0 if l[0][1].is_front else 1

        assert l[front_idx][1].is_front

        return l[front_idx][0], l[front_idx][1], l[1 - front_idx][0], l[1 - front_idx][1]

    def try_get_left_change(self, rid):
        return self._try_get_change(rid, PgRoadRoadType.LeftNeighbor)

    def try_get_right_change(self, rid):
        return self._try_get_change(rid, PgRoadRoadType.RightNeighbor)

    def _try_get_change(self, rid, ctype):
        res = []

        for k, v in self.road_road_forward[rid].items():
            if v.type == ctype:
                res.append(k)

        assert len(res) < 2

        if not res:
            return None
        else:
            return res[0]

    def try_get_vehicle(self, vid):
        try:
            return self.veh_nodes[vid]
        except KeyError:
            return None

    def add_vehicle(self, vid, node, edge_depart, edge_arrive, ratio_depart, ratio_arrive, dist_depart, dist_arrive):
        assert abs(ratio_arrive + ratio_depart - 1) < 1e-5
        assert abs(dist_depart + dist_arrive - self.road_road_forward[edge_depart][edge_arrive].distance) < .01

        self.add_veh_node(vid, node)
        self.add_veh_road_edge(vid, edge_depart, PgVehicleRoadEdge(ratio_depart, dist_depart, False))
        self.add_veh_road_edge(vid, edge_arrive, PgVehicleRoadEdge(ratio_arrive, dist_arrive, True))

    def with_encoded_route_info(self, reachable, continuable, goal):
        """
        Returns a copy of this point graph with sharing the road and vehicle graph but with modified route info
        :param reachable: Iterable of reachable nodes
        :param continuable: Iterable of continuable nodes
        :param goal: Iterable of goal nodes
        :return: Copy of point graph
        """

        copy = PointGraph()
        copy.sim_id = self.sim_id
        copy.time_stamp = self.time_stamp
        copy.road_nodes = self.road_nodes
        copy.veh_nodes = self.veh_nodes
        copy.road_road_forward = self.road_road_forward
        copy.road_road_backward = self.road_road_backward
        copy.veh_road_forward = self.veh_road_forward
        copy.veh_road_backward = self.veh_road_backward

        copy.route_reachable_set = set(reachable)
        copy.route_continuable_set = set(continuable)
        copy.route_goal_set = set(goal)

        return copy

    def find_lerps(self, edge, position):
        # Does this really always hold?
        # TODO: No it doesn't
        on_internal = edge[0] == ':'

        # Let a be distance from last node and t be total distance
        a, t = None, None

        # Let a_node, b_node be last and next node name respectively
        a_node, b_node = None, None

        if not on_internal:
            a_node = edge + '_α'
            b_node = edge + '_ω'
            t = self.road_road_forward[a_node][b_node].distance
            a = position
        else:
            inc_name = self.get_internal_incoming(edge)
            out_name = self.get_internal_outgoing(edge)
            inc_val = self.road_nodes[inc_name]
            out_val = self.road_nodes[out_name]

            own_length = self.road_nodes[edge].lane_length
            p_rel = position / own_length

            if p_rel < .5:
                a_node = inc_name
                b_node = edge
                if inc_val.is_internal:
                    a = inc_val.lane_length / 2 + position
                    t = inc_val.lane_length / 2 + own_length / 2
                else:
                    a = position
                    t = own_length / 2
            else:
                a_node = edge
                b_node = out_name
                if out_val.is_internal:
                    a = position - own_length / 2
                    t = own_length / 2 + out_val.lane_length / 2
                else:
                    a = position - own_length / 2
                    t = own_length / 2

        assert 0 <= a <= t, "0 < {} < {} for {}@{}".format(a, t, edge, position)

        return a_node, b_node, a / t, 1 - a / t, a, t - a

    def to_dot(self, filename='point_graph.dot', engine='neato', use_xy=True, comment=None, veh_colors=None):
        import graphviz
        from HeterogeneousGraphRL.Representation.PointGraphRendering import add_road_node, add_veh_node, add_road_road_edge, add_veh_road_edge

        if veh_colors is None:
            veh_colors = dict()

        dot = graphviz.Digraph(engine=engine, comment=comment)

        for node_id in self.road_nodes.keys():
            add_road_node(dot, self, node_id, use_xy=use_xy)

        for node_id, node_val in self.veh_nodes.items():
            color = veh_colors[node_id] if node_id in veh_colors else None
            add_veh_node(dot, self, node_id, use_xy=use_xy, color=color)

        for node_from, edges in self.road_road_forward.items():
            for node_to in edges.keys():
                add_road_road_edge(dot, self, node_from, node_to, forward=True)

        for node_from, edges in self.veh_road_forward.items():
            for node_to in edges.keys():
                add_veh_road_edge(dot, self, node_from, node_to)

        dot.render(filename)


def point_graph_route_from_traci(pgraph: PointGraph, nq: NetworkQuery, traci_route):
    """
    Calculate the routing information of a point graph based on the traci edge route.
    :param pgraph: Point graph
    :param nq: Network query the point graph was constructed from
    :param traci_route: The list of edges the vehicle is routed over as given by traci
    :return: The set of reachable nodes, set of continuable nodes and set of goal nodes. All nodes which the vehicle
    may be linked to while traversing the route are considered reachable. All nodes the vehicle may have a front pointer
    to while traversing the route while not having to exercise a lane change before advancing the front pointer are
    considered continuable.
    """
    reachable_nodes = set()
    continuable_nodes = set()
    goal_nodes = set()

    for start, stop in zip(traci_route, traci_route[1:]):
        start_lanes = nq.edges[start].lanes.values()
        stop_lanes = nq.edges[stop].lanes.values()

        have_junction_crossing = False

        for al in start_lanes:
            reachable_nodes.add(al + '_α')
            reachable_nodes.add(al + '_ω')

            for bl in stop_lanes:
                try:
                    route = point_graph_find_route_through_junction(pgraph, al + '_ω', bl + '_α')
                    reachable_nodes.update(route)
                    continuable_nodes.update(route)
                    continuable_nodes.add(al + '_α')
                    have_junction_crossing = True
                except ValueError:
                    pass

        if not have_junction_crossing:
            raise ValueError("Failed to find a crossing from {} to {}".format(start, stop))

    for al in nq.edges[traci_route[-1]].lanes.values():
        reachable_nodes.add(al + '_α')
        reachable_nodes.add(al + '_ω')
        goal_nodes.add(al + '_α')
        goal_nodes.add(al + '_ω')
        # On the exit edge accept all as continuable (because we will terminate)
        # TODO: Should arrival lane be honored?
        continuable_nodes.add(al + '_α')
        continuable_nodes.add(al + '_ω')

    assert all((r in pgraph.road_nodes for r in reachable_nodes))
    assert all((r in pgraph.road_nodes for r in continuable_nodes))
    return reachable_nodes, continuable_nodes, goal_nodes


def point_graph_find_route_through_junction(pgraph: PointGraph, from_road, to_road):
    assert from_road != to_road

    frontier = [from_road]
    came_from = {from_road: None}

    def backtrace():
        result = [to_road]

        while result[-1] != from_road:
            result.append(came_from[result[-1]])

        return result[::-1]

    while len(frontier) > 0:
        current = frontier.pop(0)

        for node, edge in pgraph.road_road_forward[current].items():
            if not edge.type.is_junction_link() or node in came_from:
                continue

            came_from[node] = current

            if node == to_road:
                return backtrace()

            frontier.append(node)

    raise ValueError("No route from {} to {}".format(from_road, to_road))


def point_graph_route(pgraph: PointGraph, from_veh, to_veh, allow_backward=True):
    assert from_veh != to_veh

    frontier = [from_veh]
    came_from = {}

    def backtrace():
        result = [(to_veh, None)]

        while result[-1][0] != from_veh:
            result.append(came_from[result[-1][0]])

        return result[::-1]

    target_road_nodes = set(pgraph.veh_road_forward[to_veh].keys())

    search_collection = pgraph.veh_road_forward  # At first search vehicle->road edge

    while len(frontier) > 0:
        current = frontier.pop(0)

        for node in search_collection[current].keys():
            if node in came_from:
                continue
            came_from[node] = current, True

            if node in target_road_nodes:
                came_from[to_veh] = node, False  # Flow direction is always inverse when going to out vehicle
                return backtrace()

            frontier.append(node)

        if allow_backward and search_collection == pgraph.road_road_forward:
            for node in pgraph.road_road_backward[current].keys():
                # For backward iteration always expect road->road edge, don't iterate backwards veh->road edges
                if node in came_from:
                    continue

                came_from[node] = current, False

                if node in target_road_nodes:
                    came_from[to_veh] = node, False  # Flow direction is always inverse when going to out vehicle
                    return backtrace()

                frontier.append(node)

        search_collection = pgraph.road_road_forward  # After first iteration always move along road->road edges

    raise ValueError("No route from {} to {}".format(from_veh, to_veh))


def __make_prioritized_road_set(pgraph: PointGraph, to_veh: str) -> Set[str]:
    # For the to-vehicle, calculate the prioritized road node set:
    # If from the forward and backward node of the vehicle, exactly one is junction and the other non-junction, set
    #   only the junction as prioritized.
    # Else if both are non-junction and the forward is followed by juntions, set all following junction nodes as
    #   prioritized which match the vehicles indicators.

    f_name, _, b_name, _ = pgraph.get_vehicle_roads(to_veh)
    f_val, b_val = pgraph.road_nodes[f_name], pgraph.road_nodes[b_name]
    veh_val = pgraph.veh_nodes[to_veh]

    result = set()

    if f_val.is_internal != b_val.is_internal:
        if f_val.is_internal:
            return {f_name}
        else:
            return {b_name}
    elif not f_val.is_internal and not b_val.is_internal:
        # Check that front has nodes
        front_outs = list(pgraph.road_road_forward[f_name].items())

        if len(front_outs) > 0 and pgraph.road_nodes[front_outs[0][0]].is_internal:
            for k, v in front_outs:
                if veh_val.signaling_left:
                    if v.type == PgRoadRoadType.LinkLeft:
                        result.add(k)
                elif veh_val.signaling_right:
                    if v.type == PgRoadRoadType.LinkRight:
                        result.add(k)
                else:
                    if v.type == PgRoadRoadType.LinkStraight:
                        result.add(k)

    return result


def point_graph_route_weighted(pgraph: PointGraph, from_veh, to_veh, allow_backward=True):
    import heapq

    assert from_veh != to_veh

    to_priority = __make_prioritized_road_set(pgraph, to_veh)

    # Distance, Random Scalar, Node Id
    # Random scalar used such that bfs is random
    # TODO: What would be a good way of handling multiple paths?
    frontier = [(0., 0., from_veh)]
    came_from = {}

    def backtrace():
        result = [(to_veh, None)]

        while result[-1][0] != from_veh:
            result.append(came_from[result[-1][0]])

        return result[::-1]

    def make_distance(cd, node_id):
        res = cd + 1
        if node_id in pgraph.route_reachable_set:
            res -= .8
        if node_id in to_priority:
            res -= .1
        return res

    target_road_nodes = set(pgraph.veh_road_forward[to_veh].keys())

    search_collection = pgraph.veh_road_forward  # At first search vehicle->road edge

    while len(frontier) > 0:
        current_distance, _, current = heapq.heappop(frontier)

        for node in search_collection[current].keys():
            if node in came_from:
                continue
            came_from[node] = current, True

            if node in target_road_nodes:
                came_from[to_veh] = node, False  # Flow direction is always inverse when going to out vehicle
                return backtrace()

            heapq.heappush(frontier, (make_distance(current_distance, node), np.random.random(), node))

        if allow_backward and search_collection == pgraph.road_road_forward:
            for node in pgraph.road_road_backward[current].keys():
                # For backward iteration always expect road->road edge, don't iterate backwards veh->road edges
                if node in came_from:
                    continue

                came_from[node] = current, False

                if node in target_road_nodes:
                    came_from[to_veh] = node, False  # Flow direction is always inverse when going to out vehicle
                    return backtrace()

                heapq.heappush(frontier, (make_distance(current_distance, node), np.random.random(), node))

        search_collection = pgraph.road_road_forward  # After first iteration always move along road->road edges

    raise ValueError("No route from {} to {}".format(from_veh, to_veh))


def point_graph_from_route(pgraph: PointGraph, route):
    res = PointGraph()

    nodes, dirs = zip(*route)

    res.add_road_node(nodes[0], pgraph.road_nodes[nodes[0]])

    for fr, to, forward in zip(nodes, nodes[1:], dirs):
        res.add_road_node(to, pgraph.road_nodes[to])

        if forward:
            res.add_road_road_edge(fr, to, pgraph.road_road_forward[fr][to])
        else:
            res.add_road_road_edge(to, fr, pgraph.road_road_forward[to][fr])

    return res
