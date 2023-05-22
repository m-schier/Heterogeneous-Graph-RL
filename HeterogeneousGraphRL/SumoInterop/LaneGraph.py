from enum import IntEnum


class ConnectionType(IntEnum):
    LeftChange = 0
    Link = 1 # Direct link
    RightChange = 2
    Continuation = 3  # Use if a single lane was split into multiple nodes
    IndirectLink = 4  # Link over internal lanes, only use for navigating
    Crossing = 5  # Crossing paths on intersection
    CrossingWithRow = 6
    CrossingWithYield = 7


class LaneNodeValue:
    def __init__(self, is_internal=False, length=0.0, speed_limit=None):
        self.is_internal = is_internal  # Whether lane is internal, i.e. junction lane
        self.length = length
        self.speed_limit = speed_limit

    def __repr__(self):
        return "LaneNodeValue(is_internal={})".format(self.is_internal)


class LaneLaneEdge:
    def __init__(self, type: ConnectionType, via_internal_id):
        self.via_internal_id = via_internal_id
        self.type = type

    def __repr__(self):
        return "LaneLaneEdge(via_internal_id={}, type={})".format(self.via_internal_id, self.type)


class VehicleNodeValue:
    def __init__(self, speed=0):
        self.speed = speed


class VehicleLaneEdge:
    def __init__(self, ratio):
        self.ratio = ratio


def start_stop_graph_encode_vehicles(lane_graph):
    from .misc import module
    from copy import deepcopy

    traci = module()
    result = deepcopy(lane_graph)

    for veh_id in traci.vehicle.getIDList():
        curr_lane = traci.vehicle.getLaneID(veh_id)
        pos = traci.vehicle.getLanePosition(veh_id)

        on_internal = lane_graph.has_node(curr_lane)

        result.add_node(veh_id, VehicleNodeValue(speed=traci.vehicle.getSpeed(veh_id)))

        if on_internal:
            internal_lane_val = lane_graph.get_node(curr_lane)
            if pos < internal_lane_val.length / 2:
                # Driving onto junction
                back_ratio = pos / (internal_lane_val.length / 2)
                front_ratio = 1 - back_ratio
                back = curr_lane
                front = [k for k, _ in lane_graph.iterate_incoming(curr_lane)]
                assert len(front) == 1
                front = front[0]
            else:
                # Driving out of junction
                front_ratio = (pos - (internal_lane_val.length / 2)) / (internal_lane_val.length / 2)
                back_ratio = 1 - front_ratio
                front = curr_lane
                back = [k for k, _ in lane_graph.iterate_outgoing(curr_lane)]
                assert len(back) == 1
                back = back[0]
        else:
            # Driving on some random lane
            front = curr_lane + '_ω'
            back = curr_lane + '_α'
            front_ratio = pos / lane_graph.get_node(front).length
            back_ratio = 1 - front_ratio

        result.add_edge(veh_id, front, VehicleLaneEdge(front_ratio))
        result.add_edge(veh_id, back, VehicleLaneEdge(back_ratio))

    return result


def make_start_stop_graph(lane_graph):
    from Sanddorn.Graphs import DiGraph
    lane_graph = lane_graph._graph
    result = DiGraph()

    for k, v in lane_graph.iterate_nodes():
        if v.is_internal:
            result.add_node(k, v)
        else:
            result.add_node(k + '_α', v)
            result.add_node(k + '_ω', v)
            result.add_edge(k + '_α', k + '_ω', LaneLaneEdge(ConnectionType.Continuation, None))

    for n_from, n_to, e_val in lane_graph.iterate_edges():
        if lane_graph.get_node(n_from).is_internal:
            continue

        if e_val.via_internal_id is not None:
            result.add_edge(n_from + '_ω', e_val.via_internal_id, e_val)
            result.add_edge(e_val.via_internal_id, n_to + '_α', e_val)
        elif e_val.type == ConnectionType.LeftChange or e_val.type == ConnectionType.RightChange:
            result.add_edge(n_from + '_α', n_to + '_α', e_val)
            result.add_edge(n_from + '_ω', n_to + '_ω', e_val)
        else:
            print(n_from, n_to, e_val)
            raise ValueError

    return result


def start_stop_graph_to_dot(graph, file='ss_graph.gv'):
    import graphviz

    def sanitize(x):
        if type(x) != str:
            raise TypeError(str(type(x)))
        return x.replace(':', '__')

    dot = graphviz.Digraph()

    for k, v in graph.iterate_nodes():
        id_name = sanitize(k)
        if type(v) == LaneNodeValue:
            label = "{} ({} m)\nv_max = {} m/s".format(k, v.length, v.speed_limit)
            dot.node(id_name, label=label, style="dashed" if v.is_internal else "solid")
        elif type(v) == VehicleNodeValue:
            dot.node(id_name, label=k, shape='box')
        else:
            raise ValueError

    for lane_from, lane_to, edge_val in graph.iterate_edges():
        kwargs = {}

        if type(edge_val) == LaneLaneEdge:
            if edge_val.type == ConnectionType.LeftChange:
                kwargs['label'] = 'Left'
            elif edge_val.type == ConnectionType.RightChange:
                kwargs['label'] = 'Right'
            elif edge_val.type == ConnectionType.Continuation:
                kwargs['style'] = 'bold'
        elif type(edge_val) == VehicleLaneEdge:
            kwargs['label'] = str(edge_val.ratio)
        else:
            raise ValueError(str(type(edge_val)))

        dot.edge(sanitize(lane_from), sanitize(lane_to), **kwargs)

    dot.render(file)


class LaneGraph:
    def __init__(self):
        from Sanddorn.Graphs import DiGraph
        self._graph = DiGraph()

    @staticmethod
    def decompose_lane(lane_id: str):
        assert type(lane_id) == str
        parts = lane_id.split('_')
        return '_'.join(parts[:-1]), int(parts[-1])

    @staticmethod
    def make_lane(edge_id: str, lane_index: int):
        assert type(edge_id) == str
        assert issubclass(type(lane_index), int)
        assert lane_index >= 0
        return '{}_{}'.format(edge_id, lane_index)

    @staticmethod
    def build_from_context():
        from .misc import module
        # import traci
        traci = module()

        graph = LaneGraph()

        lane_ids = set(traci.lane.getIDList())

        for lane_id in lane_ids:
            graph._graph.add_node(lane_id, LaneNodeValue(length=traci.lane.getLength(lane_id),
                                                         speed_limit=traci.lane.getMaxSpeed(lane_id)))

        for lane_id in lane_ids:
            for outgoing_info in traci.lane.getLinks(lane_id):
                approached_id, _, _, _, via_internal_id, _, _, _ = outgoing_info

                via_internal_id = via_internal_id if via_internal_id != '' else None

                if via_internal_id is not None:
                    # Set corresponding lane node to be internal
                    graph._graph.get_node(via_internal_id).is_internal = True

                    # When processing an indirect link, check that no direct links were previously added, this may happen
                    # If to major lanes are connected through a series of internal lanes
                    edge = graph._graph.try_get_edge(lane_id, approached_id)

                    if edge is not None and type(edge) == LaneLaneEdge and edge.type == ConnectionType.Link:
                        graph._graph.remove_edge(lane_id, approached_id)

                    graph._graph.add_edge(lane_id, approached_id, LaneLaneEdge(ConnectionType.IndirectLink, via_internal_id))
                    graph._graph.add_edge(lane_id, via_internal_id, LaneLaneEdge(ConnectionType.Link, None))

                    # Similarly, when adding the the second part of the connection, check that no higher priority
                    # indirect link is overwritten
                    edge = graph._graph.try_get_edge(via_internal_id, approached_id)

                    if edge is None or (edge.type != ConnectionType.IndirectLink):
                        graph._graph.add_edge(via_internal_id, approached_id, LaneLaneEdge(ConnectionType.Link, None))

        # Insert lane change transitions.
        # TODO Currently not considering lane changing into opposite traffic where allowed
        # Do this after discovering all internals
        for lane_id in lane_ids:
            edge_id, lane_index = LaneGraph.decompose_lane(lane_id)
            left_lane = LaneGraph.make_lane(edge_id, lane_index + 1)
            if left_lane in lane_ids:
                graph._graph.add_edge(lane_id, left_lane, LaneLaneEdge(ConnectionType.LeftChange, None))
            if lane_index > 0:
                right_lane = LaneGraph.make_lane(edge_id, lane_index - 1)
                if right_lane in lane_ids:
                    graph._graph.add_edge(lane_id, right_lane, LaneLaneEdge(ConnectionType.RightChange, None))

        return graph

    def remove_indirect_nav_links(self):
        for edge_from, edge_to, edge_val in list(self._graph.iterate_edges()):
            if type(edge_val) == LaneLaneEdge and edge_val.type == ConnectionType.IndirectLink:
                self._graph.remove_edge(edge_from, edge_to)

    def get_change_left(self, lane_id):
        for k, v in self._graph.iterate_outgoing(lane_id):
            if v.type == ConnectionType.LeftChange:
                return k
        return None

    def get_change_right(self, lane_id):
        for k, v in self._graph.iterate_outgoing(lane_id):
            if v.type == ConnectionType.RightChange:
                return k
        return None

    def to_dot(self, filename='graph.png'):
        import graphviz

        def sanitize(x):
            return x.replace(':', '__')

        dot = graphviz.Digraph()

        for lane_id, lane_val in self._graph.iterate_nodes():
            id_name = sanitize(lane_id)
            label = "{} ({} m)\nv_max = {} m/s".format(lane_id, lane_val.length, lane_val.speed_limit)
            dot.node(id_name, label=label, style="dashed" if lane_val.is_internal else "solid")

        for lane_from, lane_to, edge_val in self._graph.iterate_edges():
            # if self._graph.get_node(lane_from).is_internal:
            #     continue

            kwargs = {}

            if edge_val.type == ConnectionType.IndirectLink:
                continue
            elif edge_val.type == ConnectionType.LeftChange:
                kwargs['label'] = 'Left'
            elif edge_val.type == ConnectionType.RightChange:
                kwargs['label'] = 'Right'

            if edge_val.via_internal_id is None:
                dot.edge(sanitize(lane_from), sanitize(lane_to), **kwargs)
            else:
                dot.edge(sanitize(lane_from), sanitize(edge_val.via_internal_id), **kwargs)
                dot.edge(sanitize(edge_val.via_internal_id), sanitize(lane_to), **kwargs)

        dot.render(filename)
