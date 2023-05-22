import torch
import warnings

from HeterogeneousGraphRL.Representation.PointGraph import PgVehicleNode, PgVehicleRoadEdge, PgRoadRoadEdge, PointGraph, PgVehicleVehicleEdge
from enum import IntEnum


def _format_tensor(t):
    if t.shape == tuple():
        return t.numpy()
    return "[" + ", ".join(["{:.2f}".format(i) for i in t.numpy()]) + "]"


def _format_node(name, data, key):
    result = "{}:".format(name)

    for entry in data.keys():
        result += "\n{} = {}".format(entry, _format_tensor(data[entry][key]))

    return result


def encoding_to_dot(tg_data, veh_lut, road_lut):
    import graphviz

    graph = graphviz.Digraph()

    assert len(veh_lut) == tg_data['veh'].num_nodes
    assert len(road_lut) == tg_data['road'].num_nodes

    node_lut = {}

    for k, v in veh_lut.items():
        k_san = k.replace(':', '__')
        node_lut[('veh', v)] = k_san
        graph.node(k_san, label=_format_node(k, tg_data['veh'], v), shape='box')

    for k, v in road_lut.items():
        k_san = k.replace(':', '__')
        node_lut[('road', v)] = k_san
        graph.node(k_san, label=_format_node(k, tg_data['road'], v))

    edges = tg_data.metadata()[1]

    for tup in edges:
        vfrom, rel, vto = tup

        for (n_from, n_to), attr in zip(tg_data[tup].edge_index.T, tg_data[tup].edge_attrs):
            id_from = node_lut[(vfrom, int(n_from))]
            id_to = node_lut[(vto, int(n_to))]
            graph.edge(id_from, id_to, "{}:\n{}".format(rel, _format_tensor(attr)))

    return graph


class EncoderResult:
    def __init__(self, tg_data, veh_lut, road_lut, extra_data=None):
        self.tg_data = tg_data
        self.veh_lut = veh_lut
        self.road_lut = road_lut
        self.extra_data = {} if extra_data is None else extra_data

    def to_dot(self):
        return encoding_to_dot(self.tg_data, self.veh_lut, self.road_lut)


class PointGraphEncoder:
    def __init__(self, veh_node_dims, lane_node_dims, lane_lane_dims, lane_veh_dims, veh_veh_dims):
        self.veh_node_dims = veh_node_dims
        self.lane_node_dims = lane_node_dims
        self.lane_lane_dims = lane_lane_dims
        self.lane_veh_dims = lane_veh_dims
        self.veh_veh_dims = veh_veh_dims

    def encode_vehicle(self, key: str, value: PgVehicleNode):
        raise NotImplementedError

    def encode_lane(self, key, value, route_reachable, route_continuable, route_goal):
        raise NotImplementedError

    def encode_lane_lane_edge(self, node_from, node_to, edge_value: PgRoadRoadEdge):
        raise NotImplementedError

    def encode_lane_vehicle_edge(self, node_from, node_to, edge_value: PgVehicleRoadEdge):
        raise NotImplementedError

    def encode_vehicle_vehicle_edge(self, node_from, node_to, edge_value: PgVehicleVehicleEdge):
        raise NotImplementedError

    def encode_extra(self, graph, mark_vid=None):
        return {}

    def encode_for_torch(self, graph: PointGraph, mark_vid=None, undirected=False):
        import torch_geometric as tg

        data = tg.data.HeteroData()

        # Enumerate nodes of type
        kv_veh = []
        kv_road = []
        # Also create LUTs for edges
        veh_lut = {}
        road_lut = {}

        for k, v in graph.road_nodes.items():
            road_lut[k] = len(kv_road)
            kv_road.append((k, v))

        for k, v in graph.veh_nodes.items():
            veh_lut[k] = len(kv_veh)
            kv_veh.append((k, v))

        veh_data = []
        for k, v in kv_veh:
            d = self.encode_vehicle(k, v)
            assert d.shape == (self.veh_node_dims,)
            veh_data.append(d)
        veh_data = torch.stack(veh_data)

        if mark_vid is not None:
            veh_data = torch.cat([torch.zeros((len(veh_data), 1)), veh_data], 1)
            veh_data[veh_lut[mark_vid], 0] = 1
            target_mask = torch.zeros(len(veh_data), dtype=torch.bool)
            target_mask[veh_lut[mark_vid]] = True
            data['veh'].target_mask = target_mask

        data['veh'].x = veh_data

        road_data = []
        for k, v in kv_road:
            route_reachable = k in graph.route_reachable_set
            route_continuable = k in graph.route_continuable_set
            route_goal = k in graph.route_goal_set
            d = self.encode_lane(k, v, route_reachable, route_continuable, route_goal)
            assert d.shape == (self.lane_node_dims,)
            road_data.append(d)

        data['road'].x = torch.stack(road_data)

        road_joins_road_index = []
        road_joined_by_road_index = []
        road_joins_road_attrs = []
        road_joined_by_road_attrs = []
        veh_on_road_index = []
        road_has_veh_index = []
        veh_on_road_attrs = []
        road_has_veh_attrs = []
        veh_has_prio_index = []
        veh_has_prio_attrs = []

        for e_from, edges in graph.road_road_forward.items():
            for e_to, e_val in edges.items():
                road_joins_road_index.append((road_lut[e_from], road_lut[e_to]))
                f = self.encode_lane_lane_edge(e_from, e_to, e_val)
                assert f.shape == (self.lane_lane_dims,)
                road_joins_road_attrs.append(f)
                road_joined_by_road_index.append((road_lut[e_to], road_lut[e_from]))
                road_joined_by_road_attrs.append(road_joins_road_attrs[-1])

        for e_from, edges in graph.veh_road_forward.items():
            for e_to, e_val in edges.items():
                veh_on_road_index.append((veh_lut[e_from], road_lut[e_to]))
                f = self.encode_lane_vehicle_edge(e_from, e_to, e_val)
                assert f.shape == (self.lane_veh_dims,)
                veh_on_road_attrs.append(f)
                road_has_veh_index.append((road_lut[e_to], veh_lut[e_from]))
                road_has_veh_attrs.append(veh_on_road_attrs[-1])

        def encode_idx_attr(tup, idxs, attrs, attr_dim):
            nonlocal data

            if len(idxs) > 0:
                data[tup].edge_index, data[tup].edge_attrs = torch.transpose(torch.tensor(idxs, dtype=torch.int64), 0, 1), torch.stack(attrs)
                fini = torch.isfinite(data[tup].edge_attrs)
                try:
                    torch.testing.assert_allclose(fini, torch.ones_like(fini), msg=tup)
                except:
                    print(data[tup].edge_attrs, fini)
                    raise
            else:
                data[tup].edge_index, data[tup].edge_attrs = torch.empty((2, 0), dtype=torch.int64), torch.empty((0, attr_dim), dtype=torch.float32)

        encode_idx_attr(('road', 'joins', 'road'), road_joins_road_index, road_joins_road_attrs, self.lane_lane_dims)
        encode_idx_attr(('road', 'joined_by', 'road'), road_joined_by_road_index, road_joined_by_road_attrs, self.lane_lane_dims)
        encode_idx_attr(('veh', 'on', 'road'), veh_on_road_index, veh_on_road_attrs, self.lane_veh_dims)
        encode_idx_attr(('road', 'has', 'veh'), road_has_veh_index, road_has_veh_attrs, self.lane_veh_dims)
        encode_idx_attr(('veh', 'has_prio', 'veh'), veh_has_prio_index, veh_has_prio_attrs, self.veh_veh_dims)

        return EncoderResult(data, veh_lut, road_lut, extra_data=self.encode_extra(graph, mark_vid=mark_vid))


class DefaultGraphEncoder(PointGraphEncoder):
    def __init__(self, extra_encoders=None, encode_logs=False, normalize=False, max_normalized_speed=50.,
                 max_normalized_dist=200., max_normalized_length=20., encode_route=True,
                 encode_crossing_distances=False):
        veh_scalars = 5
        lane_scalars = 1
        lane_lane_scalars = 3 if encode_crossing_distances else 1
        lane_veh_scalars = 2
        veh_veh_scalars = 1
        factor = 2 if encode_logs else 1
        super(DefaultGraphEncoder, self).__init__(veh_scalars * factor + 2, lane_scalars * factor + 3,
                                                  lane_lane_scalars * factor + 8, lane_veh_scalars * factor + 1,
                                                  veh_veh_scalars * factor)
        self.extra_encoders = {} if extra_encoders is None else extra_encoders
        self.encode_logs = encode_logs
        self.encode_route = encode_route
        self.encode_crossing_distances = encode_crossing_distances
        self.normalize = normalize
        self.max_normalized_speed = max_normalized_speed
        self.max_normalized_dist = max_normalized_dist
        self.max_normalized_length = max_normalized_length

    def encode_speed_val(self, val):
        if self.normalize:
            if not -self.max_normalized_speed <= val <= self.max_normalized_speed:
                warnings.warn("The given speed of {} exceeds the encodable range [{}, {}]"
                              .format(val, -self.max_normalized_speed, self.max_normalized_speed))
            return max(-1, min(1, val / self.max_normalized_speed))
        else:
            return val

    def encode_dist_val(self, val):
        if self.normalize:
            return max(-1, min(1, val / self.max_normalized_dist))
        else:
            return val

    def encode_length_val(self, val):
        if self.normalize:
            return max(0, min(1, val / self.max_normalized_length))
        else:
            return val

    def encode_with_logs(self, scalars, cats):
        from math import log
        if self.encode_logs:
            log_scalars = [log(x) if x > 0 else -1000 for x in scalars]
            args = [scalars, log_scalars, cats]
        else:
            args = [scalars, cats]

        return torch.cat([t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32) for t in args])

    def encode_vehicle(self, key, value):
        scalar_vals = [
            value.random_token,
            self.encode_speed_val(value.speed),
            self.encode_speed_val(value.prev_speed),
            self.encode_speed_val(value.max_speed),
            self.encode_length_val(value.length),
        ]
        cat_vals = [value.signaling_left, value.signaling_right]
        return self.encode_with_logs(scalar_vals, cat_vals)

    def encode_lane(self, key, value, route_reachable, route_continuable, route_goal):
        scalar_vals = [self.encode_speed_val(value.speed_limit)]
        cat_vals = [
            float(route_reachable) if self.encode_route else 0.,
            float(route_continuable) if self.encode_route else 0.,
            float(route_goal)
        ]
        return self.encode_with_logs(scalar_vals, cat_vals)

    def encode_lane_lane_edge(self, node_from, node_to, edge_value):
        assert issubclass(type(edge_value.type), IntEnum), "Bad type {}".format(type(edge_value.type))
        assert int(edge_value.type) >= 0
        cat_vals = torch.zeros(len(type(edge_value.type)), dtype=torch.float32)
        cat_vals[int(edge_value.type)] = 1
        scalar_vals = [self.encode_dist_val(edge_value.distance)]

        if self.encode_crossing_distances:
            scalar_vals += [
                self.encode_dist_val(edge_value.own_crossing_distance),
                self.encode_dist_val(edge_value.other_crossing_distance)
            ]

        return self.encode_with_logs(scalar_vals, cat_vals)

    def encode_lane_vehicle_edge(self, node_from, node_to, edge_value):
        scalar_vals = [edge_value.ratio, self.encode_dist_val(edge_value.distance)]
        cat_vals = [edge_value.is_front]
        return self.encode_with_logs(scalar_vals, cat_vals)

    def encode_extra(self, graph, mark_vid=None):
        return {k: v(graph, mark_vid) for k, v in self.extra_encoders.items()}

    def encode_vehicle_vehicle_edge(self, node_from, node_to, edge_value: PgVehicleVehicleEdge):
        scalar_vals = [self.encode_dist_val(edge_value.distance)]
        cat_vals = []
        return self.encode_with_logs(scalar_vals, cat_vals)
