import torch
from HeterogeneousGraphRL.Representation.PointGraph import point_graph_route, point_graph_route_weighted, PointGraph
from enum import IntEnum
from typing import List, Optional, Tuple


def map_packed(pack_seq: torch.nn.utils.rnn.PackedSequence, fn) -> torch.nn.utils.rnn.PackedSequence:
    output = fn(pack_seq.data)
    assert output.shape[:-1] == pack_seq.data.shape[:-1]

    # Torch says not to do this but doesn't provide a way to remap packed sequences
    return torch.nn.utils.rnn.PackedSequence(output, *pack_seq[1:])


class NoopEdgePredictorModule(torch.nn.Module):
    def __init__(self, output_features=16):
        super(NoopEdgePredictorModule, self).__init__()
        self.output_features = output_features

    def forward(self, *args):
        batch_size = args[0].shape[0]

        return torch.zeros((batch_size, self.output_features), dtype=args[0].dtype, device=args[0].device)


class RecurrentEdgePredictorModule2(torch.nn.Module):
    def __init__(self, start_features=3, mid_features=26, end_features=7, hidden_features=None,
                 output_features=16, recurrent_arch='lstm'):
        super(RecurrentEdgePredictorModule2, self).__init__()

        if hidden_features is None:
            # When using default hidden_features, select more for rnn for comparable total parameters
            if recurrent_arch.startswith('lstm'):
                hidden_features = 64
            elif recurrent_arch.startswith('rnn'):
                hidden_features = 106
            else:
                raise ValueError(recurrent_arch)

        self.hidden_features = hidden_features

        self.start_encoder = torch.nn.Linear(start_features, hidden_features)
        self.mid_encoder = torch.nn.Linear(mid_features, hidden_features)
        self.end_encoder = torch.nn.Linear(end_features, hidden_features)

        if output_features != 0:
            self.output_encoder = torch.nn.Linear(hidden_features * 3, output_features)
        else:
            self.output_encoder = None

        if recurrent_arch == 'lstm':
            self.lstm = torch.nn.LSTM(input_size=hidden_features, hidden_size=hidden_features)
        elif recurrent_arch == 'rnn_relu':
            self.lstm = torch.nn.RNN(input_size=hidden_features, hidden_size=hidden_features, nonlinearity='relu')
        else:
            raise ValueError(recurrent_arch)

    def __mid_encode(self, x):
        return torch.relu(self.mid_encoder(x))

    def collate(self, starts, mids, mid_lens, ends):
        non_empty = mid_lens > 0  # Skip empty road sequences (e.g. vehicles on same road, since LSTM cannot handle)
        if torch.any(non_empty):
            packed_seq = torch.nn.utils.rnn.pack_padded_sequence(mids[non_empty], mid_lens[non_empty].cpu(),
                                                                 batch_first=True, enforce_sorted=False)
        else:
            packed_seq = None
        return starts, packed_seq, non_empty, ends

    def forward(self, starts, mids, mid_lens, ends):
        # Slightly confusing due to backwards compatibility, but check second argument to determine if collated
        if mids is None or isinstance(mids, torch.nn.utils.rnn.PackedSequence):
            packed_seq, non_empty = mids, mid_lens
        else:
            starts, packed_seq, non_empty, ends = self.collate(starts, mids, mid_lens, ends)

        start_enc = torch.relu(self.start_encoder(starts))
        end_enc = torch.relu(self.end_encoder(ends))

        mid_outputs = torch.zeros_like(start_enc)  # Should be Batchsize x HiddenFeatures

        if packed_seq is not None:
            packed_seq = map_packed(packed_seq, self.__mid_encode)
            packed_lstm, _ = self.lstm(packed_seq)
            padded_lstm, pad_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm, batch_first=True)
            mid_outputs[non_empty] = torch.relu(padded_lstm[torch.arange(len(padded_lstm)), pad_lens - 1])

        x = torch.cat([start_enc, mid_outputs, end_enc], dim=-1)

        if self.output_encoder is not None:
            return torch.relu(self.output_encoder(x))
        else:
            return x


class EdgePredictorModuleHandcrafted(torch.nn.Module):
    """
    A standard MLP is used for handcrafted edge features
    """
    def __init__(self, features=None):
        from torch.nn import Sequential, ReLU, Linear

        super(EdgePredictorModuleHandcrafted, self).__init__()

        if features is None:
            features = [256, 128, 16]  # Selected to be similar in parameter count to REP2

        last = 4
        l = []

        for f in features:
            l.append(Linear(last, f))
            l.append(ReLU())
            last = f

        self.enc = Sequential(*l)

    def forward(self, batch):
        return self.enc(batch)


class Relation(IntEnum):
    LeftBlocker = 0
    RightBlocker = 1
    RightOfWay = 2
    Leader = 3


def encode_relation_handcrafted(pgraph: PointGraph, own_id, other_id, encoder):
    import numpy as np

    own_veh = pgraph.veh_nodes[own_id]
    other_veh = pgraph.veh_nodes[other_id]

    ego_x, ego_y = own_veh.xy
    ego_angle = own_veh.angle / 180 * np.pi
    ego_speed = own_veh.speed

    def make_v(speed, angle):
        return np.array([np.sin(angle), np.cos(angle), 0]) * speed

    # Transform from the world coordinate system to the ego vehicle coordinate system (therefore, position and roation inverted)
    # World coordinate system is first quadrant but positive rotation is CW (x away from y)
    # Vehicle coordinate system is y towards front, x towards right, positive rotation CW
    transform = np.array([
        [np.cos(ego_angle), -np.sin(ego_angle), 0],
        [np.sin(ego_angle), np.cos(ego_angle), 0],
        [0, 0, 1]
    ]) @ np.array([
        [1, 0, -ego_x],
        [0, 1, -ego_y],
        [0, 0, 1]
    ])

    np.testing.assert_allclose(transform @ np.array([ego_x, ego_y, 1]), np.array([0, 0, 1]), atol=1e-8)

    other_x, other_y = other_veh.xy
    other_angle = other_veh.angle / 180 * np.pi
    other_speed = other_veh.speed

    dx, dy, _ = (transform @ np.array([other_x, other_y, 1]))
    own_v = make_v(ego_speed, ego_angle)
    other_v = make_v(other_speed, other_angle)
    np.testing.assert_allclose(transform @ own_v, np.array([0, ego_speed, 0]), atol=1e-8)

    dvx, dvy, _ = (transform @ (other_v - own_v))

    return torch.tensor([
        encoder.encode_dist_val(dx),
        encoder.encode_dist_val(dy),
        encoder.encode_speed_val(dvx),
        encoder.encode_speed_val(dvy)
    ], dtype=torch.float32)


class RecurrentPathVisualization:
    def __init__(self, pgraph: PointGraph, ego_id: str, veh_colors=None):
        from graphviz import Digraph
        self.dot = Digraph()
        self.next_instance = 0
        self.pgraph = pgraph
        self.ego_id = ego_id
        self.veh_colors = {} if veh_colors is None else veh_colors

    def add_route(self, route: List[Tuple[str, bool]]):
        from HeterogeneousGraphRL.Representation.PointGraphRendering import add_veh_node, add_road_node, add_road_road_edge, add_veh_road_edge

        assert route[0][0] == self.ego_id
        print(route)

        # Add initial vehicle node only for first instance
        if self.next_instance == 0:
            add_veh_node(self.dot, self.pgraph, route[0][0], instance=0, use_xy=False, color=self.veh_colors.get(route[0][0], None))

        for i in range(1, len(route) - 1):
            add_road_node(self.dot, self.pgraph, route[i][0], instance=self.next_instance, use_xy=False)

            if i == 1:
                add_veh_road_edge(self.dot, self.pgraph, route[0][0], route[1][0], road_instance=self.next_instance)
            else:
                forward = route[i - 1][1]
                fi, ti = (i - 1, i) if forward else (i, i - 1)
                add_road_road_edge(self.dot, self.pgraph, route[fi][0], route[ti][0], from_instance=self.next_instance,
                                   to_instance=self.next_instance)

        # Add final vehicle and veh-road edge
        add_veh_node(self.dot, self.pgraph, route[-1][0], use_xy=False, instance=self.next_instance, color=self.veh_colors.get(route[-1][0], None))
        add_veh_road_edge(self.dot, self.pgraph, route[-1][0], route[-2][0], veh_instance=self.next_instance,
                          road_instance=self.next_instance)

        self.next_instance += 1


class RouteTooLongException(Exception):
    pass


def encode_relation_padded(pgraph: PointGraph, own_id, other_id, encoder, max_seq=10, precalc_route=None,
                           weighted=False, viz: Optional[RecurrentPathVisualization] = None):

    if precalc_route is None:
        if weighted:
            route = point_graph_route_weighted(pgraph, own_id, other_id)
        else:
            route = point_graph_route(pgraph, own_id, other_id)
    else:
        route = precalc_route

    if viz is not None:
        viz.add_route(route)

    if len(route) - 3 > max_seq:
        raise RouteTooLongException

    assert route[0][1]  # First transition should always be forward flow
    assert not route[-2][1]  # Last transition should always be reverse flow

    fc_lead_in = encoder.encode_lane_vehicle_edge(route[0][0], route[1][0],
                                                        pgraph.veh_road_forward[route[0][0]][route[1][0]])

    seq = torch.zeros((max_seq, 2 * (encoder.lane_node_dims + encoder.lane_lane_dims)))
    seq_len = torch.tensor(len(route) - 3)

    for i in range(1, len(route) - 2):
        forward = route[i][1]
        fi, ti = (i, i + 1) if forward else (i + 1, i)
        seq_t = torch.concat([
            encoder.encode_lane(route[i][0], pgraph.road_nodes[route[i][0]], route[i][0] in pgraph.route_reachable_set, route[i][0] in pgraph.route_continuable_set, route[i][0] in pgraph.route_goal_set),
            encoder.encode_lane_lane_edge(route[fi][0], route[ti][0],
                                                pgraph.road_road_forward[route[fi][0]][route[ti][0]])
        ], dim=-1)

        seq[i - 1] = torch.concat([torch.zeros_like(seq_t), seq_t]) if forward else torch.concat(
            [seq_t, torch.zeros_like(seq_t)])

    fc_lead_out = torch.concat([
        encoder.encode_lane(route[-2][0], pgraph.road_nodes[route[-2][0]], route[-2][0] in pgraph.route_reachable_set, route[-2][0] in pgraph.route_continuable_set, route[-2][0] in pgraph.route_goal_set),
        # Connection direction is vehicle->road in graph
        encoder.encode_lane_vehicle_edge(route[-1][0], route[-2][0],
                                               pgraph.veh_road_forward[route[-1][0]][route[-2][0]])
    ])

    return fc_lead_in, seq, seq_len, fc_lead_out
