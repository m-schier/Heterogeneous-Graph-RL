import os
import sys
from typing import Optional, List

from gym import ObservationWrapper
import torch
import wandb

from HeterogeneousGraphRL.Representation.PointGraph import PointGraph
from HeterogeneousGraphRL.Representation.VehicleRelations import RecurrentPathVisualization, RouteTooLongException
from HeterogeneousGraphRL.Representation.VehicleSelectors import get_attended_vehicles_by_radius, get_attended_vehicles_flood3


def get_relevant_road_nodes_pgraph(pgraph: PointGraph, own_id: str):
    front_id, front_edge, *_ = pgraph.get_vehicle_roads(own_id)

    assert front_id in pgraph.route_reachable_set

    return [front_id] + [k for k, v in pgraph.road_road_forward[front_id].items()
                         if v.type.is_link() and k in pgraph.route_reachable_set]


class SteppedObservation:
    def __init__(self, pgraph: PointGraph, ego_id: str):
        self.pgraph = pgraph
        self.ego_id = ego_id
        self.cache = {}


class EnvWrapper(ObservationWrapper):
    def __init__(self, wrapped_env, encode_logs=False, veh_veh_mode='rnn', vehicle_selection='flood3',
                 encode_route=True, max_view_dist=100., store_visualization=False, normalize=True,
                 include_wrapped=True, encode_crossing_distances=False):
        from HeterogeneousGraphRL.Representation.Encoding import DefaultGraphEncoder

        super(EnvWrapper, self).__init__(wrapped_env)

        self.encoder = DefaultGraphEncoder(normalize=normalize, encode_logs=encode_logs, encode_route=encode_route,
                                           encode_crossing_distances=encode_crossing_distances)
        self.include_followers = True
        self.vehicle_selection = vehicle_selection
        self.veh_veh_mode = veh_veh_mode
        self.max_view_dist = max_view_dist
        self.store_visualization = store_visualization
        self.visualization: Optional[RecurrentPathVisualization] = None
        self.visualization_veh_colors = None
        self.include_wrapped = include_wrapped

        # We don't support a particular gym.space.Space since it doesn't cover dynamically sized spaces
        self.observation_space = None

        self._last_obs = None

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def _get_attended(self, pgraph, ego_id):
        if self.vehicle_selection == 'flood3':
            return get_attended_vehicles_flood3(pgraph, ego_id, max_depth=7, max_dist=self.max_view_dist)
        elif self.vehicle_selection == 'radius':
            return get_attended_vehicles_by_radius(pgraph, ego_id, self.max_view_dist)
        else:
            raise ValueError("self.attention_mode: {}".format(self.vehicle_selection))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if done and reward < -.8:
            # Debug that the ego vehicle saw at least a car
            # The observation should be the last frame's observation since GraphSpeedEnv reports the last obs again
            # on episode end
            att_ids = self._get_attended(observation.pgraph, observation.ego_id)

            if len(att_ids) == 0 or (info.ego_collision_veh is not None and info.ego_collision_veh not in att_ids):
                import pickle

                path = os.path.join(wandb.run.dir if wandb.run is not None else 'tmp', 'terminal_fail.pickle')
                try:
                    print("Dumping point graph on terminal error to \"{}\"".format(path), file=sys.stderr)
                    with open(path, 'wb') as fp:
                        pickle.dump((observation, reward, done, info), fp)
                    print("Dump completed successfully", file=sys.stderr)
                except Exception as ex:
                    import traceback
                    print("Error dumping:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    raise ex

                raise ValueError("Bad terminal observation, dump saved to \"{}\"".format(path))

        return self.observation(observation), reward, done, info

    def encode_vehicle_edges(self, obs: SteppedObservation, attended_ids: List[str], return_used_attended=False):
        from HeterogeneousGraphRL.Representation.VehicleRelations import encode_relation_padded, encode_relation_handcrafted, \
            RecurrentPathVisualization

        want_rnn = self.veh_veh_mode in ['rnn']
        want_hk = self.veh_veh_mode in ['handcrafted']
        assert want_hk or want_rnn

        result = tuple()
        used_attended = attended_ids[:]

        if want_rnn:
            used_attended = []
            max_seq = 12
            self.visualization = RecurrentPathVisualization(obs.pgraph, obs.ego_id, veh_colors=self.visualization_veh_colors) if self.store_visualization else None

            enc = []

            for other_id in attended_ids:
                try:
                    enc.append(encode_relation_padded(obs.pgraph, obs.ego_id, other_id, self.encoder, weighted=True,
                                                      max_seq=max_seq, viz=self.visualization))
                    used_attended.append(other_id)
                except RouteTooLongException:
                    pass  # This should rarely happen for correct settings of max_seq

            if len(enc) > 0:
                result = tuple([torch.stack(x) for x in zip(*enc)])
            else:
                result = torch.empty((0, self.encoder.lane_veh_dims)), \
                         torch.empty((0, max_seq, 2 * (self.encoder.lane_lane_dims + self.encoder.lane_node_dims))), \
                         torch.empty((0,), dtype=torch.int64), \
                         torch.empty((0, self.encoder.lane_node_dims + self.encoder.lane_veh_dims))

        if want_hk:
            # Important to reuse used_attended here in case recurrent dropped
            enc = [encode_relation_handcrafted(obs.pgraph, obs.ego_id, other_id, self.encoder) for other_id in used_attended]
            if len(enc) > 0:
                enc = torch.stack(enc)
            else:
                enc = torch.empty((0, 4))
            result = result + (enc,)

        if len(result) == 1:
            # Backwards compatibility
            result = result[0]

        if return_used_attended:
            return result, used_attended
        else:
            return result

    def observation(self, obs: SteppedObservation):
        ego_enc = self.encoder.encode_vehicle(obs.ego_id, obs.pgraph.veh_nodes[obs.ego_id])

        # Add encodings for current and upcoming road node
        important_road_nodes = get_relevant_road_nodes_pgraph(obs.pgraph, obs.ego_id)
        curr_road_node = important_road_nodes[0]
        road_enc = [self.encoder.encode_lane_vehicle_edge(obs.ego_id, curr_road_node,
                                                          obs.pgraph.veh_road_forward[obs.ego_id][curr_road_node]),
                    self.encoder.encode_lane(curr_road_node, obs.pgraph.road_nodes[curr_road_node],
                                             curr_road_node in obs.pgraph.route_reachable_set,
                                             curr_road_node in obs.pgraph.route_continuable_set,
                                             curr_road_node in obs.pgraph.route_goal_set)]

        enc = []
        for rn in important_road_nodes[1:]:
            enc.append(torch.cat(
                [self.encoder.encode_lane_lane_edge(curr_road_node, rn, obs.pgraph.road_road_forward[curr_road_node][rn]),
                 self.encoder.encode_lane(rn, obs.pgraph.road_nodes[rn], rn in obs.pgraph.route_reachable_set,
                                          rn in obs.pgraph.route_continuable_set,
                                          rn in obs.pgraph.route_goal_set)]))

        if enc:
            # TODO: Basic mean over road nodes best choice?
            enc = torch.mean(torch.stack(enc), dim=0)
        else:
            # Plausability check, if no roads follow, the current node must be a goal node
            assert curr_road_node in obs.pgraph.route_goal_set, "No next nodes but current is not goal"
            enc = torch.zeros(self.encoder.lane_lane_dims + self.encoder.lane_node_dims)

        ego_enc = torch.cat([ego_enc] + road_enc + [enc])

        attended_ids = self._get_attended(obs.pgraph, obs.ego_id)
        edge_enc, attended_ids = self.encode_vehicle_edges(obs, attended_ids, return_used_attended=True)
        attended_enc = [self.encoder.encode_vehicle(b, obs.pgraph.veh_nodes[b]) for b in attended_ids]

        if len(attended_enc) > 0:
            attended_enc = torch.stack(attended_enc)
        else:
            attended_enc = torch.empty((0, self.encoder.veh_node_dims))

        return ego_enc, attended_enc, edge_enc, obs if self.include_wrapped else None
