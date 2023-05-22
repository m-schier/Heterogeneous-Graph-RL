import os
import sys
import warnings

import numpy as np
import omegaconf

from HeterogeneousGraphRL.GymRL import GraphSpeedEnvEncoder, GraphSpeedEnv
from HeterogeneousGraphRL.Representation.GraphConstruction import make_pgraph, pgraph_encode_objects
from HeterogeneousGraphRL.Representation.PointGraph import point_graph_route_from_traci
from HeterogeneousGraphRL.Representation.Observations import SteppedObservation, EnvWrapper
import torch
from HeterogeneousGraphRL.QLearning.SteppedDQN import SteppedDQN, PERSteppedDQN
from HeterogeneousGraphRL.SumoInterop import Context
from typing import List, Tuple, Union
from HeterogeneousGraphRL.Util.Timer import Timer
from omegaconf import DictConfig


def create_point_graph(ctx: Context, ego_id: str):
    pgb, nq = make_pgraph(return_nq=True, ctx=ctx)
    module = ctx.get_traci_module()
    traci_route = module.vehicle.getRoute(ego_id)
    route_info = point_graph_route_from_traci(pgb, nq, traci_route)
    return pgb.with_encoded_route_info(*route_info)


def create_point_graph_with_vehicles(ctx, point_graph, last_vehicle_point_graph, in_circle=None):
    return pgraph_encode_objects(point_graph, last_pgraph_objs=last_vehicle_point_graph,
                                 in_circle=in_circle, ctx=ctx)


class EnvEncoder(GraphSpeedEnvEncoder):
    def __init__(self):
        self.pgraph_base = None
        self.last_obj_pgraph = None

    def reset(self):
        self.pgraph_base = None
        self.last_obj_pgraph = None

    def __call__(self, ego_id, ctx):
        if self.pgraph_base is None:
            self.pgraph_base = create_point_graph(ctx, ego_id)

        module = ctx.get_traci_module()
        ego_pos = module.vehicle.getPosition(ego_id)
        vision_circle = ego_pos + (60.,)  # Radius in meters

        pgraph = create_point_graph_with_vehicles(ctx, self.pgraph_base, self.last_obj_pgraph, in_circle=vision_circle)
        self.last_obj_pgraph = pgraph
        return SteppedObservation(pgraph, ego_id)


class StepThreeModule(torch.nn.Module):
    def __init__(self, conv='gatv2', src_dims=23, dst_dims=27, hidden_dims=16, heads=5):
        from torch_geometric.nn import GATv2Conv, SAGEConv, GATConv

        super(StepThreeModule, self).__init__()
        self.src_encoder = torch.nn.Sequential(torch.nn.Linear(src_dims, hidden_dims), torch.nn.ReLU())
        self.dst_encoder = torch.nn.Sequential(torch.nn.Linear(dst_dims, hidden_dims), torch.nn.ReLU())

        if conv == 'gatv2':
            self.graph_conv = GATv2Conv(hidden_dims, hidden_dims, heads=heads, add_self_loops=False)
        elif conv == 'gat':
            self.graph_conv = GATConv(hidden_dims, hidden_dims, heads=heads, add_self_loops=False)
        elif conv == 'sage':
            assert heads == 1
            self.graph_conv = SAGEConv((hidden_dims, hidden_dims), hidden_dims * heads)
        else:
            raise ValueError('conv = {}'.format(conv))

        # self.output_enc = torch.nn.Sequential(torch.nn.Linear(out_size, output_dims), torch.nn.ReLU())
        self.heads = heads
        self.hidden_dims = hidden_dims

    def forward(self, batch):
        dst_encodings = self.dst_encoder(batch['dst'].x)

        if 'src' in batch.node_types:
            src_encodings = self.src_encoder(batch['src'].x)
            dst_convs = torch.relu(self.graph_conv((src_encodings, dst_encodings), batch['src', 'to', 'dst'].edge_index))
        else:
            dst_convs = torch.zeros((batch['dst'].num_nodes, self.heads * self.hidden_dims), device=dst_encodings.device)

        return torch.cat([dst_encodings, dst_convs], -1)
        # return self.output_enc(torch.cat([dst_encodings, dst_convs], -1))


def pinned_transfer(tensor: torch.Tensor, device: torch.device):
    # Support Optional[torch.Tensor] as well
    if tensor is None:
        return None

    def is_on_cpu():
        if isinstance(tensor, torch.nn.utils.rnn.PackedSequence):
            return tensor.data.device.type == 'cpu'
        else:
            return tensor.device.type == 'cpu'

    assert is_on_cpu()

    if device.type == 'cpu':
        # Skip pinning when staying on cpu
        return tensor
    else:
        # Else pin and copy
        return tensor.pin_memory().to(device=device, non_blocking=True)


class BaseModule(torch.nn.Module):
    def __init__(self, conv='gatv2', edge_mode='lstm2', include_followers=None, encode_logs=None, graph_src_dims=23,
                 graph_dst_dims=27, graph_hidden_dims=16, graph_heads=5, lane_lane_dims=9):
        from HeterogeneousGraphRL.Representation.VehicleRelations import EdgePredictorModuleHandcrafted, RecurrentEdgePredictorModule2, NoopEdgePredictorModule
        super(BaseModule, self).__init__()

        if include_followers is not None:
            warnings.warn("include_followers for BaseModule.__init__ is deprecated and unused")

        if encode_logs is not None:
            warnings.warn("encode_logs for BaseModule.__init__ is deprecated and unused")

        print("BaseModule: edge_mode={}".format(edge_mode), file=sys.stderr)
        self.edge_mode = edge_mode
        self.s3 = StepThreeModule(conv=conv, src_dims=graph_src_dims, dst_dims=graph_dst_dims,
                                  hidden_dims=graph_hidden_dims, heads=graph_heads)

        if self.edge_mode == 'lstm2':
            self.edge_predictor = RecurrentEdgePredictorModule2(recurrent_arch='lstm', mid_features=8 + 2 * lane_lane_dims,
                                                                output_features=graph_hidden_dims)
            self.edge_predictor_is_recurrent = True
        elif self.edge_mode == 'rnn2':
            self.edge_predictor = RecurrentEdgePredictorModule2(recurrent_arch='rnn_relu',
                                                                output_features=graph_hidden_dims,
                                                                mid_features=8 + 2 * lane_lane_dims)
            self.edge_predictor_is_recurrent = True
        elif self.edge_mode == 'handcrafted':
            self.edge_predictor_is_recurrent = False
            self.edge_predictor = EdgePredictorModuleHandcrafted(features=[256, 128, graph_hidden_dims])
        elif self.edge_mode == 'noop':
            self.edge_predictor_is_recurrent = False
            self.edge_predictor = NoopEdgePredictorModule(output_features=graph_hidden_dims)
        else:
            raise ValueError(self.edge_mode)

        print("BaseModule: Parameter count             : {}".format(sum(p.numel() for p in self.parameters())), file=sys.stderr)
        print("BaseModule: Edge encoder parameter count: {}".format(sum(p.numel() for p in self.edge_predictor.parameters())), file=sys.stderr)

    def collate_batch(self, states: List[Tuple]):
        from torch_geometric.data import HeteroData

        def make_target_idx(lens):
            res = []
            for i, el in enumerate(lens):
                res += [i] * el
            return res
        ego_veh_encodes, other_veh_encodes, encoded_relations, *_ = zip(*states)
        flat_encoded_vehs = torch.cat(other_veh_encodes)

        len_flat_encoded_vehs = len(flat_encoded_vehs)
        have_src_vehs = len_flat_encoded_vehs > 0
        if have_src_vehs:
            with Timer.get_current().time('mod_prepare_edges'):
                if self.edge_predictor_is_recurrent:
                    rec_edge_embs = list(zip(*encoded_relations))  # Consume for profiling
                    edge_batch = [torch.cat(x) for x in rec_edge_embs]
                    assert len(edge_batch[0]) == len(flat_encoded_vehs)
                    edge_batch = self.edge_predictor.collate(*edge_batch)
                else:
                    flat_encoded_relations = torch.cat(encoded_relations)
                    edge_batch = [flat_encoded_relations]
                    assert len(flat_encoded_vehs) == len(flat_encoded_relations)
        else:
            flat_encoded_vehs = None
            edge_batch = None

        with Timer.get_current().time('mod_build_graph'):
            dst_emb = torch.stack(ego_veh_encodes)

            graph_batch = HeteroData()
            graph_batch['dst'].x = dst_emb

            if have_src_vehs:
                graph_batch['src'].x = None  # Must be calculated later

                encoded_lengths = [e.shape[0] for e in other_veh_encodes]

                # Found to be faster to construct in python
                dst_idx = torch.tensor(make_target_idx(encoded_lengths), dtype=torch.int64)

                assert len(dst_idx) == len(flat_encoded_vehs)

                graph_batch['src', 'to', 'dst'].edge_index = torch.stack([torch.arange(len(flat_encoded_vehs)), dst_idx])

        return graph_batch, flat_encoded_vehs, edge_batch

    @staticmethod
    def transfer_batch(batch, dev):
        graph_batch, flat_encoded_vehs, edge_batch = batch

        assert (flat_encoded_vehs is not None) == (edge_batch is not None)

        graph_batch['dst'].x = pinned_transfer(graph_batch['dst'].x, dev)

        if flat_encoded_vehs is not None:
            flat_encoded_vehs = pinned_transfer(flat_encoded_vehs, dev)
            edge_batch = [pinned_transfer(x, dev) for x in edge_batch]
            graph_batch['src', 'to', 'dst'].edge_index = pinned_transfer(graph_batch['src', 'to', 'dst'].edge_index, dev)

        return graph_batch, flat_encoded_vehs, edge_batch

    def collate(self, batch):
        batch = self.collate_batch(batch)
        return BaseModule.transfer_batch(batch, next(self.parameters()).device)

    def predict_batch(self, batch_on_dev):
        graph_batch, flat_encoded_vehs, edge_batch = batch_on_dev

        with Timer.get_current().time('mod_predict_edges'):
            if edge_batch is not None:
                edge_outputs = self.edge_predictor(*edge_batch)
                graph_batch['src'].x = torch.cat([edge_outputs, flat_encoded_vehs], -1)

        with Timer.get_current().time('mod_predict_graph'):
            base_out = self.s3(graph_batch)

        return base_out

    def forward(self, states: Union[Tuple, List[Tuple]]):
        # Either pre-collated, then tuple, or not yet collated, then list and do so lazily
        if type(states) == list:
            states = self.collate(states)
        elif type(states) != tuple:
            raise TypeError(str(type(states)))

        return self.predict_batch(states)


class StandardRlModule(BaseModule):
    def __init__(self, graph_hidden_dims=16, graph_heads=5, head_dims=8, out_dims=3, **kwargs):
        from torch.nn import Sequential, Linear, ReLU
        super(StandardRlModule, self).__init__(graph_heads=graph_heads, graph_hidden_dims=graph_hidden_dims, **kwargs)
        # TODO: Static input size
        head_input_dims = (graph_heads + 1) * graph_hidden_dims
        self.out_lin = Sequential(Linear(head_input_dims, head_dims), ReLU(),
                                  Linear(head_dims, out_dims))

    def forward(self, states: List[SteppedObservation]):
        return self.out_lin(super(StandardRlModule).forward(states))


class DuellingRlModule(BaseModule):
    def __init__(self, graph_hidden_dims=16, graph_heads=5, head_dims=128, out_dims=3, **kwargs):
        from torch.nn import Sequential, Linear, ReLU
        super(DuellingRlModule, self).__init__(graph_heads=graph_heads, graph_hidden_dims=graph_hidden_dims, **kwargs)
        # TODO: Static input size
        head_input_dims = (graph_heads + 1) * graph_hidden_dims
        branch_input_dims = head_input_dims

        # If somehting, then:
        if True:
            inter_dims = 512
            self.intermediate = Sequential(Linear(head_input_dims, inter_dims), ReLU())
            branch_input_dims = inter_dims
        else:
            self.intermediate = None
            branch_input_dims = head_input_dims

        self.adv_lin = Sequential(Linear(branch_input_dims, head_dims), ReLU(),
                                  Linear(head_dims, out_dims))
        self.val_lin = Sequential(Linear(branch_input_dims, head_dims), ReLU(),
                                  Linear(head_dims, 1))
        self.val = None
        self.adv = None

    def forward(self, states):
        base_out = super(DuellingRlModule, self).forward(states)

        if self.intermediate is not None:
            base_out = self.intermediate(base_out)

        adv, val = self.adv_lin(base_out), self.val_lin(base_out)
        self.val = val
        self.adv = adv
        return val + (adv - adv.mean(dim=1, keepdim=True))


def get_veh_veh_mode(edge_mode: str):
    if edge_mode in ['noop', 'handcrafted']:
        return 'handcrafted'
    else:
        return 'rnn'


def main(cfg: DictConfig):
    print("Entering python main...", file=sys.stderr)

    import wandb
    from HeterogeneousGraphRL.ExtendedModule import train_extended
    from HeterogeneousGraphRL.Util.TorchPool import TorchPool, NumpyTransmissionWrapper
    from HeterogeneousGraphRL.Scenarios import get_eval_scenario_list, get_train_scenario_list

    print(cfg)

    scenario, punish_standing_collision = get_train_scenario_list(cfg.env.train)
    print(f"{scenario = }", file=sys.stderr)

    env_kwargs = {
        'timeout_reward': np.nan,
        'punish_standing_collision': punish_standing_collision,
        'allow_overspeed': cfg.architecture.allow_overspeed,
        'super_steps': 4,
        'micro_step_length': .1,
        'accel': cfg.env.accel,
        'randomize_speeds': cfg.env.randomize_speeds,
    }

    drop_nan_rewards = True

    if cfg.train.terminal_reward == 'nan':
        env_kwargs['terminal_reward'] = np.nan
    elif isinstance(cfg.train.terminal_reward, (int, float)):
        env_kwargs['terminal_reward'] = cfg.train.terminal_reward
    else:
        raise ValueError(cfg.train.terminal_reward)

    graph_dims = 16 if 'graph_dims' not in cfg.architecture else cfg.architecture.graph_dims
    graph_heads = 5 if 'graph_heads' not in cfg.architecture else cfg.architecture.graph_heads

    wrapper_veh_veh_mode = get_veh_veh_mode(cfg.architecture.edge_mode)
    max_view_dist = 100. if 'view_distance' not in cfg.architecture else cfg.architecture.view_distance
    encode_crossing = False if 'encode_crossing' not in cfg.architecture else cfg.architecture.encode_crossing

    wrapper_kwargs = {
        'veh_veh_mode': wrapper_veh_veh_mode,
        'vehicle_selection': cfg.architecture.vehicle_selection,
        'encode_route': cfg.architecture.encode_route,
        'max_view_dist': max_view_dist,
        'include_wrapped': False,
        'encode_crossing_distances': encode_crossing
    }

    def make_env(sc, default_sumo_mode='libsumo', scenario_selection='random'):
        return EnvWrapper(GraphSpeedEnv(sc, EnvEncoder(), default_sumo_mode=default_sumo_mode,
                                        scenario_selection=scenario_selection, **env_kwargs),
                          **wrapper_kwargs)

    def env_provider():
        return make_env(scenario)

    n_pool_workers = cfg.train.workers if 'workers' in cfg.train else 4
    train_env_pool = TorchPool(env_provider, n_pool_workers, state_wrapper=NumpyTransmissionWrapper())
    # Must use 'sumo' instead of 'libsumo' here, since multiple 'libsumo' connections may not exist in the same process
    greedy_env = make_env(scenario, default_sumo_mode='sumo')

    net_provider_args = {
        'conv': cfg.architecture.conv,
        'edge_mode': cfg.architecture.edge_mode,
        'graph_src_dims': graph_dims + 7,
        'graph_dst_dims': 18 + greedy_env.encoder.lane_lane_dims,
        'graph_hidden_dims': graph_dims,
        'graph_heads': graph_heads,
        'lane_lane_dims': greedy_env.encoder.lane_lane_dims
    }

    dqn_class = SteppedDQN if not cfg.architecture.per else PERSteppedDQN
    provider = DuellingRlModule if cfg.architecture.duelling else StandardRlModule

    net_provider = lambda n_act, **kwargs: provider(out_dims=n_act, **kwargs)

    eps_last_epoch = 1000 * (cfg.train.eps_last_epoch if 'eps_last_epoch' in cfg.train else cfg.train.max_epochs)
    eps_end = cfg.train.eps_end if 'eps_end' in cfg.train else .02

    last_step = 1000 * cfg.train.max_epochs

    dqn = dqn_class(greedy_env=greedy_env, train_env_pool=train_env_pool, net_provider=net_provider, net_provider_args=net_provider_args,
                    render_interval=cfg.log.render_interval, use_double=cfg.architecture.double, batch_size=cfg.train.batch_size,
                    gamma=cfg.train.gamma, learning_rate=cfg.train.learning_rate, drop_nan_rewards=drop_nan_rewards,
                    eps_last_frame=eps_last_epoch, eps_end=eps_end, verbose_logging=cfg.log.verbose,
                    last_step=last_step)

    # Set the user data, which contains training parameters that are not model parameters
    dqn.user_data = {
        'wrapper_kwargs': wrapper_kwargs,
        'env_kwargs': env_kwargs,
        'configuration': omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    }

    # Parse the evaluation options already to fail fast on misconfiguration
    do_eval = bool(cfg.env.eval)
    if do_eval:
        eval_scenarios = get_eval_scenario_list(cfg.env.train, cfg.env.eval)
        print(f"{eval_scenarios = }", file=sys.stderr)
        eval_episodes = 2000 if 'eval_episodes' not in cfg.train else cfg.train.eval_episodes
    else:
        eval_scenarios, eval_episodes = None, None

    # Detect available GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("CUDA is available with at least one device and will be used", file=sys.stderr)
        dqn = dqn.to(torch.device('cuda'))
    else:
        print("Failed to detect any compatible GPUs, falling back to CPU", file=sys.stderr)

    wandb_dir = os.path.join(cfg.log.dir if 'dir' in cfg.log else 'tmp/', cfg.log.project)
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(project="Auto", resume='never', dir=wandb_dir, tags=[cfg.log.project],
               config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    log_interval = 1 if 'interval' not in cfg.log else cfg.log.interval

    train_extended(dqn, max_epoch=cfg.train.max_epochs, log_interval=log_interval)
    # Do not access configuration beyond this point! Want to have fully parsed before training loop to avoid late faults

    train_env_pool.close()

    # Always evaluate on CPU
    dqn = dqn.to(torch.device('cpu'))

    if do_eval:
        from .Evaluation import test_metrics
        import wandb

        greedy_env.close()  # Must close to not keep an open context (which libsumo doesn't support)

        for evi, evs in enumerate(eval_scenarios):
            eval_env = make_env(evs, scenario_selection='sequential')
            metrics = test_metrics(dqn.net, eval_env, runs=eval_episodes)
            eval_env.close()

            log_dict = {
                "success_rate": metrics.succeeded_runs / metrics.total_runs,
                "early_termination_rate": metrics.failed_runs / metrics.total_runs,
                "avg_reward": metrics.avg_reward,
                "avg_success_steps": metrics.avg_successful_episode_steps,
                "avg_steps": metrics.avg_episode_steps
            }

            log_dict = {"eval{}/{}".format(evi + 1, k): v for k, v in log_dict.items()}
            wandb.log(log_dict)
