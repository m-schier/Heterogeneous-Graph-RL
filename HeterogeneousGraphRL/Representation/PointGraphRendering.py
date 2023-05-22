from graphviz import Digraph
from HeterogeneousGraphRL.Representation.PointGraph import PointGraph, PgRoadRoadType


def __dot_sanitize(x, instance=0):
    return x.replace(':', '__') + str(instance)


def __dot_make_pos(x, y):
    scale = 1
    return '{},{}!'.format(x * scale, y * scale)


def __make_color(col):
    import numpy as np

    if np.shape(col) == (3,) and np.issubdtype(type(col[0]), np.integer):
        return "#{:02x}{:02x}{:02x}".format(*col)
    elif type(col) == str:
        return col
    else:
        raise ValueError("Color value not parseable: {}".format(col))


def add_road_node(dot: Digraph, pgraph: PointGraph, node_id: str, use_xy: bool = True, instance=0):
    id_name = __dot_sanitize(node_id, instance)
    node_val = pgraph.road_nodes[node_id]
    header = node_id
    extra = 'R' if node_id in pgraph.route_reachable_set else ''
    if node_id in pgraph.route_continuable_set:
        extra += 'C'
    if extra:
        header += ' (' + extra + ')'
    label = "{}\nv_max = {:.2f} m/s".format(header, node_val.speed_limit)
    kwargs = {}
    if node_val.xy is not None and use_xy:
        kwargs['pos'] = __dot_make_pos(*node_val.xy)
    dot.node(id_name, label=label, style="dashed" if node_val.is_internal else "solid", **kwargs)


def add_veh_node(dot: Digraph, pgraph: PointGraph, node_id: str, use_xy: bool = True, color=None, instance=0):
    id_name = __dot_sanitize(node_id, instance)
    node_val = pgraph.veh_nodes[node_id]
    header = node_id
    if node_val.signaling_left:
        header = "⇦ " + header
    if node_val.signaling_right:
        header = header + " ⇨"
    label = "{}\nv = {:.2f} m/s (was {:.2f} m/s)".format(header, node_val.speed, node_val.prev_speed)
    kwargs = {'fillcolor': 'lightgrey', 'shape': 'box', 'style': 'filled'}
    if color is not None:
        kwargs['fillcolor'] = __make_color(color)
    if node_val.xy is not None and use_xy:
        kwargs['pos'] = __dot_make_pos(node_val.xy[0], node_val.xy[1] + 1)
    dot.node(id_name, label=label, **kwargs)


def add_road_road_edge(dot: Digraph, pgraph: PointGraph, from_id: str, to_id: str, forward=True, from_instance=0,
                       to_instance=0):
    lut = pgraph.road_road_forward if forward else pgraph.road_road_backward
    edge_val = lut[from_id][to_id]

    kwargs = {}
    if edge_val.type == PgRoadRoadType.CrossingWithRow:
        kwargs['color'] = 'green'
        kwargs['style'] = 'bold'
        # kwargs['label'] = 'a={:.2f}m, b={:.2f}m'.format(edge_val.own_crossing_distance,
        #                                                 edge_val.other_crossing_distance)
    elif edge_val.type == PgRoadRoadType.CrossingWithYield:
        kwargs['color'] = 'red'
        kwargs['style'] = 'bold'
        kwargs['label'] = 'a={:.2f}m, b={:.2f}m'.format(edge_val.own_crossing_distance,
                                                        edge_val.other_crossing_distance)
    elif edge_val.type == PgRoadRoadType.LeftNeighbor:
        kwargs['label'] = 'Left'
    elif edge_val.type == PgRoadRoadType.RightNeighbor:
        kwargs['label'] = 'Right'
    elif edge_val.type in [PgRoadRoadType.LinkStraight, PgRoadRoadType.LinkRight, PgRoadRoadType.LinkLeft,
                           PgRoadRoadType.Continuation]:
        kwargs['label'] = 'd={:.2f}m'.format(edge_val.distance)
        if edge_val.type == PgRoadRoadType.LinkLeft:
            kwargs['label'] += '\nLeftLink'
        elif edge_val.type == PgRoadRoadType.LinkRight:
            kwargs['label'] += '\nRightLink'
        elif edge_val.type == PgRoadRoadType.LinkStraight:
            kwargs['label'] += '\nStraightLink'
        elif edge_val.type == PgRoadRoadType.Continuation:
            kwargs['label'] += '\nContinuation'
    else:
        raise ValueError("Connection type not understood: {}".format(edge_val.type))

    dot.edge(__dot_sanitize(from_id, from_instance), __dot_sanitize(to_id, to_instance), **kwargs)


def add_veh_road_edge(dot: Digraph, pgraph: PointGraph, veh_id: str, road_id: str, veh_instance=0, road_instance=0):
    edge_val = pgraph.veh_road_forward[veh_id][road_id]
    kwargs = {'label': '{} {:.2f}\n{:.2f}m'.format('Front' if edge_val.is_front else 'Back', edge_val.ratio,
                                                   edge_val.distance)}
    dot.edge(__dot_sanitize(veh_id, veh_instance), __dot_sanitize(road_id, road_instance), **kwargs)
