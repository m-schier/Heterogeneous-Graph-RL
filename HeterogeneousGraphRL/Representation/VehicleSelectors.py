import sys

import numpy as np

from HeterogeneousGraphRL.Representation.PointGraph import PointGraph
from typing import Iterable


def get_next_vehicle(pgraph: PointGraph, on, d_from, is_forward, filter_back=None):
    best_dist = None
    best_id = None

    for v_id, val in pgraph.veh_road_backward[on].items():
        if not val.is_front:
            continue

        if filter_back is not None:
            if pgraph.get_vehicle_roads(v_id)[2] != filter_back:
                continue

        dist = val.distance
        if is_forward:
            if dist < d_from and (best_dist is None or dist > best_dist):
                best_dist = dist
                best_id = v_id
        else:
            if dist > d_from and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_id = v_id

    return best_id, best_dist


def get_attended_vehicles_flood3(pgraph: PointGraph, ego_id: str, max_depth: int = 5, max_vehs: int = 1, max_dist=100.,
                                 verbose=False):

    # Flood fill from road node to road node

    def get_connecting_edges(rfrom, rto):
        return set([v for k, v in pgraph.road_road_forward[rfrom].items() if k == rto] +
                   [v for k, v in pgraph.road_road_backward[rfrom].items() if k == rto])

    veh_ids = set()

    def find_next_vehicles(vehicles_encountered, search_start, front_id, back_id, is_forward):
        # Find new vehicles
        if verbose:
            print("FLOOD3: Looking for vehicles between front {} and back {} with encountered {}"
                  .format(front_id, back_id, vehicles_encountered), file=sys.stderr)

        while vehicles_encountered < max_vehs:
            next_id, next_dist = get_next_vehicle(pgraph, front_id, search_start, is_forward, filter_back=back_id)

            if next_id is None:
                break
            search_start = next_dist

            vehicles_encountered = vehicles_encountered + 1

            if verbose:
                print("FLOOD3: Adding vehicle {} between {} and {}".format(next_id, front_id, back_id), file=sys.stderr)

            veh_ids.add(next_id)

        return vehicles_encountered

    ego_front_id, ego_front_edge, ego_back_id, _ = pgraph.get_vehicle_roads(ego_id)

    frontier = [
        (ego_front_id, find_next_vehicles(0, ego_front_edge.distance, ego_front_id, ego_back_id, True), 0),
        (ego_back_id, find_next_vehicles(0, ego_front_edge.distance, ego_front_id, ego_back_id, False), 0),
    ]

    visited_edges = set(get_connecting_edges(ego_front_id, ego_back_id))
    visited_nodes = set()

    while len(frontier) > 0:
        node_id, veh_encount, depth = frontier.pop(0)

        if depth > max_depth or veh_encount >= max_vehs or node_id in visited_nodes:
            continue

        if verbose:
            print("FLOOD3: Entering {} with encountered {}".format(node_id, veh_encount), file=sys.stderr)

        visited_nodes.add(node_id)

        # Search on all road road edges for vehicles

        for r_id, edge in pgraph.road_road_forward[node_id].items():
            if edge in visited_edges or not edge.type.is_link():
                continue

            if verbose:
                print("FLOOD3: Examining edge from {} to {}".format(r_id, node_id), file=sys.stderr)

            visited_edges.add(edge)

            ve_new = find_next_vehicles(veh_encount, np.inf, r_id, node_id, True)

            if verbose:
                print("FLOOD3: Adding to frontier: {}".format((r_id, ve_new, depth + 1)))

            frontier.append((r_id, ve_new, depth + 1))

        for r_id, edge in pgraph.road_road_backward[node_id].items():
            if edge in visited_edges or not edge.type.is_link():
                continue

            if verbose:
                print("FLOOD3: Examining edge from {} to {}".format(node_id, r_id), file=sys.stderr)

            visited_edges.add(edge)

            ve_new = find_next_vehicles(veh_encount, -np.inf, node_id, r_id, False)
            if verbose:
                print("FLOOD3: Adding to frontier: {}".format((r_id, ve_new, depth + 1)))
            frontier.append((r_id, ve_new, depth + 1))

        # Lastly, add all outgoing none-links to frontier

        for r_id, edge in pgraph.road_road_forward[node_id].items():
            if not edge.type.is_link():
                if verbose:
                    print("FLOOD3: Adding to frontier: {}".format((r_id, veh_encount, depth + 1)))
                frontier.append((r_id, veh_encount, depth + 1))

    return filter_vehicles_by_radius(pgraph, ego_id, list(veh_ids), max_dist)


def filter_vehicles_by_radius(pgraph: PointGraph, ego_id: str, veh_list: Iterable[str], radius_m: float):
    ego = pgraph.veh_nodes[ego_id]
    x, y = ego.xy

    result = []

    for v_id in veh_list:
        if ego_id == v_id:
            continue

        v_val = pgraph.veh_nodes[v_id]

        ox, oy = v_val.xy

        if (x - ox) ** 2 + (y - oy) ** 2 <= radius_m ** 2:
            result.append(v_id)

    return result


def get_attended_vehicles_by_radius(pgraph: PointGraph, ego_id: str, radius_m):
    return filter_vehicles_by_radius(pgraph, ego_id, pgraph.veh_nodes.keys(), radius_m)
