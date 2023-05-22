from typing import Union, Tuple, Optional

from HeterogeneousGraphRL.SumoInterop import Context
from HeterogeneousGraphRL.Representation.PointGraph import PointGraph, PgVehicleNode
from HeterogeneousGraphRL.Representation.NetworkQuery import NetworkQuery, NqClassifyConflictResult


def make_pgraph(return_nq=False, ctx=None) -> Union[PointGraph, Tuple[PointGraph, NetworkQuery]]:
    ctx = Context.get_current_context() if ctx is None else ctx
    nq = NetworkQuery(ctx.net_file)
    pgraph = PointGraph.from_query(nq)
    pgraph.sim_id = ctx.get_identifier()
    pgraph.time_stamp = ctx.get_traci_module().simulation.getTime()

    for a, b, ctype in nq.get_all_junction_conflicts():
        a_dist, b_dist = find_intersect(pgraph, nq, a, b)
        if ctype == NqClassifyConflictResult.A_HAS_ROW:
            pgraph.add_crossing(a, b, True, own_dist=a_dist, other_dist=b_dist)
            pgraph.add_crossing(b, a, False, own_dist=b_dist, other_dist=a_dist)
        elif ctype == NqClassifyConflictResult.B_HAS_ROW:
            pgraph.add_crossing(a, b, False, own_dist=a_dist, other_dist=b_dist)
            pgraph.add_crossing(b, a, True, own_dist=b_dist, other_dist=a_dist)
        else:
            raise ValueError

    if return_nq:
        return pgraph, nq
    else:
        return pgraph


def find_intersect(pgraph: PointGraph, nq: NetworkQuery, a, b):
    from shapely import geometry
    # For all internals calculate the shape over the entire internal

    lane_a = nq.lanes[a]
    lane_b = nq.lanes[b]

    geom_a = geometry.LineString(lane_a.shape)
    geom_b = geometry.LineString(lane_b.shape)

    intersect = geom_a.intersection(geom_b)

    def get_incoming_dist(key):
        edges = [edge for edge in pgraph.road_road_backward[key].values() if edge.type.is_junction_link()]
        assert len(edges) == 1
        return edges[0].distance

    if not isinstance(intersect, geometry.Point):
        # Happens when left lane has multiple internals, can ignore while not using left turning observed
        from warnings import warn
        warn("Bad intersect: {}, {}".format(a, b))
        a_dist, b_dist = -1000, -1000
    else:
        a_dist = geom_a.project(intersect) - get_incoming_dist(a)
        b_dist = geom_b.project(intersect) - get_incoming_dist(b)

    return a_dist, b_dist


def pgraph_encode_objects(pgraph: PointGraph, last_pgraph_objs: Optional[PointGraph] = None, in_circle=None, ctx=None) -> PointGraph:
    ctx = Context.get_current_context() if ctx is None else ctx

    if pgraph.sim_id != ctx.get_identifier():
        raise ValueError("Point graph created on {}, current context is {}".format(pgraph.sim_id, ctx.get_identifier()))

    pgraph = pgraph.copy_share_road_and_route()
    module = ctx.get_traci_module()
    pgraph.time_stamp = module.simulation.getTime()

    vehicles = module.vehicle.getIDList()

    if in_circle is not None:
        x, y, r = in_circle

    for veh_id in vehicles:
        if in_circle is not None:
            vx, vy = module.vehicle.getPosition(veh_id)
            dist = (vx - x) ** 2 + (vy - y) ** 2
            if dist > r * r:
                continue

        last_veh_node = None
        if last_pgraph_objs is not None:
            last_veh_node = last_pgraph_objs.try_get_vehicle(veh_id)
        vnode = PgVehicleNode.from_id(veh_id, prev_node=last_veh_node, module=module)
        pgraph.add_vehicle(veh_id, vnode,
                            *pgraph.find_lerps(module.vehicle.getLaneID(veh_id),
                                                module.vehicle.getLanePosition(veh_id)))

    return pgraph
