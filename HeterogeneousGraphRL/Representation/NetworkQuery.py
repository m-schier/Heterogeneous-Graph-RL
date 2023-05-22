from enum import Enum
from collections import namedtuple
import sumolib


class NqConnectionDirection(Enum):
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1


IqEdgeObj = namedtuple("IqEdgeObj", ["internal", "priority", "lanes"])
IqJuncObj = namedtuple("IqJuncObj", ["type", "int_lanes", "requests"])
IqReqObj = namedtuple("IqReqObj", ["response", "foes"])
IqLaneObj = namedtuple("IqLaneObj", ["index", "parent", "next_lanes", "speed", "length", "shape"])
IqConnObj = namedtuple("IqConnObj", ["to", "dir"])


class NqClassifyConflictResult(Enum):
    NO_CONFLICT = 0
    A_HAS_ROW = 1
    B_HAS_ROW = 2


class NetworkQuery:
    def __init__(self, net_file: str):
        self._junc_dict = {}
        self._int_to_major_junction = {}
        self.edges = {}
        self.lanes = {}
        self._int_lane_to_req_lane = {}  # Translate internal lane to the request lane for multipart lanes

        for edge in sumolib.xml.parse(net_file, ['edge']):
            p = None if edge.priority is None else int(edge.priority)

            lane_dict = {}

            for lane in edge.getChild('lane'):
                lane_dict[int(lane.index)] = lane.id
                self.lanes[lane.id] = IqLaneObj(int(lane.index), edge.id, [], float(lane.speed), float(lane.length),
                                                NetworkQuery.parse_shape(lane.shape))

            self.edges[edge.id] = IqEdgeObj(edge.function == 'internal', p, lane_dict)

        # After parsing edges and lanes, add connections between them
        self.__build_edge_graph(net_file)

        for junction in sumolib.xml.parse(net_file, ['junction']):
            int_lanes = junction.intLanes.split(' ') if len(junction.intLanes) > 0 else []

            if junction.type != 'internal':
                for int_lane in int_lanes:
                    self._int_to_major_junction[int_lane] = junction.id
                    self._int_lane_to_req_lane[int_lane] = int_lane

            requests = {}

            if junction.type in ['priority', 'right_before_left']:
                for req in junction.getChild('request'):
                    requests[int(req.index)] = IqReqObj(req.response, req.foes)

            self._junc_dict[junction.id] = IqJuncObj(junction.type, int_lanes, requests)

        for connection in sumolib.xml.parse(net_file, ['connection']):
            via_lane = connection.via
            if via_lane is None:
                continue

            from_edge = connection.attr_from
            from_lane = self.edges[from_edge].lanes[int(connection.fromLane)]
            via_edge = self.lanes[via_lane].parent
            if self.edges[from_edge].internal and self.edges[via_edge].internal:
                self._int_lane_to_req_lane[from_lane] = via_lane

    def __build_edge_graph(self,  netfile):
        # Connect all lanes according to their connections

        # SUMO has a weird encoding of, i.e. when A->B->C->D given, then SUMO might encode A->B->D, B->C->D.
        # Thus, track all known "directly linked" candidates, i.e. A->B, B->D, B->C, C->D.
        # Then, find all connections known to be indirect, i.e. A->B->D => A->D, B->C->D => B->D.
        # Remove all indirects from candidates

        candidates = []  # List of candidate from-to tuples, where it is not known whether connection is direct
        known_indirects = set()  # List of connections known not to be direct

        for connection in sumolib.xml.parse(netfile, ['connection']):
            from_edge = connection.attr_from
            from_lane = self.edges[from_edge].lanes[int(connection.fromLane)]
            to_edge = connection.to
            to_lane = self.edges[to_edge].lanes[int(connection.toLane)]

            if connection.via is None:
                candidates.append((from_lane, to_lane, connection.dir))
            else:
                known_indirects.add((from_lane, to_lane))
                candidates.append((from_lane, connection.via, connection.dir))
                candidates.append((connection.via, to_lane, connection.dir))

        filtered = [c for c in candidates if c[:2] not in known_indirects]
        # TODO: Why is this required?
        filtered = set(filtered)

        for lfrom, lto, d in filtered:
            self.lanes[lfrom].next_lanes.append(IqConnObj(lto, d))

    def iterate_lanes(self):
        return self.lanes.items()

    def get_next_lanes(self, lane_id):
        return self.lanes[lane_id].next_lanes[:]

    def get_left_lane(self, lane_id):
        lane = self.lanes[lane_id]
        edge = self.edges[lane.parent]
        if lane.index + 1 in edge.lanes:
            return edge.lanes[lane.index + 1]
        else:
            return None

    def get_right_lane(self, lane_id):
        lane = self.lanes[lane_id]
        if lane.index <= 0:
            return None

        return self.edges[lane.parent].lanes[lane.index - 1]

    def get_all_junction_conflicts(self):
        result = []

        for jid, junc in self._junc_dict.items():
            if junc.type not in ['dead_end', 'internal']:
                result += self.get_conflicts_for_junctions(jid)

        return result

    def get_conflicts_for_junctions(self, jid):
        result = []

        junc = self._junc_dict[jid]

        n = len(junc.int_lanes)

        for a in range(n):
            l_a = junc.int_lanes[a]
            for b in range(a + 1, n):
                l_b = junc.int_lanes[b]

                res = self.classify_conflict(l_a, l_b)

                if res != NqClassifyConflictResult.NO_CONFLICT:
                    result.append((l_a, l_b, res))

        return result

    def classify_conflict(self, a, b):
        junca = self._int_to_major_junction[a]
        juncb = self._int_to_major_junction[b]

        if junca != juncb:
            raise ValueError("Internals {} and {} are not on same junction: {} != {}".format(a, b, junca, juncb))

        if self._junc_dict[junca].type not in ['priority', 'right_before_left']:
            raise ValueError("Junction {} has unsupported type: {}".format(junca, self._junc_dict[junca].type))

        # Determine who has more priority
        maj_a = self._int_lane_to_req_lane[a]
        maj_b = self._int_lane_to_req_lane[b]

        j = self._junc_dict[junca]
        a_idx = j.int_lanes.index(maj_a)
        b_idx = j.int_lanes.index(maj_b)

        a_is_bs_foe = j.requests[b_idx].foes[-a_idx - 1] == '1'
        b_is_as_foe = j.requests[a_idx].foes[-b_idx - 1] == '1'

        assert a_is_bs_foe == b_is_as_foe

        if not a_is_bs_foe:
            return NqClassifyConflictResult.NO_CONFLICT

        a_has_row = j.requests[b_idx].response[-a_idx - 1] == '1'
        b_has_row = j.requests[a_idx].response[-b_idx - 1] == '1'

        assert a_has_row != b_has_row

        return NqClassifyConflictResult.A_HAS_ROW if a_has_row else NqClassifyConflictResult.B_HAS_ROW

    @staticmethod
    def parse_shape(shape_str):
        return [p[:2] for p in sumolib.net.convertShape(shape_str)]
