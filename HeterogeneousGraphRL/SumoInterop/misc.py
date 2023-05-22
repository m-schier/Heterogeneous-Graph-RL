def module():
    from fahrzeug.Context import Context
    return Context.get_current_context().get_traci_module()


def populate(before=5, after=5, step_wait=20):
    import random

    traci = module()

    routes = traci.route.getIDList()

    def add_vehicle(v_id):
        route = random.choice(routes)
        traci.vehicle.addFull(v_id, route, departSpeed='max', departLane='allowed')
        #                                       xx            No sublane changes
        #                                         xx          Ignore other drivers when fulfilling traci request
        #                                           xx        No right drive change
        #                                             xx      No speed gain change
        #                                               xx    No cooperative changes
        #                                                 xx  Strategic changes unless conflicting with TraCI
        traci.vehicle.setLaneChangeMode(v_id, 0b000000000001)
        for _ in range(step_wait):
            traci.simulationStep()

    for i in range(before):
        add_vehicle('leader_{}'.format(i))

    add_vehicle('ego_0')
    traci.vehicle.setColor('ego_0', (255, 0, 255))

    for i in range(after):
        add_vehicle('follower_{}'.format(i))
