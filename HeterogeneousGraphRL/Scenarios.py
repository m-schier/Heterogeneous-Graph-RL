from typing import List, Tuple


__SCENARIOS = {
    'All': ['S1', 'S1b', 'S2', 'S2b', 'S3', 'S3b', 'S4', 'S4b', 'S5', 'S5'],  # Repeat last for equal distribution
}


def __punish_standing_for_list(scenario_list):
    from HeterogeneousGraphRL.GymRL.GraphSpeedEnv import SINGLE_SPAWN_SCENARIOS

    punish_standing_collision = sum((s in SINGLE_SPAWN_SCENARIOS for s in scenario_list))
    # All must agree
    assert punish_standing_collision == 0 or punish_standing_collision == len(scenario_list)
    return punish_standing_collision > 0


def get_train_scenario_list(train_sc) -> Tuple[List[str], bool]:
    if train_sc in __SCENARIOS:
        scenario_list = __SCENARIOS[train_sc]
    elif train_sc[0] == '-':
        scenario_list = [s for s in __SCENARIOS['All'] if not s.startswith(train_sc[1:])]
    else:
        scenario_list = [train_sc]

    return scenario_list, __punish_standing_for_list(scenario_list)


def get_eval_scenario_list(train_sc, eval_sc) -> List[List[str]]:
    # Special value indicating for the train scenario to be reused
    if eval_sc is True:
        return [get_train_scenario_list(train_sc)[0]]
    elif eval_sc == 'opposite':
        if train_sc[0] == '-':
            subset = __SCENARIOS['All']
        else:
            raise ValueError((train_sc, eval_sc))

        eval1_scenarios = [s for s in subset if s.startswith(train_sc[1:])]
        eval2_scenarios = [s for s in subset if not s.startswith(train_sc[1:])]
        return [eval1_scenarios, eval2_scenarios]
    else:
        return [get_train_scenario_list(eval_sc)[0]]
