from typing import List, Dict


def get_non_final_mask(next_state: List[Dict]) -> List[bool]:
    non_final_mask = []
    for observation in next_state:
        agent_obs = observation.values()
        for obs in agent_obs:
            non_final_mask.append(True if obs is not None else False)
    return non_final_mask
