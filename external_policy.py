import numpy as np
import utils as U
import math

"""
This file will contain external policies which can be used by the contaminated group of agents in the game.
A policy will map each given observation to a selected actions.
An observation in our space will be a relative position and angle of all the agents in our observable area.
An action is a movement in a given direction
"""

ATTRACTION_FORCE = 1.0
REPULSION_FORCE = -1.5

MIN_DIST = 3

def go_to_closest_neighbor(agent_state, internal_agent_state, observation):
    """
    Given state of an agent and his observation return the action which directs him to his
    closest neighbor which has the same state.
    :param agent_state:
    :param internal_agent_state Internal state of the agent: Healthy or Contaminated.
    :param observation: List of all the agents that the given agent can see.
    :return:
    """

    # If there are no observations return a random action.
    if len(observation) == 0:
        return np.random.randn(1,2)

    min_dist = float('inf')
    closest_agent = None
    for idx, agent_obs in enumerate(observation):
        if agent_obs.dist < min_dist and agent_obs.state.value == internal_agent_state:
            min_dist = agent_obs.dist
            closest_agent = idx
    # If there is no close agent of our type wander.
    if closest_agent is None:
        return np.array([1,1])

    orientation_diff = observation[closest_agent].bearing + 2 * np.pi - agent_state[2]
    rotation_matrix = U.get_rotation_matrix(orientation_diff)
    return np.dot(rotation_matrix, np.array([1,0]))

def potential_fields(observation, internal_state):
    """
    Based on the given observation and the internal state of the agent, build the scaled vector
    which represents the potential force applied to this agent.
    :param observation:
    :param internal_state: Internal state of the agent.
    :return:
    """
    if len(observation) == 0:
        return random_action()

    potential_force = np.zeros((1,2))

    for agent_obs in observation:
        current_force = np.array([math.cos(agent_obs.bearing), math.sin(agent_obs.bearing)]) * \
                           ((agent_obs.dist - MIN_DIST) / agent_obs.dist)

        factor = ATTRACTION_FORCE if agent_obs.state.value == internal_state.value else REPULSION_FORCE
        potential_force += current_force * factor

    return potential_force



def random_action():
    return np.random.rand(1,2)





