import numpy as np

import external_policy as External
from adversary import Adversary
import utils as U

dynamics = ['point', 'unicycle', 'box2d', 'direct', 'unicycle_acc']


class World(object):
    def __init__(self, world_size, torus, agent_dynamic):
        self.nr_agents = None
        # world is square
        self.world_size = world_size
        # dynamics of agents
        assert agent_dynamic in dynamics
        self.agent_dynamic = agent_dynamic
        # periodic or closed world
        self.torus = torus
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # matrix containing agent states
        self.healthy_states = None
        # matrix containing landmark states
        self.contaminated_states = None
        # x,y of everything
        self.nodes = None
        self.distance_matrix = None
        self.angle_matrix = None
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.01
        self.action_repeat = 10
        self.timestep = 0
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        self.adversary = Adversary()


    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all healthy agents which we control.
    @property
    def healthy_agents(self):
        return {idx: agent for idx,agent in self.agents.items() if agent.internal_state.value == 1}
        # return [agent for agent in self.agents if agent.internal_state.value == 1]

    # Return all the contaminated agents which aren't controlled by our controller.
    @property
    def contaminated_agents(self):
        return {idx: agent for idx,agent in self.agents.items() if agent.internal_state.value == 2}
        # return [agent for agent in self.agents if agent.internal_state.value == 2]

    def update_agent_states(self):

        if len(self.healthy_agents) > 0:
            self.healthy_states = np.vstack([agent.get_position()
                                             for agent in self.healthy_agents.values()])
        else:
            self.healthy_states = None

        if len(self.contaminated_agents) > 0:
            self.contaminated_states = np.vstack([agent.get_position()
                                                  for agent in self.contaminated_agents.values()])
        else:
            self.contaminated_states = None
        self.states = np.vstack([self.agents[idx].get_state() for idx in range(len(self.agents))])

    def update_distance_angle(self):
        """
        Update the distance and angle matrices.
        :return:
        """
        self.nodes = np.vstack([self.agents[idx].get_position() for idx in range(len(self.agents))])
        # self.nodes = np.vstack(
        #     [self.healthy_states[:, 0:2],
        #      self.contaminated_states[:, 0:2]]) if self.contaminated_states is not None else self.healthy_states[:,
        #                                                                              0:2]
        # Compute the distance between each pair of agents.
        self.distance_matrix = U.get_distance_matrix(self.nodes,
                                                     torus=self.torus, world_size=self.world_size, add_to_diagonal=-1)

        # Compute relative angle between each pair of agents based on their given orientations.
        angles = np.vstack([U.get_angle(self.nodes, self.agents[idx].get_position(),
                                        torus=self.torus, world_size=self.world_size) - self.agents[idx].get_orientation()
                            for idx in range(len(self.agents))])
        angles_shift = -angles % (2 * np.pi)
        self.angle_matrix = np.where(angles_shift > np.pi, angles_shift - 2 * np.pi, angles_shift)


    def reset(self):
        self.timestep = 0

        for agent in self.agents.values():
            agent.reset(agent.get_state())

        self.update_distance_angle()

    def set_agent_states(self, internal_state, states):
        relevant_agents =  self.healthy_agents if internal_state == 1 else self.contaminated_agents
        for i, agent in enumerate(relevant_agents.values()):
            agent.set_state(states[i])

    def move_agents(self, agents, save_nodes=True):
        if self.agent_dynamic == 'direct':
            # Extract actions from the agents.
            actions = np.zeros([len(agents), 2])
            for idx, agent in enumerate(agents.values()):
                actions[idx, :] = agent.action

            # direct state manipulation, normalize the actions of all agents.
            action_norm = np.linalg.norm(actions, axis=1)
            scaled_actions = np.empty_like(actions)
            scaled_actions[:, 0] = np.where(action_norm <= 1, actions[:, 0],
                                            actions[:, 0] / action_norm)
            scaled_actions[:, 1] = np.where(action_norm <= 1, actions[:, 1],
                                            actions[:, 1] / action_norm)

            next_coord = np.vstack(
                [agent.get_position()] for agent in agents.values()) + actions  # * self.dt
            if self.torus:
                next_coord = np.where(next_coord < 0, next_coord + self.world_size, next_coord)
                next_coord = np.where(next_coord > self.world_size, next_coord - self.world_size, next_coord)
            else:
                next_coord = np.where(next_coord < 0, 0, next_coord)
                next_coord = np.where(next_coord > self.world_size, self.world_size, next_coord)

            agent_states_next = next_coord  # + np.ones((self.nr_pursuers, 2)) * np.random.rand(self.nr_pursuers, 2) * 1e-6
            for i, agent in enumerate(agents.values()):
                agent.set_position(agent_states_next[i, :])

            if save_nodes:
                self.nodes = agent_states_next

    def step(self):

        self.timestep += 1
        self.update_agent_states()

        # Use code in comment if you wish to use a local policy on each one of the agents.
        # for agent in self.contaminated_agents.values():
        #     # action = External.go_to_closest_neighbor(agent.get_state(), agent.internal_state.value,
        #     #                                          agent.get_observation(self.distance_matrix[agent.index, :],
        #     #                                                                self.angle_matrix[agent.index, :],
        #     #                                                                self.agents))
        #     # action = External.random_action()
        #     action = External.potential_fields(agent.get_observation(self.distance_matrix[agent.index, :],
        #                                                                    self.angle_matrix[agent.index, :],
        #                                                                    self.agents), agent.internal_state)
        #     # print(action)
        #
        #     next_coord = agent.get_position() + action #* self.dt
        #     if self.torus:
        #         next_coord = np.where(next_coord < 0, next_coord + self.world_size, next_coord)
        #         next_coord = np.where(next_coord > self.world_size, next_coord - self.world_size, next_coord)
        #     else:
        #         next_coord = np.where(next_coord < 0, 0, next_coord)
        #         next_coord = np.where(next_coord > self.world_size, self.world_size, next_coord)
        #     agent.set_position(next_coord)

            # self.contaminated_states[i, :] = agent.get_state()

        if len(self.healthy_agents) > 0:
            self.move_agents(agents=self.healthy_agents)

        self.adversary.gather_all()
        if len(self.contaminated_agents) > 0:
            self.move_agents(agents=self.contaminated_agents, save_nodes=False)

        # Update the distance and angle based on movement.
        self.update_distance_angle()