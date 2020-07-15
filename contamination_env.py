from gym import Env
import base
from agent import ContaminationAgent, InternalState
import numpy as np
# import os
import matplotlib.pyplot as plt
import shutil
from collections import namedtuple
import matplotlib.transforms as mtrans
from simulation_data import SimulationData, represent_as_box_plot, to_dataframe
from global_actor import GlobalActor
import logging
from clusters.cluster_manager import ClusterManager

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

GlobalState = namedtuple("GlobalState", "healthy_agents contaminated_agents distance_matrix angle_matrix")

total_healthy_wins = 0


class ContaminationEnv(Env):
    """
    Implementation of the contamination environments using the API defined by OpenAI Gym, we will assume that we only
    control the healthy agents whereas the contaminated agents are controlled by external unknown policies.
    """
    metadata = {'render.modes': ['human', 'animate']}

    def __init__(self, robot_radius,num_healthy, num_contaminated, world_size,
                 min_obs_rad, max_obs_rad, torus=False, stop_on_win=True, dynamics='direct'):
        Env.__init__(self)
        self.robot_radius = robot_radius
        self.num_healthy = num_healthy
        self.num_contaminated = num_contaminated
        self.world_size = world_size
        self.min_obs_rad = min_obs_rad
        self.max_obs_rad = max_obs_rad
        self.stop_on_win = stop_on_win
        self.torus = torus
        self.world = base.World(world_size, torus, dynamics)
        self.global_actor = GlobalActor(min_obs_rad, max_obs_rad)
        ClusterManager(self.min_obs_rad, self.max_obs_rad, robot_radius)
        self.reset()



    def render(self, mode='human', debug=False):
        if self.plot and self.winner is not None and self.stop_on_win:
            self.plot.text(self.world_size / 2 - 20, self.world_size + 10,
                           'Game Over. Winner is ' + str(self.winner),
                           bbox=dict(facecolor='red' if self.winner == InternalState.CONTAMINATED.name else 'blue'
                                     , alpha=0.5),
                           fontsize=14)
            plt.pause(0.1)
            return

        if mode == 'animate':
            output_dir = "/tmp/video/"
            if self.timestep == 0:
                import os
                try:
                    shutil.rmtree(output_dir)
                except FileNotFoundError:
                    pass
                os.makedirs(output_dir, exist_ok=True)

        if not self.plot:
            fig, ax = plt.subplots(figsize=(50, 100))
            ax.set_aspect('equal')
            ax.set_xlim((0, self.world_size))
            ax.set_ylim((0, self.world_size))
            self.plot = ax

        else:
            self.plot.clear()
            self.plot.set_aspect('equal')
            self.plot.set_xlim((0, self.world_size))
            self.plot.set_ylim((0, self.world_size))

        min_obs_circles = []
        max_obs_circles = []

        if self.world.contaminated_states is not None:
            self.plot.scatter(self.world.contaminated_states[:, 0], self.world.contaminated_states[:, 1], c='r', s=1,
                              zorder=2)

        if len(self.world.healthy_agents) > 0:
            self.plot.scatter(self.world.healthy_states[:, 0], self.world.healthy_states[:, 1], c='b', s=0.25,
                              zorder=2)

        for i in range(self.num_healthy + self.num_contaminated):
            min_obs_circles.append(plt.Circle((self.world.states[i, 0],
                                       self.world.states[i, 1]),
                                      self.min_obs_rad, color='r', fill=False, zorder=1))
            self.plot.add_artist(min_obs_circles[i])

            max_obs_circles.append(plt.Circle((self.world.states[i, 0],
                                            self.world.states[i, 1]),
                                           self.max_obs_rad, color='g', fill=False, zorder=1))
            self.plot.add_artist(max_obs_circles[i])

        #Render the arrow which show the orientation of the agents.
        m2 = np.array((0, 1))
        s1 = np.array((0.5, 1.8))
        s2 = np.array((30, 30))
        kw = dict(xycoords='data', textcoords='offset points', size=10)

        # Use if we wish to plot the number of cluster for each agent group.
        # for agent in self.world.healthy_agents.values():
        #     plot_str = str(agent.get_cluster_id())
        #     if debug:
        #         plot_str += " to " + str(agent.target_cluster)
        #
        #     rot = mtrans.Affine2D().rotate(agent.orientation)
        #     self.plot.annotate(plot_str, xy=agent.get_position() - rot.transform_point(m2 * s1),
        #                 xytext=rot.transform_point(m2 * s2),color='b', **kw)
        #
        #
        # for agent in self.world.contaminated_agents.values():
        #     plot_str = str(agent.get_cluster_id())
        #     if debug:
        #         plot_str += " to " + str(agent.target_cluster)
        #
        #     rot = mtrans.Affine2D().rotate(agent.orientation)
        #     self.plot.annotate(plot_str, xy=agent.get_position() - rot.transform_point(m2 * s1),
        #                 xytext=rot.transform_point(m2 * s2),color='r', **kw)



        # In debug mode present the clusters of adversaries.
        if debug:
            for agent in self.world.contaminated_agents.values():
                rot = mtrans.Affine2D().rotate(agent.orientation)
                self.plot.annotate(str(agent.get_cluster_id()), xy=agent.get_position() - rot.transform_point(m2 * s1),
                               xytext=rot.transform_point(m2 * s2), **kw)

        # Present number of healthy and contaminated agents in a textbox.
        textstr = '\n'.join(('Healthy: ' + str(len(self.world.healthy_agents)),
                             'Contaminated: ' + str(len(self.world.contaminated_agents))))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        self.plot.text(-0.4, 0.95, textstr, transform=self.plot.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)


        if mode == 'human':
            plt.pause(0.01)
        elif mode == 'animate':
            if self.timestep % 1 == 0:
                plt.savefig(output_dir + format(self.timestep//1, '04d'))

            if self.is_terminal:
                import os
                os.system("ffmpeg -r 10 -i " + output_dir + "%04d.png -c:v libx264 -pix_fmt yuv420p -y /tmp/out.mp4")

    def reset(self):
        """
        Reset our environment and return list of observations of all the agents.
        :return:
        """
        self.timestep = 0
        self.winner = None
        self.plot = None
        self.global_actor.reset()
        ClusterManager.instance.reset()
        self.sim_data = SimulationData()
        healthy_agents = {idx: ContaminationAgent(self, InternalState.HEALTHY, idx, self.robot_radius) for idx in range(self.num_healthy)}
        contaminated_agents = {idx: ContaminationAgent(self, InternalState.CONTAMINATED, idx, self.robot_radius) for idx in
                               range(self.num_healthy, self.num_healthy + self.num_contaminated)}
        self.world.agents = {**healthy_agents, **contaminated_agents}

        # Generate initial locations for the groups of agents.
        self.world.set_agent_states(InternalState.HEALTHY.value,
                                    ContaminationEnv.generate_initial_location(self.num_healthy, self.world_size))
        self.world.set_agent_states(InternalState.CONTAMINATED.value,
                                    ContaminationEnv.generate_initial_location(self.num_contaminated, self.world_size))
        self.world.reset()
        return self.get_observations()

    @staticmethod
    def generate_initial_location(num_agents, world_size):
        """
        Compute a random initial location and orientation for the agents in the contamination environment.
        We generate only (x,y) coordinates of the agents, there is no meaning to orientation in our problem, we will
        also generate a random orientation for each agent.
        :return:
        """
        states = np.random.rand(num_agents, 3)
        states[:, :2] = world_size * ((0.95 - 0.05) * states[:, :2] + 0.05)
        states[:, 2:3] = 2 * np.pi *  states[:, 2:3]
        return states

    def get_observations(self, transition=False):
        """
        Get the observations of all the agents in our environment.
        We will only return the observations from the healthy agents which we control.
        :param transition True if we should perform state transition based on the observation of each agent.
        :return:
        """
        observations = {}
        healthy_observations = []

        for idx, agent in self.world.agents.items():
            curr_observation = agent.get_observation(self.world.distance_matrix[idx, :],
                                                      self.world.angle_matrix[idx, :],
                                                      self.world.agents)

            if transition:
                observations[idx] = curr_observation

            # Collect observations of the healthy agents.
            if agent.internal_state.value == 1:
                healthy_observations.append(curr_observation)

        if transition:
            for idx, observation in observations.items():
                self.world.agents[idx].state_transition(observation)

        return healthy_observations

    @property
    def timestep_limit(self):
        return 1024

    @property
    def is_terminal(self):
        return self.timestep >= self.timestep_limit or self.winner is not None


    def step(self, actions=None, plot=True):
        """
        :param actions:
        :return: Observations of all the agents.
        Reward which can be specific for each agent
        or global.
        Done - Whether we should reset the environment.
        info - diagnostic information for debugging.
        """
        if self.winner is not None and plot and self.stop_on_win:
            self.sim_data.plot()
            return [], 1, True, {}

        self.timestep += 1

        if actions is not None:
            assert len(actions) == self.num_healthy
            clipped_actions = np.clip(actions, self.world.agents[0].action_space.low, self.world.agents[0].action_space.high)

            for agent, action in zip(self.world.agents.values(), clipped_actions):
                agent.action = action
        else:
            # Compute the clusters which will be used by the global players.
            ClusterManager.instance.update_clusters(self.global_state)
            # self.global_actor.gather_conquer_act()
            # self.global_actor.strategic_movement(self.global_state)
            self.global_actor.odc_strategic_movement(self.global_state)



        # This is where the opponent chooses its move.
        self.world.step(self.global_state)
        next_observations = self.get_observations(transition=True)

        if len(self.world.healthy_agents) == 0:
            print("CONT WON")
            self.winner = InternalState.CONTAMINATED.name
        elif len(self.world.contaminated_agents) == 0:
            global total_healthy_wins
            total_healthy_wins += 1
            print("HEALTHY WON")
            self.winner = InternalState.HEALTHY.name


        # TODO- Inspect different reward mechanisms.
        reward = 1
        info = {'healthy_states': self.world.healthy_states,
                'contaminated_states': self.world.contaminated_states,
                # 'state': np.vstack([self.world.healthy_states[:, 0:2], self.world.contaminated_states[:, :2]]),
                'actions': actions}

        self.sim_data.add_data(self.timestep, len(self.world.healthy_agents), len(self.world.contaminated_agents))
        return next_observations, reward, self.is_terminal, info

    @property
    def global_state(self):
        return GlobalState(self.world.healthy_agents, self.world.contaminated_agents,
                           self.world.distance_matrix, self.world.angle_matrix)

    @property
    def observation_space(self):
        pass

    @property
    def action_space(self):
        pass

    @property
    def state_space(self):
        pass


if __name__=="__main__":
    env = ContaminationEnv(0.1,35, 35, 100, 2, 6, stop_on_win=True)
    num_episodes = 50

    simulations_data = []

    for num_episode in range(num_episodes):
        print("Episode {0}".format(num_episode))
        obs = env.reset()
        for t in range(1024):
            a = np.vstack([np.array([1, 1]) for _ in range(20)])
            o, rew, dd, _ = env.step(plot=False)

            # Use if we wish to gather data.
            # if env.winner is not None:
            #     simulations_data.append(env.sim_data)
            #     break
            #
            env.render(debug=False)

    # Use to represent data in series of box plots.
    # print("Total Healthy Wins: {0}, Number of Episodes: {1}, Winning Percentage: {2}".format(total_healthy_wins, num_episodes,
    #                                                                                          float(total_healthy_wins) / num_episodes))
    # represent_as_box_plot(to_dataframe(simulations_data, save=True))