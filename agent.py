import logging
import math
from operator import itemgetter

import numpy as np
import random
from gym import spaces

import external_policy as External
from comm.router import *
from utils import get_slope, get_agent_arc_size, euclidean_dist, get_max_stable_cycle_size, get_max_dense_circle_size


class InternalState(Enum):
    HEALTHY=1
    CONTAMINATED=2

class CircleState(Enum):
    SINGLE=1
    CONVERGING=2
    CIRCLE=3



class CircleMode(Enum):
    """
    Once we are in a uniform circle we can adopt three different modes:
    1. DISCOVERY: In this mode each member of the circle shares what it can observe in the exterior. After a full cycle
    of DISCOVERY each member of the circle knows everything that goes on outside.
    2. COORDINATE: In this mode there is a vote going  on between the circle members on what we should do, after a full
    cycle of this mode each member of the circle knows the voting of all the members.
    3. MOVE: In this mode there is a coordinated movement toward a specific target.

    Initially, the circle is in DISCOVERY mode and it switches its mode in each cycle.
    """
    PUBLICIZE=0
    DISCOVERY=1
    COORDINATE=2
    MOVE=3

# An observation will contain the distance bearing and state of the agent that we observed.
Observation = namedtuple("Observation", "dist bearing state circle_state position index")


class DiscoveryState(object):
    """
    Each agent in CIRCLE state holds a discovery state object. This object contains info about all the foreign circles
    surrounding its current circle and the different merge proposals that were sent to this circle.
    """

    def __init__(self, foreign, proposals):
        self.foreign_circles = foreign
        self.proposals = proposals

class ContaminationAgent(object):
    """
    Description of the agent in the contamination problem. The agent will hold it's current position
    in the simulated 2D space and it's velocity in each axis.
    The agent can't explicitly communicate with other agents and it can observe agents which are at a distance
    between it's minimal and maximal observation radii.
    """

    def __init__(self, environment, state, agent_idx, robot_radius, router):
        self.min_obs_rad = environment.min_obs_rad
        self.max_obs_rad = environment.max_obs_rad
        self.torus = environment.torus
        self.world_size = environment.world_size
        self.internal_state = state
        self.index = agent_idx
        self.robot_radius = robot_radius
        self.circle_state = CircleState.SINGLE
        self.router = router
        self.time = 1

        # Position of our agent.
        self.pos = None

        # Velocity of the agent.
        self.vel = None

        # Orientation of the agent.
        self.orientation = None

        # Angular velocity.
        self.angular_velocity = None
        self.action = None

        # Rotation matrix which will be used to direct the actions based on the orientation of the agent.
        self.rotation_matrix = None

        # Current cluster the agent is in.
        self.cluster = None

        # Save for debugging purposes the target cluster.
        self.target_cluster = None
        self.action = np.array([0, 0])

        self.circle_info = None

        # The expected location of this agent on the uniform circle.
        self.circle_target_loc = None

        # Info about the last circle that we proposed via comm.
        self.last_proposed_circle = None

        # The circle is either in PUBLICIZE, DISCOVERY, COORDINATE, or MOVE mode.
        self.circle_mode = None

        # The state from the last discovery.
        self.discovery_state = None

        self.angle = None

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=+1, shape=(2,), dtype=np.float32)

    @property
    def observation_space(self):
        return spaces.Box(low=np.array([self.min_obs_rad, self.min_obs_rad, 0]),
                          high=np.array([self.max_obs_rad, self.max_obs_rad, 2 * np.pi]), shape=(3,), dtype=np.float32)

    def reset(self, state):
        self.set_state(state)
        self.vel = np.zeros_like(state)
        self.angular_velocity = 0

    def set_orientation(self, orientation):
        self.orientation = orientation

        # Rotation of moving vector counterclockwise in orientation degrees.
        r_matrix_2 = np.squeeze([[np.cos(orientation), -np.sin(orientation)],
                                 [np.sin(orientation), np.cos(orientation)]])

        # Computes the rotation matrix based on the given angle
        self.rotation_matrix = r_matrix_2

    def get_orientation(self):
        return np.array([self.orientation])

    def get_state(self):
        return np.concatenate((self.pos, np.array([self.orientation])))

    def set_state(self, state):
        self.set_position(state[:2])
        self.set_orientation(state[2])


    def set_position(self, pos):
        self.pos = np.squeeze(pos)

    def get_position(self):
        return self.pos

    def conceal_from_me(self,target,potential_obstacle):
        target = np.array(target)
        potential_obstacle = np.array(potential_obstacle)
        pos = np.array(self.pos)

        dist = np.linalg.norm(np.cross(pos - target, pos - potential_obstacle)) / np.linalg.norm(pos - target)
        return dist < self.robot_radius

    def does_any_conceals(self, target, candidates):
        """
        Checks whether there is any candidate that conceals a given target.
        """
        target = np.array(target)
        pos = np.array(self.pos)
        for candidate in candidates:
            np_candidate = np.array(candidate)
            dist = np.linalg.norm(np.cross(pos - target, pos - np_candidate)) / np.linalg.norm(pos - target)
            if dist < self.robot_radius:
                return True
        return False

    def get_observation(self, distance_matrix, angle_matrix, agents, org_idx=0,set=False):
        """
        Get the current observation of the agent based on the given distance matrix.
        The observation of the agent will be it's distance from each agent and bearing.
        Note: We don't keep index of agents that we see because we wish to have anonymity.
        :param distance_matrix: Distance from other agents.
        :param angle_matrix Angles relative to agents.
        :param List of all the agents in the environment.
        :return: A list of observations.
        """
        observation = []

        candidates = []
        in_area = []
        # if set:
        #     print("In Dist Mat: ",distance_matrix[org_idx])
        for idx, dist in enumerate(distance_matrix):

            if dist <= self.max_obs_rad and dist > 0:
                in_area.append(idx)
                if dist >= self.min_obs_rad:
                    candidates.append(idx)
            # if dist >= self.min_obs_rad and dist <= self.max_obs_rad:
            #     observation.append(Observation(dist, angle_matrix[idx], agents[idx].internal_state))

        for first_idx in candidates:
            # curr_agents = [agents[idx].pos for idx in in_area if idx != first_idx]
            # if not self.does_any_conceals(agents[first_idx].pos, curr_agents):
                observation.append(Observation(distance_matrix[first_idx],
                                               angle_matrix[first_idx], agents[first_idx].internal_state,
                                               agents[first_idx].circle_state, agents[first_idx].get_position(),
                                               first_idx))



        return observation



    def state_transition(self, observation):
        """
        Update the agent's state based on the given observation.
        If there is a majority of some internal state which isn't our agent's state than we
        will switch states.
        :param observation:
        :return:
        """
        healthy_cnt = contaminated_cnt = 0
        prev_state = self.internal_state

        if prev_state.value == 1:
            healthy_cnt += 1
        else:
            contaminated_cnt += 1

        for agent_obs in observation:
            if agent_obs.state.value == 1:
                healthy_cnt += 1
            else:
                contaminated_cnt += 1


        # if prev_state.value == 2:
        #     logging.info("Contamianted agent " + str(self.index) + " observes " + str(healthy_cnt) + " healthy and " +
        #                  str(contaminated_cnt) + " contaminated.")

        if healthy_cnt == contaminated_cnt:
            return

        self.internal_state = InternalState.HEALTHY if healthy_cnt > contaminated_cnt else InternalState.CONTAMINATED
        if prev_state.value != self.internal_state.value and self.is_allocated():
            self.free(notify=True)

    def allocate(self, cluster):
        """
        Allocate the agent to a given cluster.
        :param cluster:
        :return:
        """
        # print("Allocated agent {0} to cluster {1}".format(self.index, cluster._id))
        self.cluster = cluster

    def free(self, notify=False):
        """
        Remove the agent from it's allocated cluster.
        :return:
        """
        if notify:
            self.cluster.release(self.index)
        self.cluster = None

    def is_allocated(self):
        """
        Check whether the agent is allocated to a given cluster.
        :return:
        """
        return self.cluster is not None

    def get_cluster_id(self):
        if self.cluster is None:
            return None
        # print("Agent {0} GetClusterId {1}".format(self.index, self.cluster._id))
        return self.cluster._id

    def log_location(self):
        logging.info("Agent " + str(self.index) + " is in position: " + str(self.pos))


    def compute_circle_target_loc(self):
        """
        Find the index of the current agent in the circle based on its
        list of participants.
        :return:
        """
        print("Agent {0} Computing For Circle {1}".format(self.index, self.circle_info))
        target_circle_idx = self.find_agent_expected_circle_index(self.circle_info, self.index)
        num_agents = len(self.circle_info.participants)
        agent_arc = get_agent_arc_size(self.robot_radius, self.max_obs_rad / 2)
        jump = (2 * math.pi - num_agents * agent_arc) / num_agents + agent_arc
        radius = self.max_obs_rad / 2
        self.circle_target_loc = np.array([self.circle_info.center[0] + radius * math.cos(jump * target_circle_idx),
                            self.circle_info.center[1] + radius * math.sin(jump * target_circle_idx)])

    def accept_circle(self, acceptance_message):
        """
        Switch state to CONVERGING and move towards location of Smax diameter circle around the center of the locations
        of all the agents.
        This is only done after we got acceptance messages from all the other participants of the circle.
        :param acceptance_message:
        :return:
        """
        logging.info("Agent {0} agreed to join a circle with participants {1}".format(self.index,
                                                                                      acceptance_message.info.participants))
        self.circle_state = CircleState.CONVERGING
        self.circle_info = acceptance_message.info
        self.compute_circle_target_loc()
        self.move_to_target(self.circle_target_loc)

    def single_strategy(self, observation, messages):
        """
        First, we go over the messages that were sent to us to establish circles with some agents.
        If there are messages from SINGLE agents find the largest common subset of agents that they all observe.
        This is our largest clique.

        Once a proposal of a circle is sent we wait for the approval of all the participants, an establishment
        message is passed once there is approval of all the members of the circle.
        :param observation:
        :param messages:
        :return:
        """

        # The agents observed are the only ones we can send messages to.
        true_observed_agents = {obs.index for obs in observation if obs.state.value == self.internal_state.value}
        single_observed = {obs.index for obs in observation if obs.circle_state.value == self.circle_state.value}
        circle_observed = {obs.index for obs in observation if obs.circle_state.value == CircleState.CIRCLE.value}
        observed_agents = true_observed_agents.union({self.index})
        subset_size = 0
        subset = {}
        largest_proposal = -1
        proposal = []
        idx_to_pos = {}
        approvals = set()

        # Check if there are any messages. These will help us construct the largest clique in our observations.
        # We only get messages from agents that can observe us.
        logging.info("Single Agent: {0} Num Messages: {1}".format(self.index, len(messages)))
        common_sets = {}
        proposals = {}
        for message in messages:
            # Only look at messages from SINGLE agents
            logging.info("Agent {0} Got Message {1}".format(self.index, message))
            # if message.publisher.circle_state.value == self.circle_state.value:

            # Collect data from observational messages. These messages contain data of observations of others.
            if message.type.value == MessageType.OBS.value:
                agent_observations = message.info
                idx_to_pos[message.publisher.index] = message.publisher.get_position()

                # Save the position of each observed agent.
                #TODO-Maybe this can be removed
                for other_obs in agent_observations:
                    idx_to_pos[other_obs.index] = other_obs.position

                # Get the indices of the intersecting set of agents observed between this agent and the other one.
                other_observed_agents = {message.publisher.index}
                other_observed_agents = other_observed_agents.union({other_obs.index for other_obs in agent_observations})
                intersecting_obs= observed_agents.intersection(other_observed_agents)
                common_sets[len(intersecting_obs)] = intersecting_obs
            elif message.type.value == MessageType.PROPOSE_CIRCLE.value and \
                self.index in message.info.participants:
                # Circle proposal message holds a potential list of participants in its info.
                proposals[len(message.info.participants)] = message.info
            elif message.type.value == MessageType.APPROVE_CIRCLE.value and \
                self.index in message.info.participants:
                # Immediately accept circle invitation.
                approvals = approvals.union({message.publisher.index})
            elif message.type.value == MessageType.CIRCLE_ESTABLISHED.value and \
                self.index in message.info.participants:
                self.accept_circle(message)
                return
            elif message.type.value == MessageType.APPROVE_MERGE.value and \
                self.index in message.info.participants:
                self.accept_circle(message)
                return


        # Get the largest common set.
        if len(common_sets) > 0:
            subset_size, subset = max(common_sets.items(), key=itemgetter(0))

        # Get the proposal of largest size.
        if len(proposals) > 0:
            largest_proposal, proposal = max(proposals.items(), key=itemgetter(0))

        # If we proposed a circle in the previous round, check if there is a full approval.
        if self.last_proposed_circle is not None:
            if len(approvals) == len(self.last_proposed_circle.participants) - 1:
                logging.info("Agent {0} Established Circle {1}".format(self.index, self.last_proposed_circle.participants))
                establishment = Message(self.last_proposed_circle, publisher=self,targets=observed_agents,
                                           time=self.time, type=MessageType.CIRCLE_ESTABLISHED)
                self.publish_message(establishment)
                self.accept_circle(establishment)
                return
            elif len(approvals) > 0:
                approvals = approvals.union({self.index})
                updated_proposal = CircleMessage(approvals, self.last_proposed_circle.center)
                logging.info("Agent {0} updating its proposal to contain {1}".format(self.index,
                                                                                     updated_proposal.participants))
                proposal = Message(updated_proposal, self, observed_agents, self.time, MessageType.PROPOSE_CIRCLE)
                self.publish_message(proposal)
                self.last_proposed_circle = updated_proposal
                return
            else:
                logging.info("Agent {0} stopped proposing".format(self.index))
                self.last_proposed_circle = None


        # Check if there is a pre-existing proposal which is better
        logging.info("SINGLE {0} SubsetSize {1} ProposalSize {2} ".format(self.index, subset_size, largest_proposal))
        if largest_proposal >= subset_size and largest_proposal > 1:
            # Send acceptance message.
            circle_desc = CircleMessage(participants=proposal.participants, center=proposal.center)

            logging.info("Agent {0} approved circle {1}".format(self.index, proposal.participants))
            # Targets of this message are only agents in its observation area.
            acceptance_message = Message(circle_desc, self, observed_agents,
                                         self.time, MessageType.APPROVE_CIRCLE)
            self.publish_message(acceptance_message)
            # self.accept_circle(acceptance_message)
            return

        if subset_size < 1:
            logging.info("Single Agent {0} NoSubset ObservedAgents {1} SingleObserved {2}".format(self.index, observed_agents,
                                                                                                  single_observed))
            if len(observed_agents) == 1:
                self.action = External.random_action()
            elif len(single_observed) > 0:
                # Construct OBS message to send to others.
                message = Message(observation, self, list(observed_agents), self.time, MessageType.OBS)
                self.publish_message(message)
            else:
                # Send Merge proposal.
                proposal = CircleMergeProposal({self.index}, self.get_position(),
                                               self.index, circle_observed)
                self.publish_message(Message(proposal, self, true_observed_agents, self.time, MessageType.MERGE_PROPOSAL))



        else:
            # If there is a clique, send proposal message, compute the center and send message to all participants.
            center = np.array([0,0], dtype=np.float64)
            for idx in subset:
                center += idx_to_pos[idx]
            center /= len(subset)
            logging.info("Agent {0} Proposing subset {1}".format(self.index, subset))
            proposal_message = CircleMessage(participants=subset, center=center)
            self.last_proposed_circle = proposal_message
            self.publish_message(Message(proposal_message, self, observed_agents, self.time,
                                         MessageType.PROPOSE_CIRCLE))


    def move_to_target(self, target, pace=0.1):
        """
        Move the agent to the given target at the given pace.
        :param target:
        :param pace:
        :return:
        """
        if abs(self.get_position()[0] - target[0]) < 1e-10:
            theta = 0
        else:
            slope = get_slope(target[0], target[1], self.get_position()[0], self.get_position()[1])
            # logging.info("Got slope: " + str(slope))
            theta = math.atan(slope)

        factor = -1 if self.get_position()[0] > target[0] else 1
        # logging.info("Setting action: " + str(np.array([math.cos(theta), math.sin(theta)]) * factor * pace))
        self.action = np.array([math.cos(theta), math.sin(theta)]) * factor * pace



    def find_agent_expected_circle_index(self, circle_info, agent_idx):
        """
        Given an agent's index and info about a certain circle, return the expected location
        of the agent in the given location.
        :param circle_info:
        :param agent_idx:
        :return:
        """
        for circle_idx, other_agent_idx in enumerate(circle_info.participants):
            if other_agent_idx == agent_idx:
                return circle_idx

    def get_observed_agents_indices(self, observation):
        observed_agents = set()
        for obs in observation:
            if obs.index != self.index:
                observed_agents.add(obs.index)
        return observed_agents

    def converging_strategy(self, observation, messages):
        """
        Converge to the required uniform circle.
        Switch to CIRCLE state only when we know for sure that all the other participants of the circle got to their
        required locations.
        Each agent in the circle bypasses information about what it knows about the members of the circle it observes.
        Once we have evidence that all the participants are at their required locations, switch to CIRCLE.
        :param observation:
        :param messages:
        :return:
        """
        observed_agents = self.get_observed_agents_indices(observation)

        known_converged = set()
        for message in messages:
            if message.type.value == MessageType.CONVERGENCE_STATE.value and \
                message.publisher.index in self.circle_info.participants:
                # The message info contains known converged agents.
                # print(message.info)
                known_converged = known_converged.union(message.info.converged_agents)


        dist_from_target = euclidean_dist(self.get_position(), self.circle_target_loc)
        print("Index:{0} Dist from target {1}".format(self.index, dist_from_target))
        if dist_from_target > 0.1:
            self.move_to_target(self.circle_target_loc)
        else:
            known_converged.add(self.index)
            self.action = np.array([0, 0])


        convergence_state = ConvergenceMessage(known_converged)
        print("Known Converged: {0}, Participants: {1}".format(known_converged, self.circle_info.participants))
        self.publish_message(Message(convergence_state, self, observed_agents, self.time, MessageType.CONVERGENCE_STATE))
        if len(known_converged) == len(self.circle_info.participants):
            print("Agent {0} SWITCHED TO CIRCLE MODE CIRCLECIRCLECIRCLE".format(self.index))
            self.circle_state = CircleState.CIRCLE
            self.last_proposed_circle = None

            # Initially, the members of the circle are in PUBLICIZE mode.
            self.circle_mode = CircleMode.PUBLICIZE



    def circle_strategy(self, observation, messages, threhold):
        """
        This is the strategy executed by each agent which is already in a given circle.
        The hard part of this strategy is how can we merge circles together. Coordinating a circle requires a full
        time step of the game since each agent in the circle can only send information to the agents
        in the circle it observes.

        In DISCOVERY mode we send info about the circle to foreign agents and inform our own circle members
        of the proposals and info we know about other circles.

        A circle proposes to another circle to merge only after there is a full agreement of all the members of the circle
        following the COORDINATE mode of the circle.
        :param observation:
        :param messages:
        :return:
        """
        observed_agents = self.get_observed_agents_indices(observation)
        logging.info("Circle: Agent {0} Mode {1} Members {2} Observing {3}".format(self.index, self.circle_mode,
                                                                                   self.circle_info.participants, observed_agents))
        # Info about proposals and other circles we observe.
        if self.circle_mode.value == CircleMode.PUBLICIZE.value:
            self.discovery_state = DiscoveryState(list(), list())

        for message in messages:
            if message.type.value == MessageType.MERGE_PROPOSAL.value and \
                self.index in message.info.proposed_to:
                # Save this proposal.
                self.discovery_state.proposals += [message.info]

                # If we got a proposal from someone we proposed to send APPROVE message and converge.
                if self.last_proposed_circle is not None and \
                        message.publisher.index in self.last_proposed_circle.proposed_to:

                    logging.info("Circle Agent {0} Got Proposal From Proposee, Its members {1}".format(self.index,
                                                                                                        self.circle_info.participants))
                    updated_circle = CircleMessage(self.circle_info.participants.union(message.info.participants),
                                                   (self.circle_info.center + message.info.center) / 2)
                    approval = Message(updated_circle, self, observed_agents, self.time, MessageType.APPROVE_CIRCLE)
                    self.publish_message(approval)
                    self.accept_circle(approval)
                    return


            elif message.type.value == MessageType.EXTERIOR_INFO.value and \
                message.publisher.index in self.circle_info.participants:
                # Append exterior info that is given to us by members of the circle.
                self.discovery_state.proposals += message.info.proposals
                self.discovery_state.foreign_circles += message.info.foreign_circles

            elif message.type.value == MessageType.APPROVE_MERGE.value and \
                    self.index in message.info.participants:
                # If there is approval move to converging.
                logging.info("Circle Agent {0} Got Approval from agent {1} Message {2}".format(self.index,
                                                                                   message.publisher.index,
                                                                                               message))
                self.publish_message(Message(message.info, self, observed_agents, self.time, MessageType.APPROVE_MERGE))
                self.accept_circle(message)
                return

            elif message.type.value == MessageType.PUBLICIZE_CIRCLE.value and \
                message.publisher.index not in self.circle_info.participants:
                print("Message: ",message.info)
                print(self.discovery_state.foreign_circles)
                self.discovery_state.foreign_circles += [message.info]

            elif message.type.value == MessageType.RANDOM_DIRECTION.value  and \
                message.publisher.index in self.circle_info.participants and self.circle_mode.value == CircleMode.MOVE.value:
                self.publish_message(Message(message.info, self, observed_agents, self.time, MessageType.RANDOM_DIRECTION))
                self.move_in_angle(message.info)
                self.switch_circle_mode()
                return



        logging.info("Info: Current DiscState: Foreign: {0} Proposals: {1}" .format(self.discovery_state.foreign_circles
                                                                                    ,self.discovery_state.proposals))

        if self.circle_mode.value == CircleMode.PUBLICIZE.value:
            self.publish_message(Message(self.circle_info, self, observed_agents, self.time, MessageType.PUBLICIZE_CIRCLE))
        elif self.circle_mode.value == CircleMode.DISCOVERY.value:
            # In Discovery mode we send info about circles we observed and proposals we got.
            exterior_info = ExteriorInfo(self.discovery_state.foreign_circles, self.discovery_state.proposals)
            self.publish_message(Message(exterior_info, self, observed_agents, self.time, MessageType.EXTERIOR_INFO))
        elif self.circle_mode.value == CircleMode.COORDINATE.value:
            # In coordinate mode we send messages once all the info is fully synchronized.

            # Find largest foreign circle.
            largest_foreign = None
            largest_circle_size = 0
            for circle in self.discovery_state.foreign_circles:
                if len(circle.participants) > largest_circle_size:
                    largest_circle_size = len(circle.participants)
                    largest_foreign = circle

            # Find proposal from largest circle.
            largest_proposal = None
            largest_proposal_size = -1
            for proposal in self.discovery_state.proposals:
                if len(proposal.participants) > largest_proposal_size:
                    largest_proposal_size = len(proposal.participants)
                    largest_proposal = proposal

            # Prefer merging with circle than proposing a new merger.

            if largest_proposal_size >= largest_circle_size and \
                    largest_proposal_size + len(self.circle_info.participants) < threhold:
                if largest_proposal.proposed_from in observed_agents:
                    # Send APPROVE_CIRCLE message to the other agent.
                    updated_circle = CircleMessage(self.circle_info.participants.union(largest_proposal.participants),
                                                   (self.circle_info.center + largest_proposal.center) / 2)
                    approval = Message(updated_circle, self, observed_agents, self.time, MessageType.APPROVE_MERGE)
                    logging.info(
                        "Circle Agent {0} Sending Approval to {1} Message: {2}".format(self.index, largest_proposal.proposed_from,
                                                                                       approval))
                    self.publish_message(approval)
                    self.accept_circle(approval)
                    return
            elif largest_circle_size > 0:
                # Send proposal to the other circle if we observe a member of it.
                observed_members = observed_agents.intersection(largest_foreign.participants)
                if len(observed_members) > 0:
                    # Propose the circle.
                    logging.info("Circle Agent {0} Sending proposal to merge to circle {1}".format(self.index,
                                                                                                   largest_foreign.participants))
                    proposal = CircleMergeProposal(self.circle_info.participants, self.circle_info.center,
                                                   self.index, observed_members)
                    self.publish_message(Message(proposal, self, observed_agents, self.time, MessageType.MERGE_PROPOSAL))

                    self.last_proposed_circle = proposal




        else:
            # In this case we move randomly in a formation. Pass Random angle and magnitude.
            self.continuous_angle()
            self.publish_message(Message(self.angle, self,observed_agents,self.time, MessageType.RANDOM_DIRECTION))
            self.move_in_angle(self.angle)

        self.switch_circle_mode()

    def init_angle(self):
        self.angle = External.random_angle()
        self.chance = 0.99

    def continuous_angle(self):
        if self.angle is None:
            self.init_angle()
        else:
            coin = random.random()
            if coin > self.chance:
                self.init_angle()
            self.chance -= 0.01


    def move_in_angle(self, angle, pace=0.5):
        self.move_to_target(self.get_position() + pace * np.array([math.cos(angle), math.sin(angle)]), pace=pace)

    def switch_circle_mode(self):
        self.circle_mode = CircleMode((self.circle_mode.value + 1) % len(CircleMode))

    def distributed_strategy(self, observation, clique_mode=False):
        """
        Based on its current circular state, execute the distributed strategy.
        Set the current action of the agent based on the given strategy.
        Some Notes:
        1. In theory, we can compute the position of any agent given its distance and bearing. To save ourselves time
        we write the position inside the observation under the assumptions that all the agents share the same coordinate
        system.
        2. We can hold some mapping between position and index but to save time we just kept the index of the
        agent in the observation. This completely throws away the assumption on anonymity that was previously held.
        :return:
        """
        # Collect messages from router.
        logging.info("Agent {0} CircleState: {1}".format(self.index, self.circle_state))

        # Zero out the action in each axis so we won't repeat the action that was done previously for no reason.
        self.action = np.array([0,0])

        # Get the messages that were sent to the agent in the last time step.
        messages = self.get_messages(observation)

        clique_size = get_max_stable_cycle_size(self.min_obs_rad, self.max_obs_rad)
        odc_size = get_max_dense_circle_size(self.min_obs_rad, self.robot_radius)

        # Act according to the agents circle_state.
        if self.circle_state == CircleState.SINGLE:
            self.single_strategy(observation, messages)
        elif self.circle_state == CircleState.CONVERGING:
            self.converging_strategy(observation, messages)
        else:
            self.circle_strategy(observation, messages, odc_size if not clique_mode else clique_size)

    def advance_time(self):
        self.time += 1

    def publish_message(self, message):
        logging.info("Agent {0} Sending Message to {1}".format(self.index, message.targets))
        self.router.publish_agent_message(self.index, message)

    def get_messages(self, observation):
        """
        Get the messages from all the agents I observe.
        :return:
        """
        messages = []
        logging.info("Agent {0} Getting Messages, Observing {1}".format(self.index, [obs.index for obs in observation]))
        for obs in observation:
            message = self.router.get_agent_message(obs.index)

            # Only collect a message if it was targeted to us.
            if message is not None and self.index in message.targets:
                messages.append(message)
        return messages


