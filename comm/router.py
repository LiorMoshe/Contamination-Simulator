from collections import namedtuple
from enum import Enum

class MessageType(Enum):
    OBS=1
    PROPOSE_CIRCLE=2 # Sent once we wish to propose a circle.
    APPROVE_CIRCLE=3 # Sent once we approve a given circle.
    CIRCLE_ESTABLISHED = 4 # Sent once a given circle is approved by all its participants.
    CONVERGENCE_STATE=5 # Sent when trying to converge to a circle.
    CIRCLE_STATE=6 # Kind of an ACK message that shows which agents are in the circle.
    MERGE_PROPOSAL=7 # Message that proposes merging two circles.
    EXTERIOR_INFO=8 # Message that shares info about the exterior.
    PUBLICIZE_CIRCLE=9 # Message that publicizes information about the current uniform circle that we are in.
    APPROVE_MERGE=10 # Message that a given circle approves a merge proposal of another circle.
    RANDOM_DIRECTION=11 # Message that contains a RANDOM WALK direction that is passed along the members of the circle.

# The different types of messages are written for ease of use.

# Generic Message, Contains info object, publisher, targets that this message is sent to, time of the message and its type.
Message = namedtuple("Message", "info publisher targets time type")

# General info for a message about a circle, contains its participants and the center of the circle.
CircleMessage = namedtuple("CircleMessage", "participants center")

# General message for CONVERGING phase that contains a set of all the agents that converged to their required locations.
ConvergenceMessage = namedtuple("ConvergenceMessage", "converged_agents")

# General message that is passed once a circle is established.
CircleEstablished = namedtuple("CircleEstablished", "participants")

# This message contains information about other circles that is passed between members of a uniform circle.
ExteriorInfo = namedtuple("ExteriorInfo", "foreign_circles proposals")

# This message describes a proposal of a merger between two circles. It contains info about the circle that is proposing
# the merge.
CircleMergeProposal = namedtuple("CircleMergeProposal", "participants center proposed_from proposed_to")


class Router(object):
    """
    The router is the communication manager of the simulation.
    It receives messages published by agents and allows other agents to read messages
    from other agents they can observe.
    """


    def __init__(self):
        self.agent_to_message = {}


    def get_agent_message(self, index):
        try:
            return self.agent_to_message[index]
        except KeyError:
            return None

    def publish_agent_message(self, index, message):
        self.agent_to_message[index] = message