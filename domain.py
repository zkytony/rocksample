import pomdp_py
import random

class RockType:
    GOOD = 'good'
    BAD = 'bad'
    @staticmethod
    def invert(rocktype):
        if rocktype == 'good':
            return 'bad'
        else:
            return 'good'
        # return 1 - rocktype 
    @staticmethod
    def random(p=0.5):
        if random.uniform(0,1) >= p:
            return RockType.GOOD
        else:
            return RockType.BAD

class State(pomdp_py.State):
    def __init__(self, position, rocktypes, terminal=False):
        """
        position (tuple): (x,y) position of the rover on the grid.
        rocktypes: tuple of size k. Each is either Good or Bad.
        terminal (bool): The robot is at the terminal state.

        (It is so true that the agent's state doesn't need to involve the map!)

        x axis is horizontal. y axis is vertical.
        """
        self.position = position
        if type(rocktypes) != tuple:
            rocktypes = tuple(rocktypes)
        self.rocktypes = rocktypes
        self.terminal = terminal
        
    def __hash__(self):
        return hash((self.position, self.rocktypes, self.terminal))
    def __eq__(self, other):
        if isinstance(other, State):
            return self.position == other.position\
                and self.rocktypes == other.rocktypes\
                and self.terminal == other.terminal
        else:
            return False
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "State(%s | %s | %s)" % (str(self.position), str(self.rocktypes), str(self.terminal))

class Action(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Action(%s)" % self.name


class MoveAction(Action):
    EAST = (1, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0)
    NORTH = (0, 1)
    SOUTH = (0, -1)
    def __init__(self, motion):
        if motion not in {MoveAction.EAST, MoveAction.WEST,
                          MoveAction.NORTH, MoveAction.SOUTH}:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        super().__init__("move-%s" % str(motion))

MoveEast = MoveAction(MoveAction.EAST)
MoveWest = MoveAction(MoveAction.WEST)
MoveNorth = MoveAction(MoveAction.NORTH)
MoveSouth = MoveAction(MoveAction.SOUTH)

class SampleAction(Action):
    def __init__(self):
        super().__init__("sample")

class CheckAction(Action):
    def __init__(self, rock_id):
        self.rock_id = rock_id
        super().__init__("check-%d" % self.rock_id)

class Observation(pomdp_py.Observation):
    def __init__(self, quality):
        self.quality = quality
    def __hash__(self):
        return hash(self.quality)
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.quality == other.quality
        elif type(other) == str:
            return self.quality == other
    def __str__(self):
        return str(self.quality)
    def __repr__(self):
        return "Observation(%s)" % str(self.quality)
