# Handcrafted planner that goes to each rock,
# samples it, and collect if it is good.
import pomdp_py
import math
from domain import *

# Utility functions
def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

class HandcraftPlanner(pomdp_py.Planner):
    """This planner will move the robot to the closest unchecked rock,
    check it. If good, sample. Do so for every rock that is not on
    the west of the robot.
    """

    def __init__(self, rock_locs):
        self._rock_locs = rock_locs
        self._next_rock_checked = False
        self._next_rock_pos = None  # next rock to sample        
        self._checked_rocks = set()

    def plan(self, agent):
        mpe_state = agent.belief.mpe()
        robot_pos = mpe_state.position

        if self._next_rock_pos is not None:

            # If robot is on top of rock, then check;
            # If checked, then sample
            if robot_pos == self._next_rock_pos:
                if self._next_rock_checked:
                    next_rock_id = self._rock_locs[self._next_rock_pos]
                    rock_type = mpe_state.rocktypes[next_rock_id]
                    self._next_rock_pos = None  # reset
                    self._next_rock_checked = False                    
                    if rock_type == RockType.GOOD:
                        # sample
                        return SampleAction()
                    else:
                        return self.plan(agent)
                else:
                    # Check
                    self._next_rock_checked = True
                    next_rock_id = self._rock_locs[self._next_rock_pos]
                    return CheckAction(next_rock_id)

            else:
                # We will return the move action that
                # moves the robot closer to this rock; will
                # not move west.
                for move_action in [MoveNorth, MoveSouth, MoveEast]:
                    next_robot_pos = (robot_pos[0] + move_action.motion[0],
                                      robot_pos[1] + move_action.motion[1])
                    if euclidean_dist(next_robot_pos, self._next_rock_pos)\
                       < euclidean_dist(robot_pos, self._next_rock_pos):
                        return move_action
                # If nothing works, move east
                return MoveEast

        else:
            # Assign next rock that is on the east of the robot or on
            # the same x coordinate; then plan again. If we have
            # picked up all rocks, just exit.
            
            min_dist, chosen_rock_pos = float('inf'), None
            for rock_loc in self._rock_locs:
                rock_id = self._rock_locs[rock_loc]
                if rock_id in self._checked_rocks:
                    continue
                if rock_loc[0] >= robot_pos[0]:
                    # getting the that rock doesn't require the
                    # robot to go backward (i.e. to west)
                    rock_dist = euclidean_dist(rock_loc, robot_pos)
                    if rock_dist < min_dist:
                        min_dist = rock_dist
                        chosen_rock_pos = rock_loc
            if chosen_rock_pos is None:
                return MoveEast
            else:
                self._next_rock_pos = chosen_rock_pos
                self._next_rock_checked = False
                return self.plan(agent)
                

    def update(self, agent, action, real_observation):
        # Update robot belief.
        new_belief = pomdp_py.update_particles_belief(agent.belief,
                                                      action, real_observation,
                                                      agent.observation_model,
                                                      agent.transition_model)
        agent.set_belief(new_belief)
        # Update checked rocks.
        if isinstance(action, CheckAction):
            self._checked_rocks.add(action.rock_id)

        
