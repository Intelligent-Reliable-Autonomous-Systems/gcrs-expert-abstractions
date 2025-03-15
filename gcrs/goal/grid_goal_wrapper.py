"""This module contains GridGoalWrapper and subclasses for providing reward based on grid abstraction subgoals."""

from cocogrid.common.cocogrid_state import CocogridState
from cocogrid.common.abstraction import GridAbstraction
from cocogrid.utils.visualize.drawing import get_camera_bounds, draw_rectangle
from cocogrid.common.color import get_color_rgba_255
from cocogrid.common.entity import get_color_name
import numpy as np

from gcrs.goal.goal_wrapper import GoalWrapper, AbstractState, Observation


class GridGoalWrapper(GoalWrapper):
    """A goal wrapper for the grid abstraction. Add visualization of the subgoal."""

    def __init__(self, *args, gamma=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def render(self):
        image = super().render()
        if image is not None:
            bounds = get_camera_bounds(self, norm_scale=True)
            pos = self.subgoal.walker_pos

            if self.subgoal._held_object == -1:
                color = np.array([255, 128, 40])
            else:
                obj = self.subgoal.objects[self.subgoal._held_object]
                color_name = get_color_name(obj[3])
                color = get_color_rgba_255(color_name)[:3]

            draw_rectangle(
                image, bounds, (pos[0], -pos[1]), (pos[0] + 1, -pos[1] - 1), color, 4
            )
        return image


class PBRSGoalWrapper(GridGoalWrapper):
    """Give a sparse potential-based reward shaping reward."""

    def extra_reward(
        self,
        obs: Observation,
        prev_abstract: AbstractState,
        prev_subgoal: AbstractState,
        prev_cost: float,
        term: bool,
    ) -> float:
        prev_potential = -prev_cost
        new_potential = -self._planner.cost
        return self.gamma * new_potential - prev_potential


class PBRSDenseGoalWrapper(GridGoalWrapper):
    """Calculate a dense potential-based reward shaping function.

    The planner cost is used as normally, but an additional dense potential term of the continuous state from the abstract subgoal.
    """

    def __init__(
        self,
        *args,
        dist_weight=0.5,
        vel_weight=0.5,
        activate_held_reward=True,
        activate_pickup_reward=True,
        gamma_override=None,
        **kwargs,
    ):
        """Construct a PBRSDenseGoalWrapper.

        Inputs:
        dist_weight (float) -- The weighting factor for agent position distance from subgoal, between 0-1. Higher gives more importance to distance.
        vel_weight (float) -- The weighting factor for agent velocity in the direction of the subgoal. Higher gives more importance to moving fast in that direction.
        activate_held_reward (bool) -- If true, then when agent holds an object (like a key), rewards will be based on object centroid instead of agent centroid.
        activate_pickup_reward (bool) -- If true, then will refine the subgoal target to coordinates of a specific object to be picked up.
        gamma_override (float) -- If specified, will replace the gamma used for PBRS. Note that this violates theoretical guarantees.
        """
        super().__init__(*args, **kwargs)
        self.dist_weight = dist_weight
        self.vel_weight = vel_weight
        self.activate_held_reward = activate_held_reward
        self.activate_pickup_reward = activate_pickup_reward
        if gamma_override is not None:
            self.gamma = gamma_override

    def reset(self, *args, **kwargs):
        output = super().reset(*args, **kwargs)
        curr_state = self.env.unwrapped.state
        dist = self.dense_potential(curr_state, self.abstract_state, self.subgoal)
        self.prev_total_dist = self._planner.cost + dist
        return output

    def extra_reward(
        self,
        obs: Observation,
        prev_abstract: AbstractState,
        prev_subgoal: AbstractState,
        prev_cost: float,
        term: bool,
    ) -> float:
        """Calculate the potential as planner cost plus dense potential to next subgoal."""
        curr_state = self.env.unwrapped.state
        dist_to_subgoal = self.dense_potential(
            curr_state, self.abstract_state, self.subgoal
        )
        new_total_dist = self._planner.cost + dist_to_subgoal

        old_potential = -self.prev_total_dist
        new_potential = -new_total_dist

        diff_in_potential = self.gamma * new_potential - old_potential

        self.prev_total_dist = new_total_dist

        return diff_in_potential

    def dense_potential(
        self,
        cocogrid_state: CocogridState,
        curr_abstract: GridAbstraction,
        subgoal_abstract: GridAbstraction,
    ) -> float:
        """Compute a continuous potential metric from the target abstract state based on distance and velocity"""
        # This function determines a source (current) and a target (subgoal).
        # Usually the source is the agent, but it could also be a held object.
        # Usually the target is a grid position region, but it could be a target object.

        # get agent position (normalized to grid scale) and velocity
        # pick_up_idx = self.get_index_of_object_to_pick_up(
        #     curr_abstract, subgoal_abstract
        # )
        if self.activate_held_reward and curr_abstract.held_object_idx >= 0:
            # Source is held object
            source_pos = cocogrid_state.get_object_pos(
                curr_abstract.held_object_idx, dim=2
            )
            source_vel = cocogrid_state.get_object_vel(
                curr_abstract.held_object_idx, dim=2
            )
        else:
            # Source is agent
            source_pos = cocogrid_state.get_agent_position(dim=2)
            source_vel = cocogrid_state.get_agent_velocity(dim=2)

        # Source needs to move to the closest point on boundary of the next subogal.
        # The subgoal boundary is defined by the four corners of the subgoal [tx, -(ty+1)], [tx+1, -ty].
        tx, ty = subgoal_abstract.walker_pos
        target_pos = np.clip(source_pos, [tx, -(ty + 1)], [tx + 1, -ty])
        # Check if an object needs to be picked up. If so, set the object's position as the target position.
        if (
            self.activate_pickup_reward
            and curr_abstract.held_object_idx != subgoal_abstract.held_object_idx
            and subgoal_abstract.held_object_idx >= 0
        ):
            target_pos = cocogrid_state.get_object_pos(
                subgoal_abstract.held_object_idx, dim=2
            )

        # Compute the distance and velocity.
        dist = np.linalg.norm(target_pos - source_pos)
        if dist == 0:
            return 0
        direction = (target_pos - source_pos) / dist
        velocity_in_target_direction = np.dot(source_vel, direction)

        # Scale distance from maximum 1.5 to maximum 1.
        distance_cost = min(1, 0.75 * dist)
        # Scale velocity from range (-3, 3) to (0,1)
        velocity_cost = max(0, min(1, (3 - velocity_in_target_direction) / 6))

        return self.dist_weight * distance_cost + self.vel_weight * velocity_cost


class DistanceCostGoalWrapper(GridGoalWrapper):
    """Give 'negative distance from goal' reward each step"""

    def extra_reward(
        self,
        obs: Observation,
        prev_abstract: AbstractState,
        prev_subgoal: AbstractState,
        prev_cost: float,
        term: bool,
    ) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(
            self.subgoal.walker_pos, curr_state
        )
        total_dist = self._planner.cost + dist
        return -total_dist


class DistanceCostFractionGoalWrapper(GridGoalWrapper):
    """Give 'negative distance from goal' reward each step, normalized by the initial plan cost. In other words, based on the fraction of progress to go."""

    def extra_reward(
        self,
        obs: Observation,
        prev_abstract: AbstractState,
        prev_subgoal: AbstractState,
        prev_cost: float,
        term: bool,
    ) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(
            self.subgoal.walker_pos, curr_state
        )
        total_dist = self._planner.cost + dist
        return -(total_dist / self._initial_plan_cost)


class SubgoalDistanceGoalWrapper(GridGoalWrapper):
    """Give dense 'negative distance from subgoal' reward each step. Useful if training a subgoal-reaching policy."""

    def extra_reward(
        self,
        obs: Observation,
        prev_abstract: AbstractState,
        prev_subgoal: AbstractState,
        prev_cost: float,
        term: bool,
    ) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(
            self.subgoal.walker_pos, curr_state
        )
        return -dist


class SubgoalGoalWrapper(GridGoalWrapper):
    """Give sparse 'negative distance from subgoal' reward each step. Useful if training a subgoal-reaching policy.

    Returns 1 reward if subgoal reached, 0 if same abstract state, and -1 if wrong subgoal reached.
    """

    def extra_reward(
        self,
        obs: Observation,
        prev_abstract: AbstractState,
        prev_subgoal: AbstractState,
        prev_cost: float,
        term: bool,
    ) -> float:
        if prev_abstract == self.abstract_state:
            return 0
        return 1 if self.abstract_state == prev_subgoal else -1
