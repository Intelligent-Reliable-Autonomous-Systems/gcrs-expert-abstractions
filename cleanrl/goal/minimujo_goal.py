from minimujo.state.goal_wrapper import GoalWrapper, AbstractState, Observation
from minimujo.state.grid_abstraction import GridAbstraction

class PBRSGoalWrapper(GoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        return prev_cost - self._planner.cost
    
class PBRSDenseGoalWrapper(GoalWrapper):

    def reset(self, *args, **kwargs):
        output = super().reset(*args, **kwargs)
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        self.prev_total_dist = self._planner.cost + dist
        return output

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        total_dist = self._planner.cost + dist
        diff = self.prev_total_dist - total_dist
        self.prev_total_dist = total_dist
        return diff
    
class DistanceCostGoalWrapper(GoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        total_dist = self._planner.cost + dist
        return -total_dist
    
class DistanceCostFractionGoalWrapper(GoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        total_dist = self._planner.cost + dist
        return -(total_dist / self._initial_plan_cost)
    
class SubgoalDistanceGoalWrapper(GoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        curr_state = self.env.unwrapped.state
        dist = GridAbstraction.grid_distance_from_state(self.subgoal.walker_pos, curr_state)
        return -dist

class SubgoalGoalWrapper(GoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        if prev_abstract == self.abstract_state:
            return 0
        return 1 if self.abstract_state == prev_subgoal else -1
    
class SubgoalLargePenaltyGoalWrapper(GoalWrapper):

    def extra_reward(self, obs: Observation, prev_abstract: AbstractState, prev_subgoal: AbstractState, prev_cost: float) -> float:
        if prev_abstract == self.abstract_state:
            return 0
        return 100 if self.abstract_state == prev_subgoal else -400