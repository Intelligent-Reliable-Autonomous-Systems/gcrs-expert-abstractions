from functools import partial

# from minimujo.state.grid_abstraction import get_minimujo_goal_wrapper
from gcrs.goal.grid_planning import construct_grid_goal_wrapper
from gcrs.goal.object_abstraction import (
    get_object_abstraction_goal_wrapper
)
from gcrs.goal.object_goal_wrapper import PBRSObjectGoalWrapper, SubgoalObjectGoalWrapper, SubgoalCostObjectGoalWrapper, SubgoalRewardOnlyObjectGoalWrapper, ObjectGoalWrapper
from gcrs.goal.grid_goal_wrapper import (
    DistanceCostGoalWrapper,
    GoalWrapper,
    PBRSGoalWrapper,
    PBRSDenseGoalWrapper,
    SubgoalDistanceGoalWrapper,
    SubgoalGoalWrapper,
)
from gcrs.env import is_cocogrid_env

# Wrappers using cococgrid grid abstraction
grid_goal_wrappers = {
    "grid-goal-only": GoalWrapper,  # Use the subgoal planning without reward shaping
    "grid-pbrs": PBRSGoalWrapper,  # Use PBRS
    "grid-pbrs-no-subgoal": PBRSGoalWrapper,  # Use PBRS, but don't give subgoals to low level controller
    "grid-pbrs-dense-pos-only": partial(  # Use PBRS with dense rewards based on agent position
        PBRSDenseGoalWrapper,
        dist_weight=1,
        vel_weight=0,
    ),
    "grid-pbrs-dense-pos-vel": partial(  # Use PBRS with dense rewards based on agent position and velocity
        PBRSDenseGoalWrapper,
        dist_weight=0.5,
        vel_weight=0.5,
    ),
    "grid-distance-cost": DistanceCostGoalWrapper,  # Give 'negative distance from goal' reward each step
    "grid-subgoal": SubgoalGoalWrapper,  # Give sparse 'negative distance from subgoal' reward each step. Useful if training a subgoal-reaching policy.
    "grid-subgoal-distance": SubgoalDistanceGoalWrapper,  # Give dense 'negative distance from subgoal' reward each step. Useful if training a subgoal-reaching policy.
}

# Wrappers using CocoGrid room/object abstraction.
object_abstraction_wrappers = {
    "object-no-rew": ObjectGoalWrapper,
    "object-pbrs": PBRSObjectGoalWrapper,
    "object-pbrs-no-subgoal": PBRSObjectGoalWrapper,
    "object-pos-pbrs": PBRSObjectGoalWrapper,
    "object-pbrs-one-hot": PBRSObjectGoalWrapper,
    "object-subgoal-achieve": SubgoalObjectGoalWrapper,
    "object-subgoal-cost": SubgoalCostObjectGoalWrapper,
    "object-subgoal-rew-only": SubgoalRewardOnlyObjectGoalWrapper,
}


def wrap_env_with_goal(env, env_id, goal_version, gamma=1) -> GoalWrapper:
    """Wrap a gymnasium environment with a goal wrapper, to augment observations and rewards with high-level planning."""
    if goal_version == "no-goal" or goal_version == "":
        return env

    if is_cocogrid_env(env_id):
        if goal_version in object_abstraction_wrappers:
            goal_cls = object_abstraction_wrappers[goal_version]
            goal_cls = partial(goal_cls, gamma=gamma)
            return get_object_abstraction_goal_wrapper(
                env,
                env_id,
                goal_cls,
                one_hot=("one-hot" in goal_version),
                use_pos=("object-pos" in goal_version),
                observe_subgoal=("no-subgoal" not in goal_version),
            )

        if goal_version in grid_goal_wrappers:
            goal_cls = grid_goal_wrappers[goal_version]
            goal_cls = partial(goal_cls, gamma=gamma)
            return construct_grid_goal_wrapper(
                env,
                env_id,
                goal_cls,
                observe_subgoal=("no-subgoal" not in goal_version),
            )
    raise Exception(f"goal version {goal_version} not valid in env {env_id}")


__all__ = [wrap_env_with_goal]
