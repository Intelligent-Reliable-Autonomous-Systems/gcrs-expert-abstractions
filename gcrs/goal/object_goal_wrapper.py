import cv2
from cocogrid.common.abstraction import ObjectAbstraction
from cocogrid.common.entity import get_color_name
from cocogrid.utils.visualize.drawing import get_camera_bounds
from cocogrid.common.color import get_color_rgba_255

from gcrs.goal.goal_wrapper import GoalWrapper
from gcrs.goal.planner import AbstractState, Observation


class ObjectGoalWrapper(GoalWrapper):
    def __init__(self, *args, gamma=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def render(self):
        image = super().render()
        if image is not None:
            
            (held_type, held_color, near_type, near_color, near_state) = self._observer.goal_observation_fn(
                self.subgoal
            )

            if held_type == -1:
                held_color = (255, 128, 40)
                held_text = "Not holding."
            else:
                held_name = ObjectAbstraction.OBJECT_IDS[held_type]
                held_color_name = get_color_name(held_color)
                held_color = tuple(get_color_rgba_255(held_color_name)[:3])
                held_text = f"Hold {held_color_name} {held_name}."

            if near_type == -1:
                near_color = (255, 128, 40)
                near_text = "Not near."
            else:
                near_name = ObjectAbstraction.OBJECT_IDS[near_type]
                near_color_name = get_color_name(near_color)
                near_color = tuple(get_color_rgba_255(near_color_name)[:3])
                door_state = ""
                if near_type == ObjectAbstraction.DOOR_IDX:
                    door_state = ["open ", "closed ", "locked ", "locked "][near_state]
                near_text = f"Go near {door_state}{near_color_name} {near_name}."

            cv2.putText(
                image,
                held_text,
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                held_color,
                1,
                lineType=2,
            )
            cv2.putText(
                image,
                near_text,
                (image.shape[0] // 2, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                near_color,
                1,
                lineType=2,
            )
        return image


class PBRSObjectGoalWrapper(ObjectGoalWrapper):
    def extra_reward(
        self,
        obs: Observation,
        prev_abstract: AbstractState,
        prev_subgoal: AbstractState,
        prev_cost: float,
        term: bool,
    ) -> float:
        return prev_cost - self._planner.cost


class SubgoalObjectGoalWrapper(ObjectGoalWrapper):
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


class SubgoalCostObjectGoalWrapper(ObjectGoalWrapper):
    def extra_reward(
        self,
        obs: Observation,
        prev_abstract: AbstractState,
        prev_subgoal: AbstractState,
        prev_cost: float,
        term: bool,
    ) -> float:
        if prev_abstract == self.abstract_state:
            return -1
        return 0 if self.abstract_state == prev_subgoal else -400


class SubgoalLargePenaltyObjectGoalWrapper(ObjectGoalWrapper):
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
        return 100 if self.abstract_state == prev_subgoal else -400


class SubgoalRewardOnlyObjectGoalWrapper(ObjectGoalWrapper):
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
        return int(self.abstract_state == prev_subgoal)
