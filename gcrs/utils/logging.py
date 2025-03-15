from typing import Any, Dict
from cocogrid.utils.logging import LoggingMetric


class SubgoalLogger(LoggingMetric):
    """Log subgoals produced by goal_wrapper"""

    def __init__(self, label_prefix: str = 'subgoals') -> None:
        self.label_prefix = label_prefix
        self.num_left_label = f'{label_prefix}/num_goals_left'
        self.frac_left_label = f'{label_prefix}/frac_goals_left'
        self.cum_task_reward_label = f'{label_prefix}/cum_task_reward'
        self.goals_achieved_label = f'{label_prefix}/subgoals_achieved'
        self.goals_failed_label = f'{label_prefix}/subgoals_failed'
        self.cum_task_reward = 0
        self.goals_achieved = 0
        self.goals_failed = 0

    def on_episode_start(self, obs: Any, info: Dict[str, Any], episode: int) -> None:
        self.cum_task_reward = 0
        self.goals_achieved = 0
        self.goals_failed = 0

    def on_episode_end(self, timesteps: int, episode: int) -> None:
        if self.summary_writer is not None:
            global_step = self.global_step_callback()
            if self.num_left is not None:
                self.summary_writer.add_scalar(self.num_left_label, self.num_left, global_step)
            if self.frac_left is not None:
                self.summary_writer.add_scalar(self.frac_left_label, self.frac_left, global_step)
            self.summary_writer.add_scalar(self.cum_task_reward_label, self.cum_task_reward, global_step)
            self.summary_writer.add_scalar(self.goals_achieved_label, self.goals_achieved, global_step)
            self.summary_writer.add_scalar(self.goals_failed_label, self.goals_failed, global_step)

    def on_step(self, obs: Any, rew: float, term: bool, trunc: bool, info: Dict[str, Any], timestep: int) -> None:
        if term or trunc:
            self.num_left = info.get('num_subgoals', None)
            self.frac_left = info.get('frac_subgoals', None)
        self.cum_task_reward += info.get('task_reward', 0)
        if info.get('is_new_goal', False):
            achieved = info.get('goal_achieved', False)
            if achieved:
                self.goals_achieved += 1
            else:
                self.goals_failed += 1

