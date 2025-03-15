from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import heapq
from typing import Callable, Hashable, Iterable, Sequence, Tuple, TypeVar, Union

import gymnasium as gym

Observation = TypeVar('Observation')
AbstractState = TypeVar('AbstractState', bound=Hashable)
Action = TypeVar('Action')
Task = Union[AbstractState, Callable[[AbstractState], Tuple[float, bool]]]

class SubgoalPlanner(ABC):

    def __init__(self):
        self._task_initialized: bool = False
        self._current_state: AbstractState = None
        self._goal_state: AbstractState = None

    def init_task(self, obs: Observation, abstract: AbstractState, env: gym.Env):
        self._task_initialized = True

    def update_state(self, state: AbstractState) -> None:
        """Updates the currently achieved subgoal"""
        self._current_state = state
    
    @property
    def goal(self) -> AbstractState:
        """Gets the target final goal"""
        return self.plan[-1]
    
    @property
    def state(self) -> AbstractState:
        """Gets the currently achieved subgoal"""
        return self._current_state
    
    @property
    def next_state(self) -> AbstractState:
        """Gets the next goal in the plan"""
        if len(self.plan) < 2:
            return self.state
        return self.plan[1]

    @property
    @abstractmethod
    def plan(self) -> Sequence[AbstractState]:
        """Gets the full plan sequence of subgoals"""
        pass

    @property
    @abstractmethod
    def cost(self) -> float:
        """Gets the cost of the computed plan"""
        pass

    @abstractmethod
    def update_plan(self) -> None:
        """Computes or updates the plan sequence of subgoals"""
        pass
    
class AStarPlanner(SubgoalPlanner):

    def __init__(
        self,
        task_getter: Callable[[Observation, AbstractState, gym.Env], Callable[[AbstractState], Iterable[Tuple[AbstractState, float, bool]]]],
        heuristic_function: Callable[[AbstractState], float] = None
    ):
        self.task_getter = task_getter
        self.heuristic_function = heuristic_function or (lambda abstract: 0)
        self._is_initialized = False
        self._plan_failure_count = 0

    def init_task(self, obs: Observation, abstract: AbstractState, env: gym.Env):
        self._task_initialized = True
        self._is_initialized = False
        self.get_edges = self.task_getter(obs, abstract, env)
        
    @property
    def plan(self) -> Sequence[AbstractState]:
        """Gets the full plan sequence of subgoals"""
        assert self._plan is not None, "Plan has not been computed. Call update_plan before referencing plan"
        return self._plan

    @property
    def cost(self) -> float:
        """Gets the cost of the computed plan"""
        assert self._plan is not None, "Plan has not been computed. Call update_plan before referencing plan cost"
        return self._costs[self._plan[-1]] - self._costs[self._plan[0]]
    
    def update_plan(self):
        # Use Djisktra
        if self._is_initialized and self.state in self._plan:
            state_idx = self._plan.index(self.state)
            self._plan = self._plan[state_idx:]
            return
        self._init_djikstra()

        while len(self._priority_queue) > 0:
            # Get the state with the lowest known distance
            prioritized_item = heapq.heappop(self._priority_queue)
            cur_state, g_current = prioritized_item.item

            # If we've already visited this state, skip it
            if cur_state in self._visited:
                continue

            # Mark the state as visited
            self._visited.add(cur_state)

            # Explore neighbors of the current state
            for neighbor, cost, goal_reached in self.get_edges(cur_state):
                if neighbor in self._visited:
                    continue

                # g score is the real cost from start to neighbor
                # task costs are negative, so negate it to make costs positive (i.e. lower score is better)
                g_neighbor = g_current + cost
                # f score is heuristic cost
                f_neighbor = g_neighbor - self.heuristic_function(cur_state)
                # If a shorter path to the neighbor is found
                if neighbor not in self._costs or g_neighbor < self._costs[neighbor]:
                    self._costs[neighbor] = g_neighbor
                    self._predecessors[neighbor] = cur_state
                    if goal_reached:
                        self._plan = self._reconstruct_path(neighbor)
                        return
                    heapq.heappush(self._priority_queue, PrioritizedItem(f_neighbor, (neighbor, g_neighbor)))
                    
        self._plan_failure_count += 1
        print('failed to plan', self.state)
        self._plan = [self.state]
        # if self._plan_failure_count > 20:
        #     raise Exception("AStarPlanner failed to plan too many times")
        
    def _init_djikstra(self) -> None:
        assert self.state is not None, "Current subgoal has not been set"
        self._visited = set()
        self._costs = {self.state: 0}
        self._priority_queue = [PrioritizedItem(-self.heuristic_function(self.state), (self.state, 0))]
        self._predecessors = {}
        self._is_initialized = True
        self._plan = None

    def _reconstruct_path(self, state: AbstractState) -> None:
        path = []
        while state in self._predecessors:
            path.append(state)
            state = self._predecessors[state]
        path.append(self.state)
        return path[::-1]

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: object = field(compare=False)