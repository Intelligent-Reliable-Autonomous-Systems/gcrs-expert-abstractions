# from minimujo.state.goal_wrapper import GoalWrapper, GoalObserver, Observation, AbstractState
from dataclasses import dataclass, astuple
from minimujo.dmc_gym import DMCGym
from minimujo.state.goal_wrapper import AStarPlanner, GoalWrapper, GoalObserver, DeterministicValueIterationPlanner
import numpy as np
import gymnasium as gym
from dm_control.manipulation.bricks import _min_stud_to_hole_distance

@dataclass(frozen=True, eq=True)
class BrickStackAbstraction:

    manipulator_x: int
    manipulator_y: int
    manipulator_z: int
    manipulator_grasp: bool
    held_brick: int
    bricks: tuple
    stacks: tuple
    bounds: tuple = (-8, -8, 8, 8)

    def __repr__(self) -> str:
        # obj_str = ','.join(map(GridAbstraction.pretty_object, self.objects))
        m_pos = (self.manipulator_x, self.manipulator_y, self.manipulator_z)
        return f'Stack[{m_pos}; {self.manipulator_grasp}, {self.held_brick}; {self.stacks}; {self.bricks}]'

    def do_action(self, action):
        mx, my, mz = self.manipulator_x, self.manipulator_y, self.manipulator_z
        grasp, held = self.manipulator_grasp, self.held_brick
        new_bricks = list(self.bricks)
        new_stacks = [list(stack) for stack in self.stacks]

        # do gravity on unheld bricks
        for idx, brick in enumerate(new_bricks):
            bx, by, bz = brick[:3]
            if bz > 0 and held != idx and brick not in new_stacks:
                bz = bz - 1
                new_bricks[idx] = (bx, by, bz)

        # move manipulator
        new_pos = (mx, my, mz)
        if mz > 0:
            # require that the manipulator can move horizontally only in the air
            if action == 0:
                # forward
                new_pos = (mx, my+1, mz)
            elif action == 1:
                # left
                new_pos = (mx-1, my, mz)
            elif action == 2:
                # backward
                new_pos = (mx, my-1, mz)
            elif action == 3:
                # right
                new_pos = (mx+1, my, mz)
        if action == 4:
            # up
            new_pos = (mx, my, mz + 1)
        elif action == 5:
            # down
            new_pos = (mx, my, mz - 1)
        new_pos = (
            max(self.bounds[0], min(self.bounds[2], new_pos[0])), 
            max(self.bounds[1], min(self.bounds[3], new_pos[1])), 
            max(0, min(1, new_pos[2]))
        )

        def get_stack(brick_idx):
            for stack_idx, stack in enumerate(new_stacks):
                if brick_idx in stack:
                    return stack_idx, stack[stack.index(brick_idx):]
            raise Exception("All bricks must be contained in a stack, even if singleton")
        
        def get_stack_pos(stack):
            return new_bricks[stack[0]][:3]

        # move held stack along with manipulator
        if grasp and held >= 0:
            sid, stack = get_stack(held)
            # move all bricks at or above the held brick
            for brick_idx in stack:
                new_bricks[brick_idx] = new_pos

        if action == 6:
            # try stacking
            if grasp and held >= 0:
                # stack whatever is held onto a stack at a position
                held_stack_id, held_stack = get_stack(held)
                # new_stacks[sid]
                # if new_bricks[held]:
                #     new_stacks.append(held)
                for stack_id, stack in enumerate(new_stacks):
                    if stack_id == held_stack_id:
                        continue
                    if get_stack_pos(stack) == new_pos:
                        new_stacks[stack_id] = [*stack, *held_stack]
                        new_stacks.pop(held_stack_id)
                        break

            # try grabbing or releasing
            grasp = not grasp
            if grasp:
                # when we grasp, try to pick from the smallest stack
                grabbed_stack_idx = -1
                stack_size = 1000
                held = -1
                for stack_idx, stack in enumerate(new_stacks):
                    if get_stack_pos(stack) == new_pos and len(stack) < stack_size:
                        grabbed_stack_idx = stack_idx
                        stack_size = len(stack)
                # grab the top brick off the stack
                if grabbed_stack_idx != -1:
                    grabbed_stack = new_stacks[grabbed_stack_idx]
                    if len(grabbed_stack) > 1:
                        # if brick was part of a larger stack, put it into a new stack
                        top_brick = grabbed_stack.pop()
                        new_stacks.append([top_brick])
                        grabbed_stack = new_stacks[-1]
                    held = grabbed_stack[-1]

                # grab_from_stack = False
                # for idx, brick in enumerate(new_bricks):
                #     if brick[:3] == new_pos and idx not in new_stacks[:-1]:
                #         # if manipulator is at brick and brick not buried in stack
                #         held = idx
                #         if idx in new_stacks:
                #             grab_from_stack = True
                #         else:
                #             grab_from_stack = False
                #             break
                # if grab_from_stack:
                #     held = new_stacks.pop()
            else:
                held = -1

        for bid in range(len(new_bricks)):
            # make sure all are included in stack
            get_stack(bid)
        if held >= 0:
            sidx, stack = get_stack(held)
            assert new_stacks[sidx][0] == held

        new_stacks = tuple(tuple(stack) for stack in sorted(new_stacks))
        return BrickStackAbstraction(*new_pos, grasp, held, tuple(new_bricks), new_stacks)
    
    @staticmethod
    def quantize_position(pos, scale=0.05, height=0.2):
        return round(pos[0]/scale), round(pos[1]/scale), int(pos[2] >= height)
    
def manipulator_state_to_abstract(state: np.ndarray, env: gym.Env):
    base_env: DMCGym = env.unwrapped
    physics = base_env._env._physics_proxy
    task = base_env._env._task

    # def discrete_pos(pos):
    #     return int(pos[0]), int(pos[1]), int(pos[2] >= 2)
    brick_names = ["duplo2x4", *[f"duplo2x4_{i}" for i in range(2,6)]]
    bricks = []
    for name in brick_names:
        brick_pos = base_env.index_obs_as_dict(state, f'{name}/position')
        if brick_pos is None:
            continue
        bricks.append(BrickStackAbstraction.quantize_position(brick_pos))
    manip_pos = BrickStackAbstraction.quantize_position(base_env.index_obs_as_dict(state, 'jaco_arm/jaco_hand/pinch_site_pos'))
    grasp = False
    held = -1

    brick_ids = list(range(len(bricks)))
    connections = {bid:None for bid in brick_ids}
    reverse_connections = {bid:None for bid in brick_ids}
    for bid1 in brick_ids:
        for bid2 in brick_ids:
            if bid1 == bid2:
                continue
            dist = _min_stud_to_hole_distance(physics, task._bricks[bid1], task._bricks[bid2])
            # print('dist', bid1, bid2, dist)
            if dist < 0.001: # whatever threshold
                connections[bid1] = bid2
                reverse_connections[bid2] = bid1

    stacks = []
    for bottom_key, _ in filter(lambda kv: kv[1] is None, reverse_connections.items()):
        # start with the bottom of the stack (for which nothing points to it)
        stack = [bottom_key]
        nxt = connections.get(bottom_key, None)
        while nxt is not None:
            stack.append(nxt)
            nxt = connections.get(nxt, None)
            connections[nxt] = None
        stacks.append(tuple(stack))
        
    return BrickStackAbstraction(*manip_pos, grasp, held, tuple(bricks), tuple(sorted(stacks)))

def brick_stack_task_getter(state: dict, abstract: BrickStackAbstraction, env: gym.Env):
    # for each new episode, get the desired order of bricks and return a function to check that order
    desired_order = tuple(env.unwrapped._env._task._desired_order)
    print('DESIRED', desired_order)
    def evaluate_abstract(abstract: BrickStackAbstraction):
        if desired_order in abstract.stacks:
            return 1, True
        return -1, False
    return evaluate_abstract

def brick_reach_task_getter(state: dict, abstract: BrickStackAbstraction, env: gym.Env):
    base_env: DMCGym = env.unwrapped
    physics = base_env._env._physics_proxy
    task = base_env._env._task
    target_x, target_y, target_z = BrickStackAbstraction.quantize_position(physics.bind(task._target).xpos)

    def evaluate_abstract(abstract: BrickStackAbstraction):
        success = abstract.manipulator_x == target_x \
            and abstract.manipulator_y == target_y \
            and abstract.manipulator_z == target_z
        return 1 if success else -1, success
    return evaluate_abstract

def manipulator_goal_observation(abstract: BrickStackAbstraction):
    # show manipulator x, y, z, grasp, and held
    # the planner will take the objects and stack into consideration, 
    #   so the controller doesn't need to observe those
    return (
        abstract.manipulator_x, abstract.manipulator_y, abstract.manipulator_z,
        int(abstract.manipulator_grasp), abstract.held_brick
    )

def get_manipulator_goal_wrapper(env, env_id, goal_cls=GoalWrapper):
    actions = list(range(7))
    transition_fn = lambda abstract, action: abstract.do_action(action)
    planner = AStarPlanner(actions, transition_fn)

    # define the augmentation of the observation space
    low = [-4, -4, 0, 0, -1]
    high = [4, 4, 1, 1, 5]
    observer = GoalObserver(manipulator_goal_observation, low, high)

    if 'stack' in env_id:
        task = brick_stack_task_getter
    elif 'reach' in env_id:
        task = brick_reach_task_getter
    else:
        raise Exception(f"Task not specified for {env_id}")

    return goal_cls(env, manipulator_state_to_abstract, task, planner, observer, use_base_reward=False)

if __name__ == "__main__":
    # 0, 0, 0, False, -1, ((0, 0, 0), (0, 0, 0)), ((0, 1, 1),)
    state = BrickStackAbstraction(0, 0, 0, False, -1, ((1,2,0), (2,-2,0)), ((0,), (1,)))
    # state = ManipulatorAbstraction(1, 1, 0, False, -1, ((1,1,0), (1,1,0)), ())

    action_map = {
        'w': 0,
        'a': 1,
        's': 2,
        'd': 3,
        'u': 4,
        'j': 5,
        'g': 6
    }
    while True:
        print("State", astuple(state))
        action = action_map[input("Action: ")]
        state = state.do_action(action)