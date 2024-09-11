import re
import numpy as np

from minimujo.state.grid_abstraction import GridAbstraction

def get_umaze_goal_getter(task, init_state):
    def get_umaze_goals(abstract_state):
        pos = abstract_state.walker_pos
        if pos == (1,1):
            return [abstract_state]
        pos_actions = {
            (2,1): GridAbstraction.ACTION_LEFT,
            (3,1): GridAbstraction.ACTION_LEFT,
            (3,2): GridAbstraction.ACTION_UP,
            (3,3): GridAbstraction.ACTION_UP,
            (2,3): GridAbstraction.ACTION_RIGHT,
            (1,3): GridAbstraction.ACTION_RIGHT
        }
        goal_sequence = []
        while pos != (1,1):
            if pos in pos_actions:
                abstract_state = abstract_state.do_action(pos_actions[pos])
                pos = abstract_state.walker_pos
                goal_sequence.append(abstract_state)
        return goal_sequence
    return get_umaze_goals

def get_randobj_goal_getter(task, init_state):
    pattern = r"Deliver a (\w+) (\w+) to tile \((\d+), (\d+)\)\."
    matches = re.search(pattern, task.description)

    if not matches:
        raise Exception("No match found.")
    # Extract the variables using group capturing
    color = matches.group(1)
    from minimujo.color import get_color_idx
    color_idx = get_color_idx(color)
    class_name = matches.group(2)
    class_idx = ['Ball','Box'].index(class_name)
    # x = int(matches.group(3))
    # y = int(matches.group(4))
    selected_obj_idx = -1
    for idx, obj in enumerate(init_state.objects):
        if obj[3] == color_idx and obj[0] == class_idx:
            selected_obj_idx = idx

    if selected_obj_idx == -1:
        raise Exception(f"No object found matching description, {color} {class_name}")
    
    def get_randobj_goals(abstract_state):
        def next_state(abstract_state):
            obj = abstract_state.objects[selected_obj_idx]
            _, obj_x, obj_y, _, held = obj
            walker_x, walker_y = abstract_state.walker_pos
            goal_idx = np.where(abstract_state.grid == 2)
            goal_x, goal_y = goal_idx[0][0], goal_idx[1][0]
            for idx, obj in enumerate(abstract_state.objects):
                if bool(obj[4]) and idx != selected_obj_idx:
                    # another object is held. drop it
                    new_objects = abstract_state.objects.copy()
                    new_objects[idx] = (*new_objects[idx][:4], 0)
                    return GridAbstraction(abstract_state.grid, abstract_state.walker_pos, new_objects)
            if held:
                if goal_x != walker_x:
                    # move horizontal towards goal w/ object
                    if goal_x < walker_x:
                        return abstract_state.do_action(GridAbstraction.ACTION_LEFT)
                    else:
                        return abstract_state.do_action(GridAbstraction.ACTION_RIGHT)
                if goal_y != walker_y:
                    # move vertical towards goal w/ object
                    if goal_y < walker_y:
                        return abstract_state.do_action(GridAbstraction.ACTION_UP)
                    else:
                        return abstract_state.do_action(GridAbstraction.ACTION_DOWN)
                # else is at goal. release object
                new_objects = abstract_state.objects.copy()
                new_objects[selected_obj_idx] = (*new_objects[selected_obj_idx][:4], 0)
                return GridAbstraction(abstract_state.grid, abstract_state.walker_pos, new_objects)
            # else move towards object
            if walker_x != obj_x:
                if obj_x < walker_x:
                    return abstract_state.do_action(GridAbstraction.ACTION_LEFT)
                else:
                    return abstract_state.do_action(GridAbstraction.ACTION_RIGHT)
            if walker_y != obj_y:
                if obj_y < walker_y:
                    return abstract_state.do_action(GridAbstraction.ACTION_UP)
                else:
                    return abstract_state.do_action(GridAbstraction.ACTION_DOWN)
            if obj_x == goal_x and obj_y == goal_y:
                return None
            # else grab object
            new_objects = abstract_state.objects.copy()
            new_objects[selected_obj_idx] = (*new_objects[selected_obj_idx][:4], 1)
            return GridAbstraction(abstract_state.grid, abstract_state.walker_pos, new_objects)
        
        state = abstract_state
        if next_state(state) is None:
            return [state]
        goal_seq = []
        while state is not None:
            state = next_state(state)
            if state is None:
                break
            goal_seq.append(state)
            if len(goal_seq) > 20:
                raise Exception(f"Search depth exceeded: {goal_seq}")
        return goal_seq

    return get_randobj_goals