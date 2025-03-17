#!/bin/bash

# Train a Box2D agent on cocogrid UMaze with GCRS on the grid abstraction
# Episode length 400 and train for 3 million steps
# Arguments:
#   --abstraction <type>: which expert abstraction and reward shaping to use. (default pbrs-grid)
#       grid-pbrs for PBRS with the grid abstraction
#       room-pbrs for PBRS with the room abstraction
#       grid-pbrs-no-subgoal for PBRS with the grid abstraction, but subgoals are not visible to agent
#       room-pbrs-no-subgoal for PBRS with the room abstraction, but subgoals are not visible to agent
#   --grid <N>: the number of grid cells in the CocoGrid environment, DoorKey-NxN. Options are 5,6,8,10,12 (default 6)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default value
GRID=6
ABSTRACTION="grid-pbrs"

# Parse optional --value argument
for arg in "$@"; do
    case $arg in
        --grid=*) GRID="${arg#*=}" ;;
    esac
    case $arg in
        --abstraction=*) ABSTRACTION="${arg#*=}" ;;
    esac
done

python "${SCRIPT_DIR}/../gcrs/rpo_continuous_action.py" \
    --env-id cocogrid/DoorKey-${GRID}x${GRID}-v0 \
    --goal-version ${ABSTRACTION} \
    --cocogrid.xy-scale 3 \
    --cocogrid.agent box2d \
    --cocogrid.observation no-arena \
    --num-envs 12 \
    --num-steps 400 \
    --total-timesteps 3000000 \
    --log-dir "${SCRIPT_DIR}/../runs/DoorKey-${GRID}x${GRID}-${ABSTRACTION}" \
    --capture-video

