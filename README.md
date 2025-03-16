# Goal-Conditioned Reward Shaping

This is an implementation of Goal-Conditioned Reward Shaping (GCRS). In order to learn a continuous-control agent with reinforcement learning that can adapt to new environments, GCRS allows you to specify an abstract representation of the environment to increase performance. While the low-level neural policy deals with the complexity of continuous state spaces, the abstraction enables higher order planning using classical methods like A\*. 

The experiments are run in [CocoGrid](https://github.com/Intelligent-Reliable-Autonomous-Systems/CocoGrid), a continuous-control extension of [MiniGrid](https://github.com/Farama-Foundation/Minigrid).

## Installation

```bash
$ git clone https://github.com/Intelligent-Reliable-Autonomous-Systems/GCRS
$ cd GCRS
$ pip install -e .
```

## Training agent

An agent can be trained with `gcrs/rpo_continuous_action.py`. 

## Manual control

You can experiment with the environments with `gcrs/test_env.py`