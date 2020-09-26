# rocksample

This is basically a copy-and-paste from the [RockSample
domain](https://github.com/h2r/pomdp-py/blob/master/pomdp_problems/rocksample/rocksample_problem.py) implemented in [pomdp_py](https://github.com/h2r/pomdp-py), except that I added
a heuristic planner in `handcraft.py` that basically moves the robot to the
closest unchecked rock, check it. If good, sample. Do so for every rock that is
not on the west of the robot. This handcrafted policy works decently well.

## Example program output

Running Rocksample(5,5). Only showing the output for the first and last steps
for each planner. Note that you can run on a different size by simply changing
the `n, k = 5, 5` to another setting.

```
*** Testing POMCP (random rollout) ***
==== Step 1 ====
True state: State((0, 2) | ('good', 'bad', 'bad', 'good', 'bad') | False)
Action: move-(-1, 0)
Observation: None
Reward: 0.0
Reward (Cumulative): 0.0
Reward (Cumulative Discounted): 0.0
__num_sims__: 4096
__plan_time__: 1.00771
World:

______ID______
.2.0.>
..4..>
R.1..>
.....>
.3...>
_____G/B_____
.x.$.>
..x..>
R.x..>
.....>
.$...>
...
==== Step 10 ====
True state: State((4, 4) | ('good', 'bad', 'bad', 'bad', 'bad') | False)
Action: move-(1, 0)
Observation: None
Reward: 10.0
Reward (Cumulative): 20.0
Reward (Cumulative Discounted): 14.040303472246089  
__num_sims__: 4096
__plan_time__: 0.99582
World:

______ID______
.2.0.>
..4..>
..1..>
.....>
.3...R
_____G/B_____
.x.$.>
..x..>
..x..>
.....>
.x...R



*** Testing Handcraft ***
==== Step 1 ====
Particle reinvigoration for 0 particles
True state: State((0, 2) | ('good', 'bad', 'bad', 'good', 'bad') | False)
Action: move-(1, 0)
Observation: None
Reward: 0.0
Reward (Cumulative): 0.0
Reward (Cumulative Discounted): 0.0
World:

______ID______
.2.0.>
..4..>
.R1..>
.....>
.3...>
_____G/B_____
.x.$.>
..x..>
.Rx..>
.....>
.$...>

...
==== Step 11 ====
Particle reinvigoration for 0 particles
True state: State((4, 0) | ('bad', 'bad', 'bad', 'good', 'bad') | False)
Action: move-(1, 0)
Observation: None
Reward: 10.0
Reward (Cumulative): 20.0
Reward (Cumulative Discounted): 12.621573705274407  
World:

______ID______
.2.0.R
..4..>
..1..>
.....>
.3...>
_____G/B_____
.x.x.R
..x..>
..x..>
.....>
.$...>

```
