# AISL/source/classical_motion_planner/nominal_test_rrt.py

# Top-Level Directory : source/classical_motion_planner
from utils import Env
from utils import Node
from RRT.RRT_star import RRTStar


obs_rect = [
    [18, 22, 5, 8, 3, 4],
    [24, 20, 5, 8, 8, 4],
    [26,  7, 5, 2, 12, 3],
    [32, 14, 5, 10, 2, 8]
]


environment = Env.Map3D(51, 31, 31)
environment.update(obs_rect=obs_rect)


planner = RRTStar(start = (18,8,2), goal=(35,22,10), env=environment)

try:
    planner.run()
finally:
    print("End of RRT*")