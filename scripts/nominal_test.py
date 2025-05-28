# AISL/scripts/nominal_test.py

# Top-Level Directory : scripts
from utils import Node, Env
from TestRRT.RRT_star import RRTStar

environment = Env.Map3D(51, 31, 31)

planner = RRTStar(start = (18,8,2), goal=(35,22,10), env=environment)

try:
    cost, path, expand = planner.run()
except:
    pass
finally:
    print("End of RRT*")