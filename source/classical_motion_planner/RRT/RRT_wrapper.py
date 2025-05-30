import torch
from typing import List, Tuple
from .RRT_star import RRTStar
from utils import Env


class RRTWrapper:
    """
    Wrapper for RRT* motion planning.

    """
    def __init__(self, start: torch.Tensor, goal: torch.Tensor, env: Env.Map3D, max_dist: float = 0.05):
        self.env = env
        self.start = start
        self.goal = goal
        self.start_tuple = tuple(self.start[:3].tolist())
        self.goal_tuple = tuple(self.goal[:3].tolist())             
        self.planner = RRTStar(start=self.start_tuple, goal=self.goal_tuple, env=env, max_dist=max_dist)
    
    def plan(self) -> torch.Tensor:
        """
        Plan a path from start to goal using RRT*.

        Parameters:
            start (torch.Tensor): Start point coordinates.
            goal (torch.Tensor): Goal point coordinates.

        Returns:
            torch.Tensor: Planned path as a tensor of shape (N, 3).
        """
        _, path, _ = self.planner.plan()

        return torch.flip(torch.as_tensor(path, dtype=self.start.dtype, device=self.start.device), dims=(0,))

    def run(self):
        self.planner.run()
