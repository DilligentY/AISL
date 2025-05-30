# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .uniform_action import UniformAction
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from dataclasses import MISSING

@configclass
class ObjectSurfaceActionTermCfg(ActionTermCfg):
    """ 
    Configuration for the action term that applies actions to the object surface.
    """
    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type[ActionTerm] = UniformAction
    """Type of the action term class."""

    body_name: str = MISSING
    """Name of the body which has the dummy mechanism connected to"""

    num_samples: int = 50
    """Number of samples to be used for the action term."""

    controller: DifferentialIKControllerCfg = MISSING
    """The configuration for the differential IK controller."""


