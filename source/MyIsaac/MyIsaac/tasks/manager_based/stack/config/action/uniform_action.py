import torch
from collections.abc import Sequence
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs import ManagerBasedEnv
from .actions_cfg import ObjectSurfaceActionTermCfg


class UniformAction(ActionTerm):

    cfg: ObjectSurfaceActionTermCfg
    """Configuration for the action term that applies actions to the object surface."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    

    def __init__(self, cfg: ObjectSurfaceActionTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize the action term with the given configuration and environment."""
        super().__init__(cfg, env)

        # Initialize the action term with the number of samples
        self._num_samples = cfg.num_samples

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        
    
    @property
    def action_dim(self) -> int:
        """Return the action dimension."""
        return 2
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions



    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Uniformly sample actions from the range [0, num_samples]"""
        return super().process_actions(actions)
    
    def apply_actions(self):
        return super().apply_actions()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0