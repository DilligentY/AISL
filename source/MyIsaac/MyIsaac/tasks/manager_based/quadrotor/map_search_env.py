from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CommandManager
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnv
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import MyIsaac.tasks.manager_based.quadrotor.mdp as mdp


class SearchSceneCfg(InteractiveSceneCfg):
    # world
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # robots
    robot_1 : ArticulationCfg = MISSING
    roobt_2 : ArticulationCfg = MISSING
    robot_3 : ArticulationCfg = MISSING
    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    
    goal_pose = MISSING




@configclass
class ActionsCfg:

    target_point : ActionTerm = MISSING
    

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy : PolicyCfg = PolicyCfg()


@configclass
class EventCfg:

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range" : (0.5, 1.5),
            "velocity_range" : (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:

    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        # 함수에 넘겨줄 파라미터를 설정
        params={"asset_cfg" : SceneEntityCfg("robot", body_names=MISSING), "command_name" : "ee_pose"}  
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        # 함수에 넘겨줄 파라미터를 설정
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        # 함수에 넘겨줄 파라미터를 설정
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        # 함수에 넘겨줄 파라미터를 설정
        params={"asset_cfg" : SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class SearchEnvCfg(ManagerBasedRLEnvCfg):

    scene : SearchSceneCfg= SearchSceneCfg(num_envs=2048, env_spacing=2.5)
    observations : ObservationsCfg = ObservationsCfg()
    actions : ActionsCfg = ActionsCfg()
    commands : CommandsCfg = CommandsCfg()

    rewards : RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):

        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim_dt = 1.0 / 10.0