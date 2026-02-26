import numpy as np
from dex_retargeting.retargeting_config import RetargetingConfig
import config as cfg


def build_retargeter(robot):
    """Build dex-retargeter and joint index mapping to SAPIEN."""
    retargeting_config           = RetargetingConfig.load_from_file(cfg.RETARGETING_CONFIG)
    retargeting_config.low_pass_alpha = cfg.FINGER_ALPHA
    retargeter                   = retargeting_config.build()

    sapien_joint_names      = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeter.joint_names

    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name)
         for name in sapien_joint_names
         if name in retargeting_joint_names]
    ).astype(int)

    return retargeter, retargeting_to_sapien
