from dex_retargeting.retargeting_config import RetargetingConfig
import inspect
import sys

sys.path.append("./eample")

urdf_path = "./assets/robots/assembly/ur5e_shadow/ur5e_shadow_right_hand_glb_fixed.urdf"

config = RetargetingConfig.load_from_file("./src/dex_retargeting/configs/teleop/shadow_hand_right_dexpilot.yml")
# config.urdf_path = urdf_path
retargeter = config.build()
print("Retargeter built OK")

print(retargeter.optimizer.target_link_human_indices)
print(len(retargeter.optimizer.target_link_human_indices))

print(retargeter.optimizer.target_joint_names)
print(len(retargeter.optimizer.target_joint_names))

print(inspect.signature(retargeter.retarget))