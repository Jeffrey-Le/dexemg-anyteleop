import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask
import numpy as np

urdf_path = "./assets/robots/assembly/ur5e_shadow/ur5e_shadow_right_hand_glb_fixed.urdf"

model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# The end effector link name in the URDF - let's find it
for i, frame in enumerate(model.frames):
    print(i, frame.name)


# configuration = pink.Configuration(model, data, pin.neutral(model))

# # Task: move the palm to a target pose
# palm_task = FrameTask(
#     "palm",
#     position_cost=1.0,
#     orientation_cost=0.0  # we only care about position for now
# )

# # Set initial target to current palm position
# configuration.update(pin.neutral(model))
# palm_task.set_target(configuration.get_transform_frame_to_world("palm"))

# print("Pink setup OK")