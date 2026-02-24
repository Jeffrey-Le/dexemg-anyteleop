import sapien.core as sapien
import numpy as np

# Scene setup
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)

scene = engine.create_scene()
scene.set_timestep(1/240)

# Lighting
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, -1, -1], [1, 1, 1])

# Ground
scene.add_ground(altitude=0)

# Load robot
loader = scene.create_urdf_loader()
loader.fix_root_link = True  # mounts the base to the world, simulates being bolted to a surface

robot = loader.load("./assets/robots/assembly/ur5e_shadow/ur5e_shadow_right_hand_glb.urdf")
robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

# Viewer
viewer = scene.create_viewer()
viewer.set_camera_xyz(x=1.5, y=0, z=1.5)
viewer.set_camera_rpy(r=0, p=-0.5, y=3.14)

# Main loop
while not viewer.closed:
    scene.step()
    scene.update_render()
    viewer.render()