import numpy as np
import sapien.core as sapien
import pinocchio as pin
import pink
from pink.limits import ConfigurationLimit
from scene_setup import build_scene_objects, TABLE_HEIGHT

import config as cfg

def show_collisions(robot, table, objects):
    # Show collisions for regular actors
    for actor in [table] + objects:
        if hasattr(actor, "render_collision"):
            actor.render_collision(True)

    # Show collisions for articulation links (robot)
    for link in robot.get_links():
        if hasattr(link, "render_collision"):
            link.render_collision(True)

def build_scene(object='box'):
    """Build SAPIEN scene, load robot, create viewer."""
    engine   = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 240)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, -1, -1], [1, 1, 1])
    scene.add_ground(altitude=0)
    table, objects = build_scene_objects(scene, object=object)

    loader = scene.create_urdf_loader()
    loader.fix_root_link          = True
    loader.load_nonconvex_collisions = False
    robot = loader.load(cfg.ROBOT_URDF)
    robot.set_root_pose(sapien.Pose([-0.15, 0, TABLE_HEIGHT / 2], [1, 0, 0, 0]))
    # disable_robot_table_collision(robot, table)

    viewer = scene.create_viewer()
    viewer.control_window.show_origin_frame = True
    viewer.set_camera_xyz(*cfg.VIEWER_XYZ)
    viewer.set_camera_rpy(*cfg.VIEWER_RPY)

    show_collisions(robot, table, objects)

    return scene, robot, viewer, table, objects


def build_ik(robot):
    """Build pinocchio model, pink configuration and return initial config."""
    model = pin.buildModelFromUrdf(cfg.ROBOT_URDF)
    data  = model.createData()
    configuration_limit = ConfigurationLimit(model)

    q_init = pin.neutral(model)
    for i, val in enumerate(cfg.Q_INIT):
        q_init[i] = val

    configuration = pink.Configuration(model, data, q_init)
    configuration.update(q_init)
    robot.set_qpos(q_init)

    return model, data, configuration, q_init, configuration_limit
