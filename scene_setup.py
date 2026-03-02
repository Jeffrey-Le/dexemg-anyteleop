import numpy as np
import sapien.core as sapien


# ─── Tunable scene parameters ─────────────────────────────────────────────────
TABLE_HEIGHT    = 0.8       # metres
TABLE_THICKNESS = 0.1
TABLE_SIZE      = [1.6, 1.0]  # [length, width]
TABLE_POS       = [0.4, 0.0, TABLE_HEIGHT / 2]  # in front of robot base

BOX_SIZE = [0.03, 0.03, 0.03]
BOXES = [
    # (name,        half_size,            position,                  color_rgba)
    ("box_red",   BOX_SIZE,  [0.55, -0.15, TABLE_HEIGHT + 0.04],  [1.0, 0.2, 0.2, 1.0]),
    ("box_green", BOX_SIZE,  [0.55,  0.00, TABLE_HEIGHT + 0.04],  [0.2, 0.9, 0.2, 1.0]),
    ("box_blue",  BOX_SIZE,  [0.55,  0.15, TABLE_HEIGHT + 0.04],  [0.2, 0.4, 1.0, 1.0]),
]


def build_scene_objects(scene: sapien.Scene):
    """
    Add a table and coloured pick-and-place boxes to an existing SAPIEN scene.

    Returns
    -------
    table : sapien.Actor
    boxes : list[sapien.Actor]
    """
    # ── Table ─────────────────────────────────────────────────────────────────
    table = _build_box(
        scene,
        name      = "table",
        half_size = [TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, TABLE_THICKNESS / 2],
        position  = TABLE_POS,
        color     = [0.72, 0.55, 0.36, 1.0],   # wood-ish brown
        is_static = True,
    )

    # ── Boxes ─────────────────────────────────────────────────────────────────
    boxes = []
    for name, half_size, position, color in BOXES:
        box = _build_box(
            scene,
            name      = name,
            half_size = half_size,
            position  = position,
            color     = color,
            is_static = False,
        )
        boxes.append(box)

    return table, boxes


# ─── Internal helper ──────────────────────────────────────────────────────────

def _build_box(scene, name, half_size, position, color, is_static):
    """Create a box actor (static or dynamic) and add it to the scene."""
    if is_static:
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=color)
        actor = builder.build_static(name=name)
    else:
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=color)
        actor = builder.build(name=name)

    actor.set_pose(sapien.Pose(p=position))
    return actor
