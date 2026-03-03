import sapien.core as sapien

# ─── Tunable scene parameters ─────────────────────────────────────────────────
TABLE_HEIGHT    = 0.8
TABLE_THICKNESS = 0.1
TABLE_SIZE      = [1.6, 1.0]  # [length, width]

TABLE_POS       = [0.4, 0.0, TABLE_HEIGHT / 2]

BOX_HALF = [0.03, 0.03, 0.03]
BOXES = [
    ("box_red",   BOX_HALF, [0.55, -0.15, TABLE_HEIGHT + 0.04], [1.0, 0.2, 0.2, 1.0]),
    ("box_green", BOX_HALF, [0.55,  0.00, TABLE_HEIGHT + 0.04], [0.2, 0.9, 0.2, 1.0]),
    ("box_blue",  BOX_HALF, [0.55,  0.15, TABLE_HEIGHT + 0.04], [0.2, 0.4, 1.0, 1.0]),
]

CYLINDERS = [
    ("cylinder_red",   [0.55, -0.15, TABLE_HEIGHT + 0.06], [1.0, 0.2, 0.2, 1.0]),
    ("cylinder_green", [0.55,  0.00, TABLE_HEIGHT + 0.06], [0.2, 0.9, 0.2, 1.0]),
    ("cylinder_blue",  [0.55,  0.15, TABLE_HEIGHT + 0.06], [0.2, 0.4, 1.0, 1.0]),
]


def build_scene_objects(scene: sapien.Scene, object: str = "box"):
    """
    Returns
    -------
    table : sapien.Entity (actor)
    objects : list[sapien.Entity]
    """

    # Friction helps grasping + prevents “ice skating”
    table_phys = scene.create_physical_material(
        static_friction=1.6,
        dynamic_friction=1.3,
        restitution=0.0,
    )
    obj_phys = scene.create_physical_material(
        static_friction=1.4,
        dynamic_friction=1.2,
        restitution=0.0,
    )

    # ── Table ────────────────────────────────────────────────────────────────
    table = _build_box(
        scene=scene,
        name="table",
        half_size=[TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, TABLE_THICKNESS / 2],
        position=TABLE_POS,
        color=[0.72, 0.55, 0.36, 1.0],
        is_static=True,
        phys_mat=table_phys,
        density=None,
    )

    objects = []

    if object == "box":
        for name, half_size, position, color in BOXES:
            objects.append(_build_box(
                scene=scene,
                name=name,
                half_size=half_size,
                position=position,
                color=color,
                is_static=False,
                phys_mat=obj_phys,
                density=10000.0, # heavier => less “blasting”
            ))

    elif object == "grasp_box":
        # bigger, easier blocks for Shadow grasping
        grasp_half = [0.06, 0.06, 0.08]
        grasp_positions = [
            ("grasp_red",   [0.55, -0.15, TABLE_HEIGHT + 0.10], [1.0, 0.2, 0.2, 1.0]),
            ("grasp_green", [0.55,  0.00, TABLE_HEIGHT + 0.10], [0.2, 0.9, 0.2, 1.0]),
            ("grasp_blue",  [0.55,  0.15, TABLE_HEIGHT + 0.10], [0.2, 0.4, 1.0, 1.0]),
        ]
        for name, pos, color in grasp_positions:
            objects.append(_build_box(
                scene=scene,
                name=name,
                half_size=grasp_half,
                position=pos,
                color=color,
                is_static=False,
                phys_mat=obj_phys,
                density=10000.0,
            ))

    elif object == "cylinder":
        # Use capsule collision (stable) as “cylinder”
        for name, position, color in CYLINDERS:
            objects.append(_build_capsule(
                scene=scene,
                name=name,
                radius=0.035,
                half_length=0.06,
                position=position,
                color=color,
                phys_mat=obj_phys,
                density=8000.0,
            ))
    elif object == "bottle":
        bottles = [
            ("bottle_blue", [0.55, 0.07, TABLE_HEIGHT + 0.01]),
        ]
        for name, pos in bottles:
            objects.append(build_bottle(scene, name=name, position=pos, color=(0.2, 0.6, 1.0, 0.9)))
    elif object == "grasp_test":
        for name, pos, color in GRASP_TEST:
            objects.append(_build_grasp_test_cylinder(scene, name, pos, color))
    else:
        raise ValueError(f"Unknown object='{object}'. Use 'box', 'grasp_box', or 'cylinder'.")

    return table, objects


# ─── Internal helpers ─────────────────────────────────────────────────────────
def _add_box_collision_with_density(builder, *, half_size, material, density):
    try:
        builder.add_box_collision(half_size=half_size, material=material, density=density)
        return
    except TypeError:
        pass
    try:
        builder.add_box_collision(half_size=half_size, material=material, material_density=density)
        return
    except TypeError:
        pass
    builder.add_box_collision(half_size=half_size, material=material)


def _add_capsule_collision_with_density(builder, *, radius, half_length, material, density):
    try:
        builder.add_capsule_collision(radius=radius, half_length=half_length, material=material, density=density)
        return
    except TypeError:
        pass
    try:
        builder.add_capsule_collision(radius=radius, half_length=half_length, material=material, material_density=density)
        return
    except TypeError:
        pass
    builder.add_capsule_collision(radius=radius, half_length=half_length, material=material)


def _build_box(*, scene, name, half_size, position, color, is_static, phys_mat, density):
    builder = scene.create_actor_builder()

    if is_static:
        builder.add_box_collision(half_size=half_size, material=phys_mat)
        builder.add_box_visual(half_size=half_size, material=color)
        ent = builder.build_static(name=name)
        ent.set_pose(sapien.Pose(p=position))
        return ent

    _add_box_collision_with_density(builder, half_size=half_size, material=phys_mat, density=density)
    builder.add_box_visual(half_size=half_size, material=color)
    ent = builder.build(name=name)

    # small clearance prevents initial penetration impulses
    ent.set_pose(sapien.Pose(p=[position[0], position[1], position[2] + 0.005]))
    return ent


def _build_capsule(*, scene, name, radius, half_length, position, color, phys_mat, density):
    builder = scene.create_actor_builder()
    _add_capsule_collision_with_density(
        builder,
        radius=radius,
        half_length=half_length,
        material=phys_mat,
        density=density,
    )
    builder.add_capsule_visual(radius=radius, half_length=half_length, material=color)
    ent = builder.build(name=name)
    ent.set_pose(sapien.Pose(p=[position[0], position[1], position[2] + 0.005]))
    return ent

import sapien.core as sapien

def _try_add_cylinder_collision(builder, *, radius, half_length, material, density, pose):
    # Try common SAPIEN 3 beta signatures
    try:
        builder.add_cylinder_collision(radius=radius, half_length=half_length,
                                       material=material, density=density, pose=pose)
        return True
    except Exception:
        pass
    try:
        builder.add_cylinder_collision(radius=radius, half_length=half_length,
                                       material=material, material_density=density, pose=pose)
        return True
    except Exception:
        pass
    return False

def _try_add_cylinder_visual(builder, *, radius, half_length, color_rgba, pose):
    try:
        builder.add_cylinder_visual(radius=radius, half_length=half_length,
                                    material=color_rgba, pose=pose)
        return True
    except Exception:
        pass
    return False

def _add_box_as_disk(builder, *, radius, half_length, material, density, pose, color_rgba):
    # Fallback: thin square “disk” (works everywhere)
    hs = [radius, radius, half_length]
    try:
        builder.add_box_collision(half_size=hs, material=material, density=density, pose=pose)
    except TypeError:
        try:
            builder.add_box_collision(half_size=hs, material=material, material_density=density, pose=pose)
        except TypeError:
            builder.add_box_collision(half_size=hs, material=material)
    try:
        builder.add_box_visual(half_size=hs, material=color_rgba, pose=pose)
    except TypeError:
        builder.add_box_visual(half_size=hs, material=color_rgba)

def build_bottle(scene: sapien.Scene,
                 name: str,
                 position,
                 color=(0.2, 0.6, 1.0, 0.9)):
    """
    Upright bottle with a flat base disk so it stands.
    Body/neck/cap are capsules (stable). Base is a short cylinder (or thin box fallback).
    """

    mat = scene.create_physical_material(
        static_friction=1.4,
        dynamic_friction=1.2,
        restitution=0.0
    )

    builder = scene.create_actor_builder()

    # --- Dimensions ---
    body_r = 0.032
    body_h = 0.090     # capsule half_length
    neck_r = 0.018
    neck_h = 0.025
    cap_r  = 0.020
    cap_h  = 0.010

    # Flat base disk (this is what makes it stand)
    base_r = 0.040
    base_h = 0.006     # half_length => total thickness ~ 1.2cm

    density = 10000.0

    # Capsules are usually aligned to X; rotate 90° about Y to align with Z (upright)
    upright_q = [0.7071, 0.0, 0.7071, 0.0]

    # Total half-height of capsule body = body_h + body_r
    body_half_height = body_h + body_r
    bottom_z = -body_half_height

    # --- Relative poses (in bottle local frame) ---
    body_pose = sapien.Pose(p=[0, 0, 0], q=upright_q)

    neck_z = (body_h + body_r) + (neck_h + neck_r) - 0.004
    neck_pose = sapien.Pose(p=[0, 0, neck_z], q=upright_q)

    cap_z = neck_z + (neck_h + neck_r) + (cap_h + cap_r) - 0.002
    cap_pose = sapien.Pose(p=[0, 0, cap_z], q=upright_q)

    # Base centered slightly above the bottom so it’s attached under the body
    base_z = bottom_z + base_h
    base_pose = sapien.Pose(p=[0, 0, base_z], q=upright_q)

    # --- Collisions (capsules) ---
    try:
        builder.add_capsule_collision(radius=body_r, half_length=body_h,
                                      material=mat, density=density, pose=body_pose)
    except TypeError:
        builder.add_capsule_collision(radius=body_r, half_length=body_h, material=mat)

    try:
        builder.add_capsule_collision(radius=neck_r, half_length=neck_h,
                                      material=mat, density=density, pose=neck_pose)
    except TypeError:
        builder.add_capsule_collision(radius=neck_r, half_length=neck_h, material=mat)

    try:
        builder.add_capsule_collision(radius=cap_r, half_length=cap_h,
                                      material=mat, density=density, pose=cap_pose)
    except TypeError:
        builder.add_capsule_collision(radius=cap_r, half_length=cap_h, material=mat)

    # --- Collision (base disk) ---
    ok = _try_add_cylinder_collision(builder, radius=base_r, half_length=base_h,
                                    material=mat, density=density, pose=base_pose)
    if not ok:
        _add_box_as_disk(builder, radius=base_r, half_length=base_h,
                         material=mat, density=density, pose=base_pose,
                         color_rgba=(0.85, 0.85, 0.9, 1.0))

    # --- Visuals (capsules) ---
    try:
        builder.add_capsule_visual(radius=body_r, half_length=body_h, material=color, pose=body_pose)
        builder.add_capsule_visual(radius=neck_r, half_length=neck_h, material=(0.92, 0.92, 0.96, 1.0), pose=neck_pose)
        builder.add_capsule_visual(radius=cap_r,  half_length=cap_h,  material=(0.95, 0.95, 0.95, 1.0), pose=cap_pose)
    except TypeError:
        builder.add_capsule_visual(radius=body_r, half_length=body_h, material=color)
        builder.add_capsule_visual(radius=neck_r, half_length=neck_h, material=(0.92, 0.92, 0.96, 1.0))
        builder.add_capsule_visual(radius=cap_r,  half_length=cap_h,  material=(0.95, 0.95, 0.95, 1.0))

    # --- Visual (base disk) ---
    okv = _try_add_cylinder_visual(builder, radius=base_r, half_length=base_h,
                                  color_rgba=(0.85, 0.85, 0.9, 1.0), pose=base_pose)
    if not okv:
        # fallback visual already added in _add_box_as_disk if cylinder not available
        pass

    bottle = builder.build(name=name)

    # Spawn slightly above the table to avoid initial penetration
    bottle.set_pose(sapien.Pose(p=[position[0], position[1], position[2] + 0.015]))

    return bottle

GRASP_TEST = [
    ("grasp_test_cyl", [0.55, 0.00, TABLE_HEIGHT + 0.08], [0.8, 0.8, 0.2, 1.0]),
]

def _build_grasp_test_cylinder(scene, name, position, color):
    mat = scene.create_physical_material(static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
    builder = scene.create_actor_builder()

    # Good Shadow target: ~55–65mm diameter, ~140mm tall
    r = 0.030     # 60mm diameter
    half_len = 0.07  # 140mm tall

    # cylinder collision if available, else capsule (ok)
    try:
        builder.add_cylinder_collision(radius=r, half_length=half_len, material=mat, density=900.0)
        builder.add_cylinder_visual(radius=r, half_length=half_len, material=color)
    except Exception:
        builder.add_capsule_collision(radius=r, half_length=half_len, material=mat, density=900.0)
        builder.add_capsule_visual(radius=r, half_length=half_len, material=color)

    actor = builder.build(name=name)
    actor.set_pose(sapien.Pose(p=[position[0], position[1], position[2] + 0.01]))
    return actor