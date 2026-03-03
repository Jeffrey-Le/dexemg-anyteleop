import sys
import os

# ─── Path setup (must come before any local imports) ─────────────────────────
from config import BASE_DIR, DEX_DIR, ROBOT_URDF, RETARGETING_CONFIG

sys.path.insert(0, os.path.join(DEX_DIR, "example/vector_retargeting"))
sys.path.insert(0, os.path.join(DEX_DIR, "src"))
sys.path.insert(0, DEX_DIR)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import cv2
import pinocchio as pin
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from loop_rate_limiters import RateLimiter
import mediapipe as mp

import config as cfg
from utils import get_hand_rotation, smooth_rotation
from robot_setup import build_scene, build_ik
from camera_setup import build_camera
from retargeting_setup import build_retargeter

def set_joint_drive(joint, stiffness, damping, force_limit):
    try:
        joint.set_drive_property(float(stiffness), float(damping), float(force_limit))
        return
    except TypeError:
        pass
    try:
        joint.set_drive_property(stiffness=float(stiffness), damping=float(damping), force_limit=float(force_limit))
        return
    except TypeError:
        pass
    try:
        joint.set_drive_property(float(stiffness), float(damping))
    except TypeError:
        try:
            joint.set_drive_property(stiffness=float(stiffness), damping=float(damping))
        except TypeError:
            print("[err] cannot set drive for joint", joint)
            return

# ─── MediaPipe ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=cfg.MP_MAX_HANDS,
    min_detection_confidence=cfg.MP_DETECTION_CONF,
    min_tracking_confidence=cfg.MP_TRACKING_CONF,
)

from single_hand_detector import SingleHandDetector
detector = SingleHandDetector(hand_type="Right", selfie=False)

# ─── Robot / IK ───────────────────────────────────────────────────────────────
scene, robot, viewer, table, objects = build_scene(object="bottle")  # box / grasp_box / cylinder / bottle
model, data, configuration, q_init, configuration_limit = build_ik(robot)

init_palm = configuration.get_transform_frame_to_world("palm")
print("init palm position :", init_palm.translation)
print("init palm rotation :\n", init_palm.rotation)

palm_task    = FrameTask("palm", position_cost=cfg.POSITION_COST, orientation_cost=cfg.ORIENTATION_COST)
posture_task = PostureTask(cost=cfg.POSTURE_COST)
posture_task.set_target(q_init)
palm_task.set_target(init_palm)

# ─── Retargeter ───────────────────────────────────────────────────────────────
retargeter, retargeting_to_sapien = build_retargeter(robot)

# ─── Camera ───────────────────────────────────────────────────────────────────
pipeline, align = build_camera()

# ─── State ────────────────────────────────────────────────────────────────────
init_pos           = init_palm.translation.copy()
palm_down_rotation = init_palm.rotation.copy()

smoothed_pos       = init_pos.copy()
smoothed_rotation  = palm_down_rotation.copy()
current_rotation   = palm_down_rotation.copy()
last_good_pos      = init_pos.copy()
last_good_rotation = palm_down_rotation.copy()

initial_R_hand       = None
hand_seen            = False
hand_lost_frames     = 0
position_initialized = False
hand_entry_pos       = None

smoothed_hand_qpos  = None

rate = RateLimiter(frequency=cfg.LOOP_FREQUENCY)

R_mp_to_sim = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
], dtype=np.float64)

# bookkeeping
arm_dof = robot.get_dof() - len(retargeting_to_sapien)
finger_start = arm_dof
finger_count = len(retargeting_to_sapien)
print(f"Arm DOF: {arm_dof}, Finger DOF: {finger_count}")

arm_dof = 6
finger_joints = robot.get_active_joints()[arm_dof:]
prev_finger_target = None

def set_joint_drive(j, kp, kd, fmax):
    # SAPIEN 3 beta safe calling
    try:
        j.set_drive_property(float(kp), float(kd), float(fmax))
        return
    except TypeError:
        pass
    try:
        j.set_drive_property(stiffness=float(kp), damping=float(kd), force_limit=float(fmax))
        return
    except TypeError:
        pass
    try:
        j.set_drive_property(float(kp), float(kd))
    except TypeError:
        j.set_drive_property(stiffness=float(kp), damping=float(kd))

# Conservative gains (increase slowly)
Kp_finger = 120.0
Kd_finger = 8.0
Fmax_finger = 25.0

for j in finger_joints:
    set_joint_drive(j, Kp_finger, Kd_finger, Fmax_finger)

SUBSTEPS = 8   # 30 Hz control -> 240 Hz physics-ish
prev_finger_target = None

# ─── Main Loop ────────────────────────────────────────────────────────────────
while not viewer.closed:
    frames = pipeline.wait_for_frames(timeout_ms=15000)
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data()) * 0.001
    color_rgb   = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    results = hands.process(color_rgb)
    _, joint_pos, keypoint_2d, _ = detector.detect(color_rgb)

    dbg = detector.draw_skeleton_on_image(color_image.copy(), keypoint_2d, style="default")
    cv2.imshow("Hand Tracking", dbg)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if results.multi_hand_landmarks:
        hand_lost_frames = 0
        hand_seen = True
        landmarks = results.multi_hand_landmarks[0]
        wrist = landmarks.landmark[0]

        # ── Depth ─────────────────────────────────────────────────────────────
        px = int(np.clip(wrist.x * cfg.RS_WIDTH,  3, cfg.RS_WIDTH  - 4))
        py = int(np.clip(wrist.y * cfg.RS_HEIGHT, 3, cfg.RS_HEIGHT - 4))
        patch = depth_image[py-3:py+3, px-3:px+3]
        valid = patch[patch > 0]
        depth = float(np.median(valid)) if len(valid) > 0 else 0.35

        # ── Position ──────────────────────────────────────────────────────────
        if not position_initialized:
            hand_entry_pos = np.array([wrist.x, wrist.y, depth])
            smoothed_pos = init_pos.copy()
            position_initialized = True
        else:
            dx_phy    = depth   - hand_entry_pos[2]
            dy_screen = wrist.x - hand_entry_pos[0]
            dz_screen = wrist.y - hand_entry_pos[1]

            sim_x = init_pos[0] + np.interp(dx_phy,    cfg.DX_PHY_RANGE,    cfg.DX_SIM_RANGE)
            sim_y = init_pos[1] + np.interp(dy_screen, cfg.DY_SCREEN_RANGE, cfg.DY_SIM_RANGE)
            sim_z = init_pos[2] + np.interp(dz_screen, cfg.DZ_SCREEN_RANGE, cfg.DZ_SIM_RANGE)

            target_pos = np.array([sim_x, sim_y, sim_z])
            smoothed_pos = cfg.ALPHA * target_pos + (1 - cfg.ALPHA) * smoothed_pos

        # ── Rotation ──────────────────────────────────────────────────────────
        world_landmarks = results.multi_hand_world_landmarks[0]
        R_hand, confidence = get_hand_rotation(world_landmarks)
        R_hand = R_mp_to_sim @ R_hand

        if initial_R_hand is None:
            initial_R_hand = R_hand.copy()

        R_delta = R_hand @ initial_R_hand.T
        current_rotation = R_delta @ palm_down_rotation

        smoothed_rotation = smooth_rotation(
            current_rotation, smoothed_rotation,
            cfg.ROTATION_ALPHA, cfg.ROTATION_DEADZONE
        )
        current_rotation = smoothed_rotation

        last_good_pos = smoothed_pos.copy()
        last_good_rotation = current_rotation.copy()

        target_pose = pin.SE3.Identity()
        target_pose.translation = smoothed_pos.copy()
        target_pose.rotation = current_rotation
        palm_task.set_target(target_pose)

        # ── Finger retargeting ────────────────────────────────────────────────
        if joint_pos is not None:
            indices        = retargeter.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices   = indices[1, :]
            ref_value      = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            hand_qpos      = retargeter.retarget(ref_value)
            hand_qpos_reordered = hand_qpos[retargeting_to_sapien]

            if smoothed_hand_qpos is None:
                smoothed_hand_qpos = hand_qpos_reordered.copy()
            else:
                smoothed_hand_qpos = (
                    cfg.FINGER_ALPHA * hand_qpos_reordered
                    + (1 - cfg.FINGER_ALPHA) * smoothed_hand_qpos
                )

    else:
        hand_lost_frames += 1
        position_initialized = False
        hand_entry_pos = None

        if hand_lost_frames > cfg.HAND_LOST_THRESHOLD:
            hand_lost_frames = 0
            initial_R_hand = None

        if not hand_seen:
            palm_task.set_target(init_palm)
        else:
            target_pose = pin.SE3.Identity()
            target_pose.translation = last_good_pos.copy()
            target_pose.rotation = last_good_rotation.copy()
            palm_task.set_target(target_pose)

    # ── IK ────────────────────────────────────────────────────────────────────
    velocity = solve_ik(
        configuration,
        [palm_task, posture_task],
        rate.dt,
        solver="quadprog",
        limits=[configuration_limit],
    )
    configuration.integrate_inplace(velocity, rate.dt)

    # ── APPLY JOINTS (arm teleport, fingers physical) ───────────────────────────
    q = robot.get_qpos()
    q[:6] = configuration.q[:6]
    robot.set_qpos(q)

    # Fingers: drive targets (no teleport into contact)
    if smoothed_hand_qpos is not None:
        if prev_finger_target is None:
            prev_finger_target = smoothed_hand_qpos.copy()

        # Clamp finger speed to avoid instant penetration (rad/s -> rad per tick)
        max_step = cfg.FINGER_QVEL_MAX * rate.dt   # e.g., 3 * 1/30 = 0.1 rad
        delta = smoothed_hand_qpos - prev_finger_target
        delta = np.clip(delta, -max_step, max_step)
        cmd = prev_finger_target + delta
        prev_finger_target = cmd.copy()

        for i, j in enumerate(finger_joints):
            j.set_drive_target(float(cmd[i]))

    # Keep pinocchio fingers fixed (prevents finger feedback into IK)
    sync_q = configuration.q.copy()
    sync_q[6:] = q_init[6:]
    configuration.update(sync_q)

    # ── Step physics (substeps for stable contact) ───────────────────────────────
    for _ in range(SUBSTEPS):
        scene.step()

    scene.update_render()
    viewer.render()
    rate.sleep()