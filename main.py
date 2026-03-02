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
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.limits import ConfigurationLimit
from loop_rate_limiters import RateLimiter
import mediapipe as mp

import config as cfg
from utils import get_hand_rotation, smooth_rotation, rotmat_to_quat, quat_to_rotmat, slerp
from robot_setup import build_scene, build_ik
from scene_setup import build_scene_objects
from camera_setup import build_camera
from retargeting_setup import build_retargeter

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
# scene, robot, viewer = build_scene()
scene, robot, viewer, table, boxes = build_scene()
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
init_pos          = init_palm.translation.copy()
palm_down_rotation = init_palm.rotation.copy()

smoothed_pos      = init_pos.copy()
smoothed_rotation = palm_down_rotation.copy()
current_rotation  = palm_down_rotation.copy()
last_good_pos     = init_pos.copy()
last_good_rotation = palm_down_rotation.copy()

initial_R_hand       = None
hand_seen            = False
hand_lost_frames     = 0
position_initialized = False
hand_entry_pos       = None

smoothed_hand_qpos   = None
hand_qpos_reordered  = None

sim_x_fixed = init_pos[0]

rate = RateLimiter(frequency=cfg.LOOP_FREQUENCY)

R_mp_to_sim = np.array([
    [ 1,  0, 0],
    [0,  0,  1],
    [ 0,  1,  0]
], dtype=np.float64)

robot_dof = robot.get_dof() - len(retargeting_to_sapien)

print(f"Robot Hand DoF: {robot_dof}")

# --- One-time DOF bookkeeping ---
arm_dof = robot.get_dof() - len(retargeting_to_sapien)
finger_start = arm_dof
finger_joints = robot.get_active_joints()[finger_start:]
assert len(finger_joints) == len(retargeting_to_sapien), "Finger joints slice mismatch"

# --- One-time finger PD drive setup (tune these for snappy grasp) ---
Kp_finger = 8000.0   # try 800 -> 6000
Kd_finger = 300.0    # try 40 -> 300
Fmax_finger = 500.0   # try 10 -> 200

for j in finger_joints:
    j.set_drive_property(stiffness=Kp_finger, damping=Kd_finger, force_limit=Fmax_finger)

# ─── Main Loop ────────────────────────────────────────────────────────────────
while not viewer.closed:
    frames = pipeline.wait_for_frames(timeout_ms=15000)
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    import numpy as np
    import pyrealsense2 as rs

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data()) * 0.001
    color_rgb   = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    results = hands.process(color_rgb)
    _, joint_pos, keypoint_2d, _ = detector.detect(color_rgb)
    color_image = detector.draw_skeleton_on_image(color_image, keypoint_2d, style="default")
    cv2.imshow("Hand Tracking", color_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if results.multi_hand_landmarks:
        hand_lost_frames = 0
        hand_seen        = True
        landmarks        = results.multi_hand_landmarks[0]
        wrist            = landmarks.landmark[0]

        # ── Depth ─────────────────────────────────────────────────────────────
        px = int(np.clip(wrist.x * cfg.RS_WIDTH,  3, cfg.RS_WIDTH  - 4))
        py = int(np.clip(wrist.y * cfg.RS_HEIGHT, 3, cfg.RS_HEIGHT - 4))
        patch = depth_image[py-3:py+3, px-3:px+3]
        valid = patch[patch > 0]
        depth = float(np.median(valid)) if len(valid) > 0 else 0.35

        # ── Position ──────────────────────────────────────────────────────────
        if not position_initialized:
            hand_entry_pos       = np.array([wrist.x, wrist.y, depth])
            smoothed_pos         = init_pos.copy()
            position_initialized = True
        else:
            dx_phy    = depth   - hand_entry_pos[2]
            dy_screen = wrist.x - hand_entry_pos[0]
            dz_screen = wrist.y - hand_entry_pos[1]

            sim_x = init_pos[0] + np.interp(dx_phy,    cfg.DX_PHY_RANGE,    cfg.DX_SIM_RANGE)
            sim_y = init_pos[1] + np.interp(dy_screen,  cfg.DY_SCREEN_RANGE, cfg.DY_SIM_RANGE)
            sim_z = init_pos[2] + np.interp(dz_screen,  cfg.DZ_SCREEN_RANGE, cfg.DZ_SIM_RANGE)

            target_pos   = np.array([sim_x, sim_y, sim_z])
            smoothed_pos = cfg.ALPHA * target_pos + (1 - cfg.ALPHA) * smoothed_pos

        # ── Rotation ──────────────────────────────────────────────────────────
        world_landmarks = results.multi_hand_world_landmarks[0]
        R_hand, confidence = get_hand_rotation(world_landmarks)
        R_hand = R_mp_to_sim @ R_hand

        if initial_R_hand is None:
            initial_R_hand = R_hand.copy()

        R_delta          = R_hand @ initial_R_hand.T
        current_rotation = R_delta @ palm_down_rotation

        effective_alpha = cfg.ROTATION_ALPHA 

        smoothed_rotation = smooth_rotation(
            current_rotation, smoothed_rotation,
            effective_alpha, cfg.ROTATION_DEADZONE
        )
        current_rotation = smoothed_rotation

        last_good_pos      = smoothed_pos.copy()
        last_good_rotation = current_rotation.copy()

        target_pose             = pin.SE3.Identity()
        target_pose.translation = smoothed_pos.copy()
        target_pose.rotation    = current_rotation
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
        hand_lost_frames     += 1
        position_initialized  = False
        hand_entry_pos        = None

        if hand_lost_frames > cfg.HAND_LOST_THRESHOLD:
            hand_lost_frames = 0
            initial_R_hand   = None

        if not hand_seen:
            palm_task.set_target(init_palm)
        else:
            target_pose             = pin.SE3.Identity()
            target_pose.translation = last_good_pos.copy()
            target_pose.rotation    = last_good_rotation.copy()
            palm_task.set_target(target_pose)

    # ── IK ────────────────────────────────────────────────────────────────────
    velocity = solve_ik(
        configuration,
        [palm_task, posture_task],
        rate.dt,
        solver="quadprog",
        limits=[configuration_limit],
    )

    # ── Apply ARM via qvel (physics-safe) ──────────────────────────────────────
    qvel = np.zeros(robot.get_dof(), dtype=np.float64)

    # IMPORTANT: use arm_dof (not robot_dof) here
    qvel[:arm_dof] = velocity[:arm_dof]

    # optional arm clamp (helps stability near contact)
    max_arm_qvel = 2.5
    qvel[:arm_dof] = np.clip(qvel[:arm_dof], -max_arm_qvel, max_arm_qvel)

    robot.set_qvel(qvel)

    # ── Apply FINGERS via PD drive targets (Dex qpos -> PD target) ─────────────
    if smoothed_hand_qpos is not None:
        # (Optional) clamp to finger limits to avoid tiny negative values
        # Better: use true per-joint limits if you have them
        finger_tgt = np.clip(smoothed_hand_qpos, 0.0, 1.57079632679)

        for j, tgt in zip(finger_joints, finger_tgt):
            j.set_drive_target(float(tgt))

    # ── Step physics ──────────────────────────────────────────────────────────
    scene.step()

    # ── Sync Pink/Pinocchio from simulated qpos (with tolerance clamp) ─────────
    q = robot.get_qpos().copy()

    lower = model.lowerPositionLimit.copy()
    upper = model.upperPositionLimit.copy()
    tol = 1e-4

    # If your pinocchio q includes extra entries (floating base), this will mismatch.
    # In your case you appear to be using fixed-base, so shapes should match.
    q = np.minimum(np.maximum(q, lower + tol), upper - tol)

    configuration.update(q)

    scene.update_render()
    viewer.render()
    rate.sleep()
