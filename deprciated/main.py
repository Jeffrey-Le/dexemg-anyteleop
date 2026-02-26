import sys
import numpy as np
from loop_rate_limiters import RateLimiter
import pyrealsense2 as rs
import cv2
import sapien.core as sapien
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.limits import ConfigurationLimit
from dex_retargeting.retargeting_config import RetargetingConfig
import mediapipe as mp
import os

# ─── Configure these paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of this script
DEX_DIR  = os.path.join(BASE_DIR, "dex-retargeting")   # one level below main

RETARGETING_CONFIG  = os.path.join(DEX_DIR, "src/dex_retargeting/configs/teleop/shadow_hand_right_dexpilot.yml")
ROBOT_URDF          = os.path.join(DEX_DIR, "assets/robots/assembly/ur5e_shadow/ur5e_shadow_right_hand_glb.urdf")
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(DEX_DIR, "example/vector_retargeting"))
sys.path.insert(0, os.path.join(DEX_DIR, "src"))
sys.path.insert(0, DEX_DIR)
from single_hand_detector import SingleHandDetector

rate = RateLimiter(frequency=30.0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Setup paths here

# --- Retargeter Setup ---
retargeting_config = RetargetingConfig.load_from_file(RETARGETING_CONFIG)
low_pass_alpha = 0.2
retargeting_config.low_pass_alpha = low_pass_alpha
retargeter = retargeting_config.build()

# --- SAPIEN Setup ---
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()
scene.set_timestep(1/240)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, -1, -1], [1, 1, 1])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader()
loader.fix_root_link = True
loader.load_nonconvex_collisions = False
robot = loader.load(ROBOT_URDF)
robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

viewer = scene.create_viewer()
viewer.control_window.show_origin_frame = True
viewer.set_camera_xyz(x=1.5, y=0, z=1.5)
viewer.set_camera_rpy(r=0, p=-0.5, y=3.14)

# --- Joint mapping ---
sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
retargeting_joint_names = retargeter.joint_names
retargeting_to_sapien = np.array(
    [retargeting_joint_names.index(name) for name in sapien_joint_names if name in retargeting_joint_names]
).astype(int)

# --- Pink Setup ---
model = pin.buildModelFromUrdf(ROBOT_URDF)
data = model.createData()
configuration_limit = ConfigurationLimit(model)

# --- Pinocchio -> SAPIEN joint mapping ---
pin_joint_names = list(model.names)
name_to_qidx = {}
for jname in pin_joint_names:
    jid = model.getJointId(jname)
    if jid == 0:
        continue
    name_to_qidx[jname] = model.idx_qs[jid]

def pin_to_sapien_q(pin_q):
    sapien_q = np.zeros(robot.dof)
    for i, sname in enumerate(sapien_joint_names):
        if sname in name_to_qidx:
            sapien_q[i] = pin_q[name_to_qidx[sname]]
    return sapien_q

# --- Initial pose ---
q_init = pin.neutral(model)
q_init[0] = -0.26
# q_init[0] = 1.0
q_init[1] = -1.25
q_init[2] = np.pi/2
q_init[3] = -0.35
q_init[4] = 1.5
q_init[5] = 1.5

configuration = pink.Configuration(model, data, q_init)
configuration.update(q_init)
robot.set_qpos(q_init)

init_palm = configuration.get_transform_frame_to_world("palm")
print("init palm position:", init_palm.translation)
print("init palm rotation:\n", init_palm.rotation)

palm_task = FrameTask("palm", position_cost=1.0, orientation_cost=3.0)
posture_task = PostureTask(cost=3e-1)
posture_task.set_target(q_init)
palm_task.set_target(init_palm)

# --- Hand Detector ---
detector = SingleHandDetector(hand_type="Right", selfie=False)

# --- RealSense Setup ---
pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(rs_config)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

# Real camera matrix and distortion from RealSense
camera_matrix = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float64)
dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

# --- SO3 helpers ---
def skew_to_vec(S):
    return np.array([S[2,1], S[0,2], S[1,0]], dtype=np.float64)

def so3_log(R):
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return skew_to_vec(w_hat) * theta

def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-6:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def rotmat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
    q = np.array([w,x,y,z], dtype=np.float64)
    return q / np.linalg.norm(q)

def quat_to_rotmat(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)

def slerp(q0, q1, t):
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = q0 + t*(q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = np.arccos(dot)
    sin_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_0
    s1 = np.sin(theta) / sin_0
    return s0*q0 + s1*q1

# --- Camera to sim rotation ---
R_cam_to_sim = np.array([
     [ 1,  0,  0],
    [0,  0,  1],
    [ 0, 1,  0]
])

# --- Workspace ---
init_pos = init_palm.translation.copy()
sim_x_fixed = init_pos[0]
sim_x_range_phy = (0.2, 0.5)
sim_x_range_sim = (init_pos[0] - 0.8, init_pos[0] + 0.8)  # wider sim range
sim_y_range = (init_pos[1] - 0.3, init_pos[1] + 0.3)
sim_z_range = (init_pos[2] - 0.2, init_pos[2] + 0.2)

alpha = 0.2
rotation_alpha = 0.15
delta_alpha = 0.5

smoothed_delta_w = np.zeros(3)
hand_qpos_reordered = None
initial_R_sim = None
hand_seen = False

def rot_x_pi():
    return np.array([[1, 0, 0],
                     [0,-1, 0],
                     [0, 0,-1]], dtype=np.float64)

def rot_y_pi():
    return np.array([[-1, 0, 0],
                     [ 0, 1, 0],
                     [ 0, 0,-1]], dtype=np.float64)

palm_down_rotation = init_palm.rotation.copy()
palm_down_rotation = palm_down_rotation #@ rot_x_pi()
palm_down_rotation = palm_down_rotation #@ rot_y_pi()

smoothed_pos = init_pos.copy()
smoothed_rotation = palm_down_rotation.copy()
current_rotation = palm_down_rotation.copy()
last_good_pos = init_pos.copy()
last_good_rotation = palm_down_rotation.copy()

hand_lost_frames = 0
HAND_LOST_THRESHOLD = 10  # reset after 10 frames without hand
max_rotation_speed = np.deg2rad(30)  # max degrees per frame

position_initialized = False
hand_entry_pos = None      # screen position when hand first detected
hand_entry_sim = None      # sim position at that moment

# Before the loop
smoothed_hand_qpos = None
finger_alpha = 0.3  # lower = smoother fingers

fist_state = False
FIST_ENTER_THRESHOLD = 0.13  # stricter - must be clearly closed to enter
FIST_EXIT_THRESHOLD = 0.18   # looser - must be clearly open to exit

# --- Main Loop ---
while not viewer.closed:
    frames = pipeline.wait_for_frames(timeout_ms=15000)
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data()) * 0.001
    color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    results = hands.process(color_rgb)
    _, joint_pos, keypoint_2d, _ = detector.detect(color_rgb)
    color_image = detector.draw_skeleton_on_image(color_image, keypoint_2d, style="default")
    cv2.imshow("Hand Tracking", color_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if results.multi_hand_landmarks:
        hand_lost_frames = 0  # reset counter when hand seen
        hand_seen = True
        landmarks = results.multi_hand_landmarks[0]
        wrist = landmarks.landmark[0]

        # --- Depth ---
        px = int(wrist.x * 640)
        py = int(wrist.y * 480)
        px = np.clip(px, 3, 636)
        py = np.clip(py, 3, 476)
        patch = depth_image[py-3:py+3, px-3:px+3]
        valid = patch[patch > 0]
        if len(valid) > 0:
            depth = np.median(valid)
            # print("depth:", depth) 
            sim_x = np.interp(depth, sim_x_range_phy, sim_x_range_sim)
        else:
            sim_x = sim_x_fixed

        # --- Position ---
        if not position_initialized:
            # First frame of detection - record where hand entered
            hand_entry_pos = np.array([wrist.x, wrist.y, depth])
            hand_entry_sim = init_pos.copy()
            smoothed_pos = init_pos.copy()
            position_initialized = True
        else:
            # Track DELTA from entry point, apply to init_pos
            dx_phy = depth - hand_entry_pos[2]
            dy_screen = wrist.x - hand_entry_pos[0]
            dz_screen = wrist.y - hand_entry_pos[1]

            sim_x = init_pos[0] + np.interp(dx_phy, [-0.15, 0.15], [-0.4, 0.4])
            sim_y = init_pos[1] + np.interp(dy_screen, [-0.3, 0.3], [-0.3, 0.3])
            sim_z = init_pos[2] + np.interp(dz_screen, [-0.3, 0.3], [0.2, -0.2])

            target_pos = np.array([sim_x, sim_y, sim_z])
            smoothed_pos = alpha * target_pos + (1 - alpha) * smoothed_pos

        # --- Rotation via MP Landmarks ---
        world_landmarks = results.multi_hand_world_landmarks[0]

        wrist_3d    = np.array([world_landmarks.landmark[0].x,  world_landmarks.landmark[0].y,  world_landmarks.landmark[0].z])
        index_mcp   = np.array([world_landmarks.landmark[5].x,  world_landmarks.landmark[5].y,  world_landmarks.landmark[5].z])
        pinky_mcp   = np.array([world_landmarks.landmark[17].x, world_landmarks.landmark[17].y, world_landmarks.landmark[17].z])
        middle_mcp  = np.array([world_landmarks.landmark[9].x,  world_landmarks.landmark[9].y,  world_landmarks.landmark[9].z])
        ring_mcp    = np.array([world_landmarks.landmark[13].x, world_landmarks.landmark[13].y, world_landmarks.landmark[13].z])

        # X axis: across the palm (pinky to index)
        x_axis = index_mcp - pinky_mcp
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Y axis: up the palm (wrist to middle)
        y_axis = middle_mcp - wrist_3d
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Z axis: palm normal, recomputed from x and y to ensure consistency
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Recompute x to ensure orthogonality
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        R_hand_cam = np.column_stack([x_axis, y_axis, z_axis])

        # Ensure valid rotation
        U, _, Vt = np.linalg.svd(R_hand_cam)
        R_hand_cam = U @ Vt
        if np.linalg.det(R_hand_cam) < 0:
            U[:, -1] *= -1
            R_hand_cam = U @ Vt

        R_sim = R_cam_to_sim @ R_hand_cam

        if initial_R_sim is None:
            initial_R_sim = R_sim.copy()

        R_delta = R_sim @ initial_R_sim.T
        current_rotation = R_delta @ palm_down_rotation

        # Deadzone - only update if rotation changed more than 2 degrees
        q_cur = rotmat_to_quat(current_rotation)
        q_smooth = rotmat_to_quat(smoothed_rotation)
        dot = abs(np.dot(q_cur, q_smooth))
        angle_diff = 2 * np.arccos(np.clip(dot, -1.0, 1.0))

        if angle_diff > np.deg2rad(2):  # only update if moved more than 2 degrees
            q_new = slerp(q_smooth, q_cur, rotation_alpha)
            smoothed_rotation = quat_to_rotmat(q_new)
            current_rotation = smoothed_rotation
        else:
            current_rotation = smoothed_rotation  # hold current

        last_good_pos = smoothed_pos.copy()
        last_good_rotation = current_rotation.copy()

        target_pose = pin.SE3.Identity()
        target_pose.translation = smoothed_pos.copy()
        target_pose.rotation = current_rotation
        palm_task.set_target(target_pose)

        # --- Finger Retargeting ---
        if joint_pos is not None:
            indices = retargeter.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            hand_qpos = retargeter.retarget(ref_value)
            hand_qpos_reordered = hand_qpos[retargeting_to_sapien]
            if smoothed_hand_qpos is None:
                smoothed_hand_qpos = hand_qpos_reordered.copy()
            else:
                smoothed_hand_qpos = finger_alpha * hand_qpos_reordered + (1 - finger_alpha) * smoothed_hand_qpos
    else:
        hand_entry_pos = None
        hand_entry_sim = None
        position_initialized = False  # reset so next entry captures new anchor
        hand_lost_frames += 1
        if hand_lost_frames > HAND_LOST_THRESHOLD:
            hand_lost_frames = 0
            initial_R_sim = None

        if not hand_seen:
            palm_task.set_target(init_palm)
        else:
            target_pose = pin.SE3.Identity()
            target_pose.translation = last_good_pos.copy()
            target_pose.rotation = last_good_rotation.copy()
            palm_task.set_target(target_pose)

    # Inverse Kinematics -> Final Pose
    velocity = solve_ik(
        configuration,
        [palm_task, posture_task],
        rate.dt,
        solver="quadprog",
        limits=[configuration_limit]
    )
    configuration.integrate_inplace(velocity, rate.dt)

    # Only take arm joints from IK
    full_qpos = configuration.q.copy()
    arm_qpos = full_qpos[:6].copy()

    # Apply arm + finger joints separately
    if smoothed_hand_qpos is not None:
        full_qpos[:6] = arm_qpos
        full_qpos[6:] = smoothed_hand_qpos
    else:
        full_qpos[:6] = arm_qpos

    robot.set_qpos(full_qpos)

    # Keep pinocchio in sync with ONLY arm joints
    # Prevents finger joints from feeding back into IK next frame
    sync_q = configuration.q.copy()
    sync_q[6:] = q_init[6:]  # reset fingers to neutral for IK
    configuration.update(sync_q)

    scene.step()
    scene.update_render()
    viewer.render()
    rate.sleep()

