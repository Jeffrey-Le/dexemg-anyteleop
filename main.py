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
import time

sys.path.append("./example/vector_retargeting")
from single_hand_detector import SingleHandDetector

rate = RateLimiter(frequency=30.0) # Increase for faster response -> frequency is fps target

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Retargeter Setup ---
retargeting_config = RetargetingConfig.load_from_file("./src/dex_retargeting/configs/teleop/shadow_hand_right_dexpilot.yml")
low_pass_aplha = 0.2 # lower = smoother fingers
retargeting_config.low_pass_alpha = low_pass_aplha
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
robot = loader.load("./assets/robots/assembly/ur5e_shadow/ur5e_shadow_right_hand_glb.urdf")
robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
# robot.set_root_pose(
#     sapien.Pose([0, 0, 0], [0.7071, 0.7071, 0, 0])
# )

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
urdf_path = "./assets/robots/assembly/ur5e_shadow/ur5e_shadow_right_hand_glb.urdf"
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()
configuration_limit = ConfigurationLimit(model)

# --- Pinocchio -> SAPIEN joint mapping (by name) ---
pin_joint_names = list(model.names)  # includes 'universe'

name_to_qidx = {}
for jname in pin_joint_names:
    jid = model.getJointId(jname)
    if jid == 0:  # universe
        continue
    name_to_qidx[jname] = model.idx_qs[jid]  # start index in q

def pin_to_sapien_q(pin_q):
    sapien_q = np.zeros(robot.dof)
    for i, sname in enumerate(sapien_joint_names):
        if sname in name_to_qidx:
            sapien_q[i] = pin_q[name_to_qidx[sname]]
    return sapien_q

# Inital Starting postion for arm and hand model
q_init = pin.neutral(model)
q_init[0] = -0.26        # shoulder_pan - centers palm
q_init[1] = -np.pi       # (you can revisit later)
q_init[2] = -np.pi/2
q_init[3] = 1.0
q_init[4] = np.pi/2
q_init[5] = 3.4

configuration = pink.Configuration(model, data, q_init)
configuration.update(q_init)
init_palm = configuration.get_transform_frame_to_world("palm")
print("init palm position:", init_palm.translation)
print("init palm rotation:\n", init_palm.rotation)
robot.set_qpos(pin_to_sapien_q(q_init))

palm_task = FrameTask("palm", position_cost=1.0, orientation_cost=0.15) # Higher Orientation Cost = Rotation tracked more aggresively

posture_task = PostureTask(cost=3e-1) # Higher cost = arm stays closer to home pose
posture_task.set_target(q_init)

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

# --- PnP Setup ---
prev_rvec = None
prev_tvec = None

MODEL_POINTS_3D = np.array([
    [  0.0,    0.0,   0.0],
    [ 36.0,   10.0,  20.0],
    [ 52.0,   25.0,  25.0],
    [ 62.0,   45.0,  20.0],
    [ 68.0,   60.0,  15.0],
    [ 30.0,   80.0,   0.0],
    [ 30.0,  115.0,   0.0],
    [ 30.0,  135.0,   0.0],
    [ 30.0,  150.0,   0.0],
    [ 10.0,   85.0,   0.0],
    [ 10.0,  125.0,   0.0],
    [ 10.0,  145.0,   0.0],
    [ 10.0,  160.0,   0.0],
    [-10.0,   80.0,   0.0],
    [-10.0,  115.0,   0.0],
    [-10.0,  135.0,   0.0],
    [-10.0,  147.0,   0.0],
    [-28.0,   72.0,   0.0],
    [-28.0,  100.0,   0.0],
    [-28.0,  115.0,   0.0],
    [-28.0,  125.0,   0.0],
], dtype=np.float64)

PALM_INDICES = [0, 1, 5, 9, 13, 17]
PALM_POINTS_3D = MODEL_POINTS_3D[PALM_INDICES]

camera_matrix = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

def solve_hand_pnp(pts_2d):
    global prev_rvec, prev_tvec
    palm_2d = pts_2d[PALM_INDICES].astype(np.float64)

    # If we don't have a previous pose, solve fresh
    if prev_rvec is None:
        success, rvec, tvec = cv2.solvePnP(
            PALM_POINTS_3D, palm_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    else:
        # Use last solution as a warm start => huge stability boost
        success, rvec, tvec = cv2.solvePnP(
            PALM_POINTS_3D, palm_2d, camera_matrix, dist_coeffs,
            rvec=prev_rvec, tvec=prev_tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    if success:
        prev_rvec, prev_tvec = rvec, tvec
    return success, rvec, tvec

def skew_to_vec(S):
    return np.array([S[2,1], S[0,2], S[1,0]], dtype=np.float64)

def so3_log(R):
    # Returns axis-angle vector (3,)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return skew_to_vec(w_hat) * theta

def so3_exp(w):
    # w is axis-angle vector
    theta = np.linalg.norm(w)
    if theta < 1e-6:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def rotmat_to_quat(R):
    # returns (w,x,y,z)
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
    # ensure shortest path
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

# Camera to sim rotation mapping
R_cam_to_sim = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0]
])

# --- Workspace ---
sim_x_fixed = 0.4
sim_x_range_phy = (0.3, 1.0)
sim_x_range_sim = (0.6, 0.2)
sim_y_range = (-0.5, 0.5)
sim_z_range = (0.2, 0.6)

alpha = 0.05
rotation_alpha = 0.05

smoothed_delta_w = np.zeros(3)  # smoothed axis-angle delta
delta_alpha = 0.05              # 0.02-0.08 typical (lower = smoother)

hand_qpos_reordered = None
pnp_initialized = False
pnp_frame_count = 0
initial_R_sim = None
hand_seen = False  # <--- add this

# Initialize workspace state from the ACTUAL initial palm pose (from q_init)
def rot_x_pi():
    return np.array([[1, 0, 0],
                     [0,-1, 0],
                     [0, 0,-1]], dtype=np.float64)

def rot_y_pi():
    return np.array([[-1, 0, 0],
                     [ 0, 1, 0],
                     [ 0, 0,-1]], dtype=np.float64)

def rot_z_pi():
    return np.array([[-1, 0, 0],
                     [ 0,-1, 0],
                     [ 0, 0, 1]], dtype=np.float64)

init_palm = configuration.get_transform_frame_to_world("palm")
palm_down_rotation = init_palm.rotation.copy()
palm_down_rotation = palm_down_rotation @ rot_x_pi()
palm_down_rotation = palm_down_rotation @ rot_y_pi()

smoothed_pos = init_palm.translation.copy()
smoothed_rotation = palm_down_rotation.copy()
current_rotation = palm_down_rotation.copy()
last_good_pos = smoothed_pos.copy()
last_good_rotation = palm_down_rotation.copy()

# Ensure the IK target starts at the true initial pose
palm_task.set_target(init_palm)

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
        landmarks = results.multi_hand_landmarks[0]
        wrist = landmarks.landmark[0]

        # Get wrist pixel position
        px = int(wrist.x * 640)
        py = int(wrist.y * 480)
        px = np.clip(px, 3, 636)
        py = np.clip(py, 3, 476)

        patch = depth_image[py-3:py+3, px-3:px+3]
        valid = patch[patch > 0]

        if len(valid) > 0:
            depth = np.median(valid)
            
            # Map depth to sim x range
            # Person is typically 0.3m to 1.0m from camera
            sim_x = np.interp(depth, [sim_x_range_phy[0], sim_x_range_phy[1]], [sim_x_range_sim[0], sim_x_range_sim[1]])  # closer = further in sim
        else:
            sim_x = sim_x_fixed  # fallback if no valid depth


        # --- Position ---
        sim_y = np.interp(wrist.x, [0.2, 0.8], [sim_y_range[0], sim_y_range[1]])
        sim_z = np.interp(wrist.y, [0.2, 0.8], [sim_z_range[1], sim_z_range[0]])
        target_pos = np.array([sim_x, sim_y, sim_z])
        smoothed_pos = alpha * target_pos + (1 - alpha) * smoothed_pos

        # --- Rotation via PnP ---
        pts_2d = np.array([[lm.x * 640, lm.y * 480] for lm in landmarks.landmark], dtype=np.float64)
        success, rvec, tvec = solve_hand_pnp(pts_2d)

        if success:
            R_cam, _ = cv2.Rodrigues(rvec)
            R_sim = R_cam_to_sim @ R_cam
             # Capture initial rotation on first detection
            if initial_R_sim is None:
                initial_R_sim = R_sim.copy()

            R_delta = R_sim @ initial_R_sim.T

            # Convert delta rotation to axis-angle, low-pass filter in tangent space
            delta_w = so3_log(R_delta)
            smoothed_delta_w = (1 - delta_alpha) * smoothed_delta_w + delta_alpha * delta_w
            R_delta_smooth = so3_exp(smoothed_delta_w)

            # Apply the smoothed delta to your base "palm down" orientation
            current_rotation = R_delta_smooth @ palm_down_rotation

            # Smooth the rotation
            q_cur = rotmat_to_quat(current_rotation)
            q_smooth = rotmat_to_quat(smoothed_rotation)

            q_new = slerp(q_smooth, q_cur, rotation_alpha)   # rotation_alpha like 0.05 ~ 0.15
            smoothed_rotation = quat_to_rotmat(q_new)
            current_rotation = smoothed_rotation

            # Reject sudden big orientation jumps (snap protection)
            R_err = last_good_rotation.T @ current_rotation
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
            if angle > np.deg2rad(35):
                current_rotation = last_good_rotation.copy()
        else:
            current_rotation = palm_down_rotation  # fallback

        # Save last good values
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

    else: # No hand detected branch
        initial_R_sim = None  # reset when hand disappears

        if not hand_seen:
            # Before the first detection, HOLD the initial pose (don’t drift to a default workspace point)
            palm_task.set_target(init_palm)
        else:
            # After you’ve seen the hand at least once, hold last good pose
            target_pose = pin.SE3.Identity()
            target_pose.translation = last_good_pos.copy()
            target_pose.rotation = last_good_rotation.copy()
            palm_task.set_target(target_pose)
        


    velocity = solve_ik(
        configuration,
        [palm_task, posture_task],
        rate.dt,
        solver="quadprog",
        limits=[configuration_limit]
    )
    configuration.integrate_inplace(velocity, rate.dt)

    full_qpos = configuration.q.copy()
    if hand_qpos_reordered is not None:
        full_qpos[6:] = hand_qpos_reordered
    robot.set_qpos(pin_to_sapien_q(full_qpos))

    scene.step()
    # print("Current qpos:", robot.get_qpos()[:6])  # just the arm joints
    scene.update_render()
    viewer.render()
    rate.sleep()