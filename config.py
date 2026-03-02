import os
import yaml

# ─── Paths (code-dependent, stay in Python) ───────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEX_DIR  = os.path.join(BASE_DIR, "dex-retargeting")

RETARGETING_CONFIG = os.path.join(DEX_DIR, "src/dex_retargeting/configs/teleop/shadow_hand_right_dexpilot.yml")
ROBOT_URDF         = os.path.join(DEX_DIR, "assets/robots/assembly/ur5e_shadow/ur5e_shadow_right_hand_glb.urdf")
# RETARGETING_CONFIG = os.path.join(DEX_DIR, "src/dex_retargeting/configs/teleop/inspire_hand_right_dexpilot.yml")
# ROBOT_URDF         = os.path.join(DEX_DIR, "assets/robots/assembly/rm75_inspire/rm75_inspire_right_hand.urdf")

# ─── Load YAML ────────────────────────────────────────────────────────────────
_yml_path = os.path.join(BASE_DIR, "config.yml")
with open(_yml_path) as f:
    _cfg = yaml.safe_load(f)

# ─── Smoothing ────────────────────────────────────────────────────────────────
ALPHA          = _cfg["alpha"]
ROTATION_ALPHA = _cfg["rotation_alpha"]
FINGER_ALPHA   = _cfg["finger_alpha"]

# ─── IK ───────────────────────────────────────────────────────────────────────
POSITION_COST    = _cfg["position_cost"]
ORIENTATION_COST = _cfg["orientation_cost"]
POSTURE_COST     = _cfg["posture_cost"]

# ─── Initial joint angles ─────────────────────────────────────────────────────
Q_INIT = _cfg["q_init"]

# ─── Workspace ────────────────────────────────────────────────────────────────
SIM_X_RANGE_PHY    = tuple(_cfg["sim_x_range_phy"])
SIM_X_RANGE_OFFSET = _cfg["sim_x_range_offset"]
SIM_Y_RANGE_OFFSET = _cfg["sim_y_range_offset"]
SIM_Z_RANGE_OFFSET = _cfg["sim_z_range_offset"]

DX_PHY_RANGE    = tuple(_cfg["dx_phy_range"])
DX_SIM_RANGE    = tuple(_cfg["dx_sim_range"])
DY_SCREEN_RANGE = tuple(_cfg["dy_screen_range"])
DY_SIM_RANGE    = tuple(_cfg["dy_sim_range"])
DZ_SCREEN_RANGE = tuple(_cfg["dz_screen_range"])
DZ_SIM_RANGE    = tuple(_cfg["dz_sim_range"])

# ─── Hand tracking ────────────────────────────────────────────────────────────
HAND_LOST_THRESHOLD = _cfg["hand_lost_threshold"]
ROTATION_DEADZONE   = _cfg["rotation_deadzone_deg"]
MIN_CONFIDENCE = _cfg["min_confidence"]
FINGER_QVEL_MAX = _cfg["finger_qvel_max"]

# ─── MediaPipe ────────────────────────────────────────────────────────────────
MP_MAX_HANDS      = _cfg["mp_max_hands"]
MP_DETECTION_CONF = _cfg["mp_detection_conf"]
MP_TRACKING_CONF  = _cfg["mp_tracking_conf"]

# ─── RealSense ────────────────────────────────────────────────────────────────
RS_WIDTH  = _cfg["rs_width"]
RS_HEIGHT = _cfg["rs_height"]
RS_FPS    = _cfg["rs_fps"]

# ─── SAPIEN viewer ────────────────────────────────────────────────────────────
VIEWER_XYZ = tuple(_cfg["viewer_xyz"])
VIEWER_RPY = tuple(_cfg["viewer_rpy"])

# ─── Loop ─────────────────────────────────────────────────────────────────────
LOOP_FREQUENCY = _cfg["loop_frequency"]
