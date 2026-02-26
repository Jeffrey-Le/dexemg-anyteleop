import numpy as np
import cv2

# --- PnP Setup ---
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

def solve_hand_pnp(pts_2d, camera_matrix, dist_coeffs):
    palm_2d = pts_2d[PALM_INDICES].astype(np.float64)
    success, rvec, tvec = cv2.solvePnP(
        PALM_POINTS_3D,
        palm_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    return success, rvec, tvec