import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)  # just depth, low fps

profile = pipeline.start(config)

# Align depth to color frame so pixel coordinates match
align = rs.align(rs.stream.color)

# Get intrinsics
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

mp_draw = mp.solutions.drawing_utils

while True:
    frames = pipeline.wait_for_frames(timeout_ms=15000)
    aligned = align.process(frames)
    
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        continue
    
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data()) * 0.001

    results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw skeleton on frame
            mp_draw.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Print wrist 3D position
            wrist = hand_landmarks.landmark[0]
            px, py = int(wrist.x * 640), int(wrist.y * 480)
            depth = np.median(depth_image[py-3:py+3, px-3:px+3])
            wrist_3d = np.array([
                (px - cx) * depth / fx,
                (py - cy) * depth / fy,
                depth
            ])
            print(f"Wrist 3D: {wrist_3d}")
    
    cv2.imshow("Hand Tracking", color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break