import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)  # just depth, low fps

pipeline.start(config)
frames = pipeline.wait_for_frames(timeout_ms=15000)
print("Got frames:", frames)