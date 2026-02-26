import pyrealsense2 as rs
import config as cfg


def build_camera():
    """Start RealSense pipeline and return (pipeline, align)."""
    pipeline  = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, cfg.RS_WIDTH, cfg.RS_HEIGHT, rs.format.bgr8, cfg.RS_FPS)
    rs_config.enable_stream(rs.stream.depth, cfg.RS_WIDTH, cfg.RS_HEIGHT, rs.format.z16,  cfg.RS_FPS)
    pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    return pipeline, align
