#!/usr/bin/env python3
"""
Subscribe to `/gemini2/color/image_raw` and write MP4 segments driven by a JSON marker file.

Marker file schema (written by Isaac recorder):
  {"state":"recording"|"idle", "episode_index": int, "t_sim_s": float}

When marker becomes `recording` for episode N, this script creates:
  <out_dir>/episode_N.mp4

When marker becomes `idle`, it closes the current writer.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import imageio  # type: ignore
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage


def _decode_ros_image(msg: RosImage) -> Optional[np.ndarray]:
    """Decode ROS Image encodings into uint8 HxWx3 RGB."""
    try:
        h = int(msg.height)
        w = int(msg.width)
        if h <= 0 or w <= 0 or msg.data is None:
            return None

        enc = (msg.encoding or "").lower()
        data = msg.data

        if enc in ("rgb8", "bgr8"):
            arr = np.frombuffer(data, dtype=np.uint8)
            expected = h * w * 3
            if arr.size < expected:
                return None
            arr = arr[:expected].reshape((h, w, 3))
            if enc == "bgr8":
                arr = arr[..., ::-1]
            return arr

        if enc in ("rgba8", "bgra8"):
            arr = np.frombuffer(data, dtype=np.uint8)
            expected = h * w * 4
            if arr.size < expected:
                return None
            arr = arr[:expected].reshape((h, w, 4))
            if enc == "bgra8":
                arr = arr[..., [2, 1, 0, 3]]
            return arr[..., :3]

        # Fallback: assume packed 3 channel
        arr = np.frombuffer(data, dtype=np.uint8)
        expected = h * w * 3
        if arr.size < expected:
            return None
        return arr[:expected].reshape((h, w, 3))
    except Exception:
        return None


@dataclass
class MarkerState:
    state: str = "idle"
    episode_index: int = -1
    t_sim_s: float = 0.0


class Gemin2MarkerMp4(Node):
    def __init__(self, marker_path: str, out_dir: str, topic: str, fps: int):
        super().__init__("gemini2_marker_mp4_recorder")
        self._marker_path = marker_path
        self._out_dir = out_dir
        self._topic = topic
        self._fps = fps

        self._lock = threading.Lock()
        self._marker = MarkerState()

        self._writer = None
        self._writer_episode: Optional[int] = None
        self._last_frame_written = 0
        self._latest_frame: Optional[np.ndarray] = None

        os.makedirs(self._out_dir, exist_ok=True)

        self._sub = self.create_subscription(RosImage, self._topic, self._cb, 10)
        self._timer = self.create_timer(0.05, self._poll_marker)

    def _read_marker(self) -> Optional[MarkerState]:
        """
        Return parsed marker, or None if the file is missing / corrupt / mid-write.
        Callers must keep the last good state — a bad read must NOT be treated as ``idle``,
        or the MP4 writer closes immediately after opening.
        """
        try:
            with open(self._marker_path, "r") as f:
                payload = json.load(f)
            state = str(payload.get("state", "idle"))
            if state not in ("recording", "idle"):
                return None
            return MarkerState(
                state=state,
                episode_index=int(payload.get("episode_index", -1)),
                t_sim_s=float(payload.get("t_sim_s", 0.0)),
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError, OSError):
            return None
        except Exception:
            return None

    def _open_writer(self, episode_index: int, first_frame: np.ndarray) -> None:
        out_path = os.path.join(self._out_dir, f"episode_{episode_index:04d}.mp4")
        # imageio will infer resolution from first append_data
        self._writer = imageio.get_writer(out_path, fps=self._fps)
        self._writer_episode = episode_index
        msg = f"MP4 START episode={episode_index} out={out_path}"
        try:
            self.get_logger().info(msg)
        except Exception:
            pass
        print(msg, flush=True)
        self._writer.append_data(first_frame)
        self._last_frame_written = 1

    def _close_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        if self._writer_episode is not None:
            msg = f"MP4 STOP episode={self._writer_episode} frames={self._last_frame_written}"
            try:
                self.get_logger().info(msg)
            except Exception:
                pass
            print(msg, flush=True)
        self._writer = None
        self._writer_episode = None
        self._last_frame_written = 0

    def _poll_marker(self) -> None:
        with self._lock:
            new_m = self._read_marker()
            if new_m is not None:
                self._marker = new_m
            marker_state = self._marker.state
            episode_index = self._marker.episode_index

        if marker_state != "recording":
            if self._writer is not None:
                self._close_writer()
            return

        if episode_index < 0:
            return

        # If a frame arrived before the episode marker, start the writer on that buffer.
        if self._writer is None:
            if self._latest_frame is not None:
                self._open_writer(episode_index, self._latest_frame)

    def _cb(self, msg: RosImage) -> None:
        frame = _decode_ros_image(msg)
        if frame is None:
            return

        with self._lock:
            marker = self._marker

        # Always buffer the latest decoded frame. If marker flips to `recording` after this callback,
        # _poll_marker can still start an MP4 using this buffered frame.
        with self._lock:
            self._latest_frame = frame

        if marker.state != "recording" or marker.episode_index < 0:
            return

        if self._writer is None or self._writer_episode != marker.episode_index:
            # Episode changed; close old and open new (keep it simple)
            if self._writer is not None:
                self._close_writer()
            self._open_writer(marker.episode_index, frame)
            return

        try:
            self._writer.append_data(frame)
            self._last_frame_written += 1
        except Exception:
            # If anything goes wrong, stop writing to avoid infinite errors.
            print("MP4 append_data failed; closing writer", flush=True)
            self._close_writer()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", required=True, help="Path to video episode marker JSON file")
    ap.add_argument("--out-dir", required=True, help="Output directory for episode_XXXX.mp4")
    ap.add_argument("--topic", default="/gemini2/color/image_raw", help="ROS2 image topic")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    rclpy.init()
    node = Gemin2MarkerMp4(args.marker, args.out_dir, args.topic, args.fps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        try:
            node.get_logger().error(f"Recorder crashed: {e}")
        except Exception:
            pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            # Avoid double-shutdown issues if the context is already torn down.
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

