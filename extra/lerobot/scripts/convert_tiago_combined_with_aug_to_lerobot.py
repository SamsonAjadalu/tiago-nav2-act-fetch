import os
from pathlib import Path

import glob
from typing import List, Tuple

import numpy as np
import torch
import torchvision.io
from lerobot.datasets.lerobot_dataset import LeRobotDataset

_DEFAULT = str(Path.home() / "isaacsim_data")
_DATA = os.environ.get("TIAGO_ISAACSIM_DATA_ROOT", _DEFAULT)

BASE_INPUT_DIR = os.environ.get("TIAGO_LEROBOT_NPZ_DIR_BASE", os.path.join(_DATA, "tiago_lerobot_npz"))
HOUSE_INPUT_DIR = os.environ.get("TIAGO_LEROBOT_NPZ_DIR_HOUSE", os.path.join(_DATA, "tiago_lerobot_npz_house"))
OUTPUT_DIR = os.environ.get("TIAGO_LEROBOT_DATASET_DIR", os.path.join(_DATA, "tiago_lerobot_dataset_combined"))

REPO_ID = os.environ.get("TIAGO_LEROBOT_REPO_ID", "local/tiago_teleop_combined")
FPS = int(os.environ.get("TIAGO_LEROBOT_FPS", "30"))

HOUSE_AUG_FACTOR = int(os.environ.get("TIAGO_HOUSE_AUG_FACTOR", "3"))
HOUSE_AUG_NOISE_STD = float(os.environ.get("TIAGO_HOUSE_AUG_NOISE_STD", "0.005"))


def _list_episodes(root: str) -> Tuple[List[str], List[str]]:
    npz_files = sorted(glob.glob(os.path.join(root, "episode_*.npz")))
    mp4_files = sorted(glob.glob(os.path.join(root, "videos", "episode_*.mp4")))
    return npz_files, mp4_files


def _check_pairs(root: str) -> Tuple[List[str], List[str]]:
    npz_files, mp4_files = _list_episodes(root)
    if len(npz_files) == 0:
        print(f"[WARN] No NPZ files found under {root}")
        return [], []
    if len(npz_files) != len(mp4_files):
        print(f"[ERROR] Mismatch under {root}: {len(npz_files)} NPZ vs {len(mp4_files)} MP4")
        return [], []
    return npz_files, mp4_files


def _process_split(
    dataset: LeRobotDataset,
    root: str,
    aug_factor: int = 1,
    noise_std: float = 0.0,
    tag: str = "",
) -> int:
    npz_files, mp4_files = _check_pairs(root)
    if not npz_files:
        return 0

    total_eps = 0
    for ep_idx, (npz_path, mp4_path) in enumerate(zip(npz_files, mp4_files)):
        print(f"[{tag}] episode {ep_idx}: {os.path.basename(npz_path)}")

        data = np.load(npz_path)
        states = torch.from_numpy(data["joint_pos"]).float()
        actions = torch.from_numpy(data["action_joint_pos"]).float()

        video_frames, _, _ = torchvision.io.read_video(mp4_path, pts_unit="sec")
        if video_frames.shape[-1] == 3:
            video_frames = video_frames.permute(0, 3, 1, 2)

        num_frames = min(len(states), len(video_frames))
        if num_frames == 0:
            continue

        for k in range(max(1, aug_factor)):
            if k == 0 or noise_std <= 0.0:
                s_k = states
                a_k = actions
            else:
                noise_s = torch.randn_like(states) * noise_std
                noise_a = torch.randn_like(actions) * noise_std
                s_k = states + noise_s
                a_k = actions + noise_a

            for i in range(num_frames):
                frame_dict = {
                    "observation.state": s_k[i],
                    "action": a_k[i],
                    "observation.images.gemini2": video_frames[i],
                    "task": "teleop_house" if tag == "HOUSE" else "teleop_base",
                }
                dataset.add_frame(frame_dict)
            dataset.save_episode()
            total_eps += 1
    return total_eps


def main() -> None:
    print(f"Base NPZ root:   {BASE_INPUT_DIR}")
    print(f"House NPZ root:  {HOUSE_INPUT_DIR}")
    print(f"Output dataset:  {OUTPUT_DIR}")
    print(f"House aug: factor={HOUSE_AUG_FACTOR} noise_std={HOUSE_AUG_NOISE_STD}")

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=OUTPUT_DIR,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (85,),
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": (85,),
                "names": None,
            },
            "observation.images.gemini2": {
                "dtype": "video",
                "shape": (3, 720, 1280),
                "names": ["c", "h", "w"],
                "info": {
                    "video.fps": FPS,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                },
            },
        },
    )

    base_eps = _process_split(dataset, BASE_INPUT_DIR, aug_factor=1, noise_std=0.0, tag="BASE")
    house_eps = _process_split(
        dataset,
        HOUSE_INPUT_DIR,
        aug_factor=max(1, HOUSE_AUG_FACTOR),
        noise_std=HOUSE_AUG_NOISE_STD if HOUSE_AUG_FACTOR > 1 else 0.0,
        tag="HOUSE",
    )

    print(f"Done. Wrote {base_eps} base episodes + {house_eps} house/aug episodes to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

