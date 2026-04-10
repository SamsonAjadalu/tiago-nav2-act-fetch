import glob
import os
from pathlib import Path
import numpy as np
import torch
import torchvision.io
from lerobot.datasets.lerobot_dataset import LeRobotDataset

_DEFAULT = str(Path.home() / "isaacsim_data")
_DATA = os.environ.get("TIAGO_ISAACSIM_DATA_ROOT", _DEFAULT)
INPUT_DIR = os.environ.get("TIAGO_LEROBOT_NPZ_DIR", os.path.join(_DATA, "tiago_lerobot_npz"))
OUTPUT_DIR = os.environ.get("TIAGO_LEROBOT_DATASET_DIR", os.path.join(_DATA, "tiago_lerobot_dataset"))
REPO_ID = "local/tiago_teleop"
FPS = 30

def main():
    # Find all episodes
    npz_files = sorted(glob.glob(os.path.join(INPUT_DIR, "episode_*.npz")))
    mp4_files = sorted(glob.glob(os.path.join(INPUT_DIR, "videos", "episode_*.mp4")))

    if len(npz_files) == 0:
        print("No NPZ files found!")
        return
    if len(npz_files) != len(mp4_files):
        print(f"Mismatch! Found {len(npz_files)} NPZ files and {len(mp4_files)} MP4 files.")
        return

    print(f"Found {len(npz_files)} episodes. Creating dataset...")

    # Create dataset
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
                "info": {"video.fps": FPS, "video.codec": "libx264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False}
            }
        }
    )

    for ep_idx, (npz_path, mp4_path) in enumerate(zip(npz_files, mp4_files)):
        print(f"Processing episode {ep_idx}...")
        
        # Load NPZ
        data = np.load(npz_path)
        states = torch.from_numpy(data['joint_pos']).float()
        actions = torch.from_numpy(data['action_joint_pos']).float()
        
        # Load Video
        # torchvision.io.read_video returns (C, T, H, W)
        video_frames, _, _ = torchvision.io.read_video(mp4_path, pts_unit='sec')
        # read_video actually returns (T, H, W, C) for some versions, let's normalize to (T, C, H, W)
        if video_frames.shape[-1] == 3:
            video_frames = video_frames.permute(0, 3, 1, 2)
            
        num_frames = min(len(states), len(video_frames))
        
        for i in range(num_frames):
            frame_dict = {
                "observation.state": states[i],
                "action": actions[i],
                "observation.images.gemini2": video_frames[i],
                "task": "teleop"
            }
            dataset.add_frame(frame_dict)
            
        dataset.save_episode()
        print(f"  Saved episode {ep_idx} with {num_frames} frames.")

    print("Consolidating dataset...")
    # dataset.consolidate()
    print(f"Done! Dataset saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
