"""
You can download the raw Libero datasets from http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# import tensorflow_datasets as tfds
import tyro

import numpy as np
import os
import re

from pathlib import Path
from typing import Tuple

REPO_NAME = "ZhiyingDu_Pi0_Dataset/calvin"  # Name of the output dataset, also used for the Hugging Face Hub

def lookup_naming_pattern(dataset_dir: Path, save_format: str) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (200, 200, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (84, 84, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (15,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    save_format = "npz"

    naming_pattern, n_digits = lookup_naming_pattern(
        data_dir, save_format
    )
    lang_data = np.load(
        data_dir + "lang_annotations/auto_lang_ann.npy",
        allow_pickle=True,
    ).item()
    ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
    lang_ann = lang_data["language"]["ann"]  # length total number of annotations
    lang_task = lang_data["language"]["task"]
    # print(lang_task[0])
    # print(lang_ann[0])
    print(len(lang_task))
    print(len(lang_ann))
    print(len(ep_start_end_ids))
    print(ep_start_end_ids[-1])
    assert 0 == 1
    for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
        # for idx in range(start_idx, end_idx + 1):
        #     lang_lookup.append(i)
        #     episode_lookup.append(idx)
        for idx in range(start_idx, end_idx + 1):
            file_name = Path(f"{naming_pattern[0]}{idx:0{n_digits}d}{naming_pattern[1]}")
            step = np.load(file_name.as_posix())
            keys = step.files
            # print(keys)
            # print(step["rgb_static"])
            # assert 0 == 1
            # print(step["rgb_static"].shape)
            # # print(step["observation"]["wrist_image"])
            # print(step["rgb_gripper"].shape)
            # print(step['actions'].shape)
            # print(step["actions"])
            # print(step['robot_obs'].shape)
            # print(step['robot_obs'])
            # assert 0 == 1
            dataset.add_frame(
                {
                    "image": step["rgb_static"],
                    "wrist_image": step["rgb_gripper"],
                    "state": step['robot_obs'],
                    "actions": step["actions"],
                }
            )
            
        dataset.save_episode(task=lang_ann[i])
            # assert 0 == 1
    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["calvin", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
