"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

from moevla.training.mixtures import DATASET_MIXTURES
import numpy as np
import tqdm
import tyro
import torch
import moevla.shared.normalize as normalize
import moevla.training.config as _config
import moevla.training.data_loader as _data_loader
import moevla.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    if len(config.total_configs) != 1:
        raise ValueError(
            f"Expected exactly one dataset in training config {config.config_name}, "
            f"but found {len(config.total_configs)}: {config.total_configs}"
        )
    data_config = config.total_configs[0].data.create(config.total_configs[0].assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    
    dataset, num_frames, num_episodes = _data_loader.create_dataset([data_config], [config.total_configs[0].assets_dirs], config.model, [1.0])
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            [*data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),]
        ],
        is_Prompt=True,
    )
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_training_config(config_name)
    data_config, dataset = create_dataset(config)
    print("*************************", config.total_configs[0].assets_dirs / data_config.repo_id)
    # assert 0 == 1
    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    batch_size = 4
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=16,
        shuffle=shuffle,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_frames//batch_size, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
