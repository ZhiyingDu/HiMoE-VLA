from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any
import os

import moevla.policies.policy as _policy
import moevla.shared.download as download
from moevla.training import config as _config
from moevla.models.moevla import MoEVLA
import moevla.transforms as transforms
from moevla.training.mixtures import DATASET_MIXTURES
import moevla.shared.normalize as _normalize

import torch

@dataclasses.dataclass
class PolicyConfig:
    model: MoEVLA
    norm_stats: dict[str, transforms.NormStats]

    input_layers: Sequence[transforms.DataTransformFn]
    output_layers: Sequence[transforms.DataTransformFn]
    default_prompt: str | None = None
    sample_kwargs: dict[str, Any] | None = None


def create_trained_policy(
    train_config: _config.TrainConfig,
    dataset_config: _config.DataConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    # check if dataset_config is in the train_config.total_configs
    total_configs = [] # List of dataset names used in the training config
    for mixed_datasets in train_config.config_name:
        datasets = DATASET_MIXTURES.get(mixed_datasets, None)
        if datasets is None:
            raise ValueError(f"Dataset mixture {mixed_datasets} not found in DATASET_MIXTURES.")
        for dataset in datasets:
            total_configs.append(dataset[0])
    if dataset_config.name not in total_configs:
        raise ValueError(
            f"Dataset config {dataset_config.name} not found in training config {train_config.config_name}. "
            f"Available datasets in training config: {total_configs}"
        )
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_config.model.create()
    # model.load_state_dict(load_file(checkpoint_dir), strict=True)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "pytorch_model.pth")), strict=True)
    model.to(device)
    
    data_config = dataset_config.data.create(dataset_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        
        norm_stats = _normalize.load(os.path.join(checkpoint_dir, data_config.asset_id))
        # print("epath.Path(train_config.assets_dirs), data_config.asset_id", epath.Path(train_config.assets_dirs), data_config.asset_id)
        # assert 0 == 1
        # norm_stats = train_config.data._load_norm_stats(epath.Path(train_config.assets_dirs), data_config.asset_id)
        if norm_stats is None:
            raise ValueError(
                f"Could not load norm stats from {os.path.join(checkpoint_dir, data_config.asset_id)}. "
                "Please make sure the norm stats are available in the assets directory."
            )

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_mask=data_config.data_mask, norm_stats=norm_stats),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(data_mask=data_config.data_mask, norm_stats=norm_stats),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
