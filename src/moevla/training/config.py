"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any, Protocol
import etils.epath as epath
from typing_extensions import override
import tyro
import moevla.models.moevla as moevla
import moevla.models.tokenizer as _tokenizer
import moevla.policies.aloha_policy as aloha_policy
import moevla.policies.oxe_policy as oxe_policy
import moevla.policies.libero_policy as libero_policy
import moevla.policies.calvin_policy as calvin_policy
import moevla.shared.normalize as _normalize
import moevla.transforms as _transforms
from moevla.training.mixtures import DATASET_MIXTURES
from lerobot.configs.policies import PreTrainedConfig
import argparse

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # data_mask
    data_mask:list | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `n_action_steps` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("action",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: PreTrainedConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: PreTrainedConfig, data_mask: list, is_libero_or_oxe: bool = False) -> _transforms.Group:
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.tokenizer_max_length),
                ),
                _transforms.PadStatesAndActions(model_config.max_action_dim, data_mask, is_libero_or_oxe=is_libero_or_oxe),
            ],
            outputs=[
                _transforms.UnpadStatesAndActions(model_config.max_action_dim, data_mask, is_libero_or_oxe=is_libero_or_oxe),
            ]
        )
    
@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # data_mask
    data_mask: list[int] = dataclasses.field(default_factory=list)

    # set to true if the data config is for Libero or OXE datasets
    is_libero_or_oxe: bool = False

    # drop_state_ratio
    drop_state_ratio: float = 0.0
    # drop_images_ratio
    drop_images_ratio: float = 0.0

    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            data_mask=self.data_mask,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(data_assets_dir)
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class Austin_Buds_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class Austin_Sailor_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )


        
        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Austin_Sirius_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Roboturk_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front_rgb",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # Prepare data for policy training
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Bc_Z_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Berkeley_Autolab_Ur5_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.hand_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Berkeley_Cable_Routing_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.wrist45_image",
                        # "observation/wrist_image2": "observation.images.wrist225_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Berkeley_Fanuc_Manipulation_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Berkeley_Mvp_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/wrist_image": "observation.images.hand_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Berkeley_Rpt_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/wrist_image": "observation.images.hand_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Bridge_Orig_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image_0",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Cmu_Play_Fusion_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Cmu_Stretch_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Dlr_Edan_Shared_Control_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )
        

        
        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Dobbe_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Droid_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.exterior_image_2_left",
                        # "observation/wrist_image": "observation.images.wrist_image_left",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Fmb_Dataset_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image_side_1",
                        # "observation/wrist_image": "observation.images.image_wrist_1",
                        # "observation/wrist_image2": "observation.images.image_wrist_2",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Fractal20220817_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Furniture_Bench_Dataset_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Iamlab_Cmu_Pickup_Insert_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Jaco_Play_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.image_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )



@dataclasses.dataclass(frozen=True)
class Language_Table_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.rgb",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Libero_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Kuka_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
    

@dataclasses.dataclass(frozen=True)
class Nyu_Franka_Play_Dataset_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )



@dataclasses.dataclass(frozen=True)
class Taco_Play_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.rgb_static",
                        # "observation/wrist_image": "observation.images.rgb_gripper",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Toto_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Ucsd_Kitchen_Dataset_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Stanford_Hydra_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Utaustin_Mutex_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        # "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )


        
        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Viola_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.agentview_rgb",
                        # "observation/wrist_image": "observation.images.eye_in_hand_rgb",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[oxe_policy.OxeInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[oxe_policy.OxeOutputs(data_mask=self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Aloha_DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.cam_high",
                        "observation/wrist_image": "observation.images.cam_left_wrist",
                        "observation/wrist_image2": "observation.images.cam_right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[aloha_policy.AlohaOutputs(data_mask=self.data_mask)],
        )

        # # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask, self.data_mask),
                    _transforms.DropStateAndImage(drop_state_ratio=self.drop_state_ratio, drop_images_ratio=self.drop_images_ratio)
            ],
            outputs=[_transforms.AbsoluteActions(delta_action_mask, self.data_mask)],
        )
        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class Aloha_Sim_DataConfig(DataConfigFactory):
    default_prompt: str | None = None
    adapt_to_pi: bool = True
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.top",
                        "observation/state": "observation.state",
                        "actions": "action",
                        # "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(data_mask=self.data_mask, adapt_to_pi=self.adapt_to_pi)],
        )

        # # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask, self.data_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask, self.data_mask)],
        )

        # #  add dataset name
        # data_transforms = data_transforms.push(
        #     inputs=[_transforms.InjectDatasetName(self.repo_id)],
        # )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[
                libero_policy.LiberoInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask),
                _transforms.DropStateAndImage(drop_state_ratio=self.drop_state_ratio, drop_images_ratio=self.drop_images_ratio)
            ],
            outputs=[libero_policy.LiberoOutputs(data_mask=self.data_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoSubDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[libero_policy.LiberoOutputs(data_mask=self.data_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotCalvinEEFFullDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[calvin_policy.CalvinInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[calvin_policy.CalvinOutputs(data_mask=self.data_mask)],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask, self.data_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask, self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )    

@dataclasses.dataclass(frozen=True)
class LeRobotCalvinJointDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[calvin_policy.CalvinInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[calvin_policy.CalvinOutputs(data_mask=self.data_mask)],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask, self.data_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask, self.data_mask)],
        )



        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotXARMFullDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: PreTrainedConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[calvin_policy.CalvinInputs(action_dim=model_config.max_action_dim, data_mask=self.data_mask)],
            outputs=[calvin_policy.CalvinOutputs(data_mask=self.data_mask)],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask, self.data_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask, self.data_mask)],
        )


        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config, self.data_mask, self.is_libero_or_oxe)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )  


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./data"

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return pathlib.Path(self.assets_base_dir).resolve()

# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    DatasetConfig(
        name="fractal20220817_data_lerobot",
        data=Fractal20220817_DataConfig(
            repo_id="oxe_lerobot/fractal20220817_data_lerobot",
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="austin_buds_dataset_lerobot",
        data=Austin_Buds_DataConfig(
            repo_id="oxe_lerobot/austin_buds_dataset_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="bridge_orig_lerobot",
        data=Bridge_Orig_DataConfig(
            repo_id="oxe_lerobot/bridge_orig_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="austin_sailor_dataset_lerobot",
        data=Austin_Sailor_DataConfig(
            repo_id="oxe_lerobot/austin_sailor_dataset_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="cmu_play_fusion_lerobot",
        data=Cmu_Play_Fusion_DataConfig(
            repo_id="oxe_lerobot/cmu_play_fusion_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="iamlab_cmu_pickup_insert_lerobot",
        data=Iamlab_Cmu_Pickup_Insert_DataConfig(
            repo_id="oxe_lerobot/iamlab_cmu_pickup_insert_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="nyu_franka_play_dataset_lerobot",
        data=Nyu_Franka_Play_Dataset_DataConfig(
            repo_id="oxe_lerobot/nyu_franka_play_dataset_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="austin_sirius_dataset_lerobot",
        data=Austin_Sirius_DataConfig(
            repo_id="oxe_lerobot/austin_sirius_dataset_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="cmu_stretch_lerobot",
        data=Cmu_Stretch_DataConfig(
            repo_id="oxe_lerobot/cmu_stretch_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="jaco_play_lerobot",
        data=Jaco_Play_DataConfig(
            repo_id="oxe_lerobot/jaco_play_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="bc_z_lerobot",
        data=Bc_Z_DataConfig(
            repo_id="oxe_lerobot/bc_z_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="dlr_edan_shared_control_lerobot",
        data=Dlr_Edan_Shared_Control_DataConfig(
            repo_id="oxe_lerobot/dlr_edan_shared_control_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="kuka_lerobot",
        data=Kuka_DataConfig(
            repo_id="oxe_lerobot/kuka_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="stanford_hydra_dataset_lerobot",
        data=Stanford_Hydra_DataConfig(
            repo_id="oxe_lerobot/stanford_hydra_dataset_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="berkeley_autolab_ur5_lerobot",
        data=Berkeley_Autolab_Ur5_DataConfig(
            repo_id="oxe_lerobot/berkeley_autolab_ur5_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="dobbe_lerobot",
        data=Dobbe_DataConfig(
            repo_id="oxe_lerobot/dobbe_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="language_table_lerobot",
        data=Language_Table_DataConfig(
            repo_id="oxe_lerobot/language_table_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="taco_play_lerobot",
        data=Taco_Play_DataConfig(
            repo_id="oxe_lerobot/taco_play_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="berkeley_cable_routing_lerobot",
        data=Berkeley_Cable_Routing_DataConfig(
            repo_id="oxe_lerobot/berkeley_cable_routing_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="roboturk_lerobot",
        data=Roboturk_DataConfig(
            repo_id="oxe_lerobot/roboturk_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="droid_lerobot",
        data=Droid_DataConfig(
            repo_id="oxe_lerobot/droid_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="toto_lerobot",
        data=Toto_DataConfig(
            repo_id="oxe_lerobot/toto_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="berkeley_fanuc_manipulation_lerobot",
        data=Berkeley_Fanuc_Manipulation_DataConfig(
            repo_id="oxe_lerobot/berkeley_fanuc_manipulation_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="fmb_dataset_lerobot",
        data=Fmb_Dataset_DataConfig(
            repo_id="oxe_lerobot/fmb_dataset_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="ucsd_kitchen_dataset_lerobot",
        data=Ucsd_Kitchen_Dataset_DataConfig(
            repo_id="oxe_lerobot/ucsd_kitchen_dataset_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="berkeley_mvp_lerobot",
        data=Berkeley_Mvp_DataConfig(
            repo_id="oxe_lerobot/berkeley_mvp_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="utaustin_mutex_lerobot",
        data=Utaustin_Mutex_DataConfig(
            repo_id="oxe_lerobot/utaustin_mutex_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="berkeley_rpt_lerobot",
        data=Berkeley_Rpt_DataConfig(
            repo_id="oxe_lerobot/berkeley_rpt_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="furniture_bench_dataset_lerobot",
        data=Furniture_Bench_Dataset_DataConfig(
            repo_id="oxe_lerobot/furniture_bench_dataset_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="viola_lerobot",
        data=Viola_DataConfig(
            repo_id="oxe_lerobot/viola_lerobot",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),

    # Aloha datasets
    DatasetConfig(
        name="aloha_mobile_chair",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_mobile_chair",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_mobile_elevator",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_mobile_elevator",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_mobile_shrimp",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_mobile_shrimp",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_mobile_wash_pan",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_mobile_wash_pan",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_mobile_wipe_wine",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_mobile_wipe_wine",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_static_candy",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_static_candy",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_static_coffee_new",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_static_coffee_new",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_static_cups_open",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_static_cups_open",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_static_pingpong_test",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_static_pingpong_test",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_static_pro_pencil",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_static_pro_pencil",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_static_screw_driver",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_static_screw_driver",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_static_tape",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_static_tape",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_static_towel",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_static_towel",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_static_vinh_cup",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_static_vinh_cup",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_static_ziploc_slide",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_static_ziploc_slide",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_pen_uncap_diverse",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_pen_uncap_diverse",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_mobile_rdt",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_mobile_rdt",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_static_rdt",
        data=Aloha_DataConfig(
            repo_id="aloha/aloha_static_rdt",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="aloha_moevla",
        data=Aloha_DataConfig(
            repo_id="aloha_moevla",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
            ),
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1
        ),
    ),
    DatasetConfig(
        name="xarm_eef",
        data=LeRobotXARMFullDataConfig(
            repo_id="xArm_fvl",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
                action_sequence_keys=("actions",)
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
        ),
        assets_base_dir="/mnt/blob/xArm"
    ),
    DatasetConfig(
        name="calvin_abc_eef",
        data=LeRobotCalvinEEFFullDataConfig(
            repo_id="calvin_abc_eef",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
                action_sequence_keys=("actions",)
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
        ),
    ),
    DatasetConfig(
        name="calvin_d_joint",
        data=LeRobotCalvinJointDataConfig(
            repo_id="calvin_d_joint",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
                action_sequence_keys=("actions",)
            ),
            data_mask = [0] * 8 + [1] * 8 + [0] * 8,
        ),
    ),
    DatasetConfig(
        name="calvin_joint",
        data=LeRobotCalvinJointDataConfig(
            repo_id="calvin_joint",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
                action_sequence_keys=("actions",)
            ),
            data_mask = [0] * 8 + [1] * 8 + [0] * 8,
        ),
    ),
    DatasetConfig(
        name="libero_10_no_noops_lerobot",
        data=LeRobotLiberoDataConfig(
            repo_id="libero_10_no_noops",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
                action_sequence_keys=("actions",)
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="libero_goal_no_noops_lerobot",
        data=LeRobotLiberoDataConfig(
            repo_id="libero_goal_no_noops",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
                action_sequence_keys=("actions",)
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="libero_object_no_noops_lerobot",
        data=LeRobotLiberoDataConfig(
            repo_id="libero_object_no_noops",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
                action_sequence_keys=("actions",)
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="libero_spatial_no_noops_lerobot",
        data=LeRobotLiberoDataConfig(
            repo_id="libero_spatial_no_noops",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
                action_sequence_keys=("actions",)
            ),
            data_mask = [1] * 7 + [0] * 9 + [0] * 8,
            is_libero_or_oxe=True,
        ),
    ),
    DatasetConfig(
        name="aloha_sim_transfer_cube_human",
        data=Aloha_Sim_DataConfig(
            repo_id="aloha/aloha_sim_transfer_cube_human",
            base_config=DataConfig(
                local_files_only=True,  
                # prompt_from_task=True,
                # action_sequence_keys=("actions",),
            ),
            default_prompt="Transfer cube",
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1,
        ),
    ),
    DatasetConfig(
        name="aloha_sim_insertion_human",
        data=Aloha_Sim_DataConfig(
            repo_id="aloha/aloha_sim_insertion_human",
            base_config=DataConfig(
                local_files_only=True,  
                # prompt_from_task=True,
                # action_sequence_keys=("actions",),
            ),
            default_prompt="aloha insertion",
            data_mask = [0] * 8 + [1] * 7 + [0] * 1 + [1] * 7 + [0] * 1,
        ),
    ),
    DatasetConfig(
        name="agibot_sim_joint",
        data=Aloha_DataConfig(
            repo_id="agibot_sim_joint",
            base_config=DataConfig(
                local_files_only=True,  
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            data_mask = [0] * 8 + [1] * 8 + [1] * 8,
        ),
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def get_dataset_config(config_name: str) -> DatasetConfig:
    """Get a config by name."""
    # print("Getting dataset config:", config_name)
    if config_name in _CONFIGS_DICT:
        return _CONFIGS_DICT[config_name]
    else:
        raise ValueError(f"Config {config_name} not found.")
        assert 0 == 1

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]

    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # data_config
    config_name: list[str] = dataclasses.field(default_factory=list)
    data_weights: list[float] = dataclasses.field(default_factory=list)
    total_configs: list[dict] = dataclasses.field(init=False)

    # Defines the model config. Some attributes (action_dim, n_action_steps, and tokenizer_max_length) are shared by all models
    # define additional attributes.
    model: PreTrainedConfig = dataclasses.field(default_factory=moevla.MoEVLAConfig)

    # training related
    resume: bool = False
    
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 4

    # gradient_accumulation_steps
    gradient_accumulation_steps: int = 1
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 16
    # Number of train steps (batches) to run.
    num_train_steps: int = 90_000
    epochs: int = 1000
    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None
    # enable_gradient_checkpointing
    enable_gradient_checkpointing: bool = True
    # mixed_precision
    mixed_precision: str = 'bf16'

    # checkpoints_total_limit
    checkpoints_total_limit: int = 40
    
    # logs
    logging_dir: str = 'logs'

    # deepspeed dir
    deepspeed: str = 'src/moevla/training/zero2.json'

    # pretrained_model_name_or_path
    pretrained_model_name_or_path: str = None

    # optimizer and scheuler
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # max_grad_norm
    max_grad_norm: float = 1.0
    
    # set_grads_to_none
    set_grads_to_none: bool = True

    # checkpointing_period
    checkpointing_period: int = 5000

    # report_to
    report_to: str = 'wandb' 

    # Finetuning settings
    freeze_vision_encoder: bool = False

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()
        # return pathlib.Path("/mnt/blob/task/finetune_calvin_eef_abcd/checkpoints")

    def __post_init__(self) -> None:
        total_configs = []
        data_weights = []
        for mixed_datasets in self.config_name:
            datasets = DATASET_MIXTURES.get(mixed_datasets, None)
            if datasets is None:
                raise ValueError(f"Dataset mixture {mixed_datasets} not found in DATASET_MIXTURES.")
            for dataset in datasets:
                total_configs.append(get_dataset_config(dataset[0]))
                data_weights.append(dataset[1])
        object.__setattr__(self, "total_configs", total_configs)
        object.__setattr__(self, "data_weights", data_weights)
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


_TrainConfigs = [
    TrainConfig(
        name="oxe_aloha_freeze_vision_encoder",
        model=moevla.MoEVLAConfig(n_action_steps=10),
        config_name = ["oxe_magic_soup", "aloha_open_source"],
        # optimizer_lr=1e-5,
        optimizer_weight_decay=1e-4,
        freeze_vision_encoder=True,
        num_train_steps=100_000,
        batch_size=16,
        checkpoint_base_dir="./pretrain"
    ),
    TrainConfig(
        name="calvin_d_joint",
        model=moevla.MoEVLAConfig(n_action_steps=10),
        config_name = ["calvin_d_joint"],
        num_train_steps=50_000,
        batch_size=4,
        pretrained_model_name_or_path="./pretrain/pytorch_model.pth",
        checkpoint_base_dir="./checkpoints"
    ),
    # libero 10 no noops finetune
    TrainConfig(
        name="libero_10_no_noops_lerobot_finetune",
        model=moevla.MoEVLAConfig(n_action_steps=10),
        config_name = ["libero_10"],
        num_train_steps=50_000,
        batch_size=8,
        pretrained_model_name_or_path="./pretrain/pytorch_model.pth",
        checkpoint_base_dir="./checkpoints"
    ),
    # libero goal no noops finetune
    TrainConfig(
        name="libero_goal_no_noops_lerobot_finetune",
        model=moevla.MoEVLAConfig(n_action_steps=10),
        config_name = ["libero_goal"],
        num_train_steps=50_000,
        batch_size=8,
        pretrained_model_name_or_path="./pretrain/pytorch_model.pth",
        checkpoint_base_dir="./checkpoints"
    ),
    TrainConfig(
        name="libero_object_no_noops_lerobot_finetune",
        model=moevla.MoEVLAConfig(n_action_steps=10),
        config_name = ["libero_object"],
        num_train_steps=50_000,
        batch_size=8,
        pretrained_model_name_or_path="./pretrain/pytorch_model.pth",
        checkpoint_base_dir="./checkpoints"
    ),
    TrainConfig(
        name="libero_spatial_no_noops_lerobot_finetune",
        model=moevla.MoEVLAConfig(n_action_steps=10),
        config_name = ["libero_spatial"],
        num_train_steps=50_000,
        batch_size=4,
        pretrained_model_name_or_path="./pretrain/pytorch_model.pth",
        checkpoint_base_dir="./checkpoints"
    ),
]
_TrainConfigs_Dict = {config.name: config for config in _TrainConfigs}

def get_training_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name in _TrainConfigs_Dict:
        return _TrainConfigs_Dict[config_name]
    else:
        assert 0 == 1

def cli() -> DatasetConfig:
    parser = argparse.ArgumentParser(
        description="Select a config and override certain fields in DatasetConfig"
    )
    # Choose a preset config name (must exist in _CONFIGS_DICT)
    parser.add_argument(
        "--deepspeed",
        type=str,
        required=True,
        help="Path to DeepSpeed config"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=list(_TrainConfigs_Dict.keys()),
        help="Config name"
    )

    # 
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment name"
    )

    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="If set, override the checkpoint directory"
    )

    # Add more arguments here if you need to override additional fields

    args = parser.parse_args()

    # Retrieve the DatasetConfig from the preset config dictionary based on the config name
    config = get_training_config(args.config)

    # Override exp_name and overwrite fields in the config using CLI arguments
    config = dataclasses.replace(config, deepspeed=args.deepspeed, exp_name=args.exp_name, resume=args.resume)

    return config
