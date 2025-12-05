from collections.abc import Iterator, Sequence
from typing import Protocol, SupportsIndex, TypeVar
import pathlib
from typing import List

import moevla.models.model as _model
import moevla.training.config as _config
import moevla.transforms as _transforms


from lerobot.configs.policies import PreTrainedConfig
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import time

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: List[Sequence[_transforms.DataTransformFn]], is_Prompt=False):
        self.is_Prompt = is_Prompt
        self._dataset = dataset
        self._transform = []
        for transform in transforms:
            self._transform.append(_transforms.compose(transform))

    def __getitem__(self, index: SupportsIndex) -> T_co:
        dataset_idx = self._dataset[index]["dataset_index"]
        batch = self._transform[dataset_idx](self._dataset[index])
        if self.is_Prompt:
            return batch
        return _model.from_dict(batch), batch["actions"]

    def __len__(self) -> int:
        return len(self._dataset)


def create_dataset(data_config: List[_config.DataConfig], assets_dirs: List[pathlib.Path], model_config: List[PreTrainedConfig], data_weights: List[float]) -> Dataset:
    """Create a dataset for training."""
    start_time = time.time()
    repo_ids = []
    data_roots = []
    delta_timestamps = []
    local_files_only = []
    dataset_meta_tasks = []
    for i in range(len(data_config)):
        repo_id = data_config[i].repo_id
        repo_ids.append(repo_id)
        data_root = assets_dirs[i]/ repo_id
        data_roots.append(data_root)
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, data_root, local_files_only=data_config[i].local_files_only)
        dataset_meta_tasks.append(dataset_meta.tasks)
        delta_timestamp = {key: [t / dataset_meta.fps for t in range(model_config.n_action_steps)]
            for key in data_config[i].action_sequence_keys
        }
        delta_timestamps.append(delta_timestamp)
        local_files_only.append(data_config[i].local_files_only)

    dataset = lerobot_dataset.MultiLeRobotDataset(
        repo_ids,
        data_weights,
        data_roots, 
        delta_timestamps=delta_timestamps,
        local_files_only=local_files_only,
    )
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"***********Time of data load: {execution_time} 秒")
    num_frames = dataset.num_frames
    num_episodes = dataset.num_episodes

    # if data_config.prompt_from_task:
    PromptFromLeRobotTask = []
    for i in range(len(dataset_meta_tasks)):
        PromptFromLeRobotTask.append([_transforms.PromptFromLeRobotTask(dataset_meta_tasks[i])])
    dataset = TransformedDataset(dataset, PromptFromLeRobotTask, is_Prompt=True)

    return dataset, num_frames, num_episodes

def transform_dataset(dataset: Dataset, data_config: List[_config.DataConfig], *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""

    dataset_transform = []
    for i in range(len(data_config)):
        sub_transform = [
            *data_config[i].repack_transforms.inputs,
            *data_config[i].data_transforms.inputs,
            _transforms.Normalize(norm_stats=data_config[i].norm_stats, data_mask=data_config[i].data_mask),
            *data_config[i].model_transforms.inputs,
        ]
        dataset_transform.append(sub_transform)
    return TransformedDataset(dataset, dataset_transform, is_Prompt=False)


def create_data_loader(
    config: _config.TrainConfig,
    *,
    skip_norm_stats: bool = False,
):
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
    """
    sub_data_config = []
    sub_config_assets_dirs = []
    for subconfig in config.total_configs:
        print("The name of sub dataset:", subconfig.name)
        sub_data_config.append(subconfig.data.create(subconfig.assets_dirs, config.model))
        sub_config_assets_dirs.append(subconfig.assets_dirs)
    # data_config = config.data.create(config.assets_dirs, config.model)

    dataset, num_frames, num_episodes = create_dataset(sub_data_config, sub_config_assets_dirs, config.model, config.data_weights)
    dataset = transform_dataset(dataset, sub_data_config, skip_norm_stats=skip_norm_stats)

    return dataset, num_frames, num_episodes


