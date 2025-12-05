from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools
import torch
from moevla.models import tokenizer as _tokenizer
from moevla.shared import array_typing as at
from moevla.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats

import random

T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class InjectDatasetName(DataTransformFn):
    dataset_name: str | None

    def __call__(self, data: DataDict) -> DataDict:
        name = self.dataset_name.split("/")[-1] if "/" in self.dataset_name else self.dataset_name
        data["prompt"] = f"[dataset:{name}] {data['prompt']}"
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    data_mask: list
    norm_stats: at.PyTree[NormStats] | None
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None:
            new_stats = dict(self.norm_stats)  # create a shallow copy to avoid side effects
            if 'action' in new_stats:
                new_stats['actions'] = new_stats.pop('action')
            if 'observation.state' in new_stats:
                new_stats['state'] = new_stats.pop('observation.state')
            object.__setattr__(self, 'norm_stats', new_stats)

            if "actions" not in self.norm_stats or "state" not in self.norm_stats:
                raise ValueError("norm_stats must contain 'actions' and 'state' keys for normalization.")

        else:
            raise ValueError("norm_stats cannot be None. Please provide valid normalization statistics.")

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            raise ValueError("norm_stats cannot be None. Please provide valid normalization statistics.")
        
        return apply_tree(
            data,
            self.norm_stats,
            self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        x = np.clip(x, stats.q01, stats.q99)
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    data_mask: list
    norm_stats: at.PyTree[NormStats] | None

    def __post_init__(self):
        if self.norm_stats is not None:
            new_stats = dict(self.norm_stats)  # 浅拷贝，避免副作用
            if 'action' in new_stats:
                new_stats['actions'] = new_stats.pop('action')
            if 'observation.state' in new_stats:
                new_stats['state'] = new_stats.pop('observation.state')
            object.__setattr__(self, 'norm_stats', new_stats)

            if "actions" not in self.norm_stats or "state" not in self.norm_stats:
                raise ValueError("norm_stats must contain 'actions' and 'state' keys for normalization.")
        else:
            raise ValueError("norm_stats cannot be None. Please provide valid normalization statistics.")

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            raise ValueError("norm_stats cannot be None. Please provide valid normalization statistics.")
        
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize,
            strict=False,
        )

    def _unnormalize(self, x, stats: NormStats):        
        return x * (stats.std + 1e-6) + stats.mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01

class DropStateAndImage:
    def __init__(self, drop_state_ratio: float = 0.0, drop_images_ratio: float = 0.0):
        """
        Args:
            drop_state_ratio (float): the probability of dropping the state information.
            drop_images_ratio (float): the probability of dropping the images.
        """
        self.drop_state_ratio = drop_state_ratio
        self.drop_images_ratio = drop_images_ratio

    def __call__(self, inputs: dict):
        """
        Args:
            inputs (dict): :
                {
                    "image": {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"},
                    "image_mask": {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"},
                    "state": ndarray,
                    ... other key ...
                }
        Returns:
            dict: updated inputs
        """
        # create a copy of inputs to avoid modifying the original data
        outputs = dict(inputs)

        # ========== Drop Images ==========
        if "image" in inputs and "image_mask" in inputs:
            in_images = inputs["image"]
            in_masks = inputs["image_mask"]

            # drop head or gripper image, it would not drop both at the same time
            drop_head_image = False
            drop_gripper_image = False

            if self.drop_images_ratio > 0.0 and np.random.rand() < self.drop_images_ratio:
                # drop either head or gripper image
                if random.random() < 0.5:
                    drop_head_image = True
                else:
                    drop_gripper_image = True

            # drop head camera
            if drop_head_image and "base_0_rgb" in in_images:
                base_image = np.zeros_like(in_images["base_0_rgb"])
                base_image_mask = np.False_
            else:
                base_image = in_images.get("base_0_rgb", None)
                base_image_mask = np.True_

            # drop gripper cameras
            if drop_gripper_image:
                left_wrist_image = np.zeros_like(in_images.get("left_wrist_0_rgb", np.zeros(1)))
                right_wrist_image = np.zeros_like(in_images.get("right_wrist_0_rgb", np.zeros(1)))
                left_wrist_image_mask = np.False_
                right_wrist_image_mask = np.False_
            else:
                left_wrist_image = in_images.get("left_wrist_0_rgb", None)
                right_wrist_image = in_images.get("right_wrist_0_rgb", None)
                left_wrist_image_mask = np.True_
                right_wrist_image_mask = np.True_

            outputs["image"] = {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            }
            outputs["image_mask"] = {
                "base_0_rgb": base_image_mask,
                "left_wrist_0_rgb": left_wrist_image_mask,
                "right_wrist_0_rgb": right_wrist_image_mask,
            }

        # Drop State 
        if "state" in inputs and self.drop_state_ratio > 0.0 and np.random.rand() < self.drop_state_ratio:
            outputs["state"] = np.zeros_like(inputs["state"])

        return outputs

@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None
    data_mask: list

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            assert "state" in data, "Cannot apply DeltaActions transform without 'actions' or 'mask'"
            return data

        state, actions = data["state"], data["actions"]

        mask = np.asarray(self.mask)
        # half_len = len(self.data_mask)
        # data_mask is for state dim (2x action dim)
        # data_mask = np.array(self.data_mask + [0] * half_len, dtype=bool)

        actions -= np.expand_dims(np.where(mask, state, 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None
    data_mask: list
    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            assert "state" in data, "Cannot apply AbsoluteActions transform without 'actions' or 'mask'"
            return data
        # print("************AbsoluteActions")
        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        # half_len = len(self.data_mask)
        # data_mask is for state dim (2x action dim)
        # data_mask = np.array(self.data_mask + [0] * half_len, dtype=bool)

        actions+= np.expand_dims(np.where(mask, state, 0), axis=-2)
        data["actions"] = actions

        return data

@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}

@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int
    data_mask: list
    is_libero_or_oxe: bool = False
    def __call__(self, data: DataDict) -> DataDict:
        # state dim is 2x action dim. (state, datamask)
        data["state"] = pad_state_to_dim(data["state"], self.model_action_dim*2, self.data_mask, axis=-1, is_libero_or_oxe=self.is_libero_or_oxe)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, self.data_mask, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class UnpadStatesAndActions(DataTransformFn):
    """Removes zero-padding from states and actions to recover original dimensions."""

    model_action_dim: int
    data_mask: list
    is_libero_or_oxe: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if self.is_libero_or_oxe:
            # For libero and oxe dataset, we always unpad to 8 dim state
            data["state"] = data["state"][..., :8]
        else:
            data["state"] = unpad_state_to_dim(data["state"], self.model_action_dim*2, self.data_mask, axis=-1)
        mask = np.array(self.data_mask, dtype=bool)
        if "actions" in data:
            data["actions"] = data["actions"][..., mask]
        return data

def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")

def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, data_mask: list, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    assert target_dim == len(data_mask)
    mask = np.array(data_mask, dtype=bool)
    count_ones = np.sum(mask)
    
    if x.shape[axis] != count_ones:
        raise ValueError(
                f"The length of x along axis {axis} ({x.shape[axis]}) does not match the number of ones in data_mask ({count_ones})"
            )
    new_shape = x.shape[:-1] + (target_dim,)
    result = np.zeros(new_shape)
    result[..., mask] = x

    return result

def unpad_state_to_dim(x: np.ndarray, target_dim: int, data_mask: list, axis: int = -1) -> np.ndarray:
    """Unpad an array to recover the original dimension along the specified axis."""
    assert target_dim == len(data_mask) * 2
    mask = np.array(data_mask, dtype=bool)
    
    x = x[..., :target_dim//2][..., mask]

    return x

def pad_state_to_dim(x: np.ndarray, target_dim: int, data_mask: list, axis: int = -1, is_libero_or_oxe: bool = False) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    assert target_dim == len(data_mask) * 2
    mask = np.array(data_mask, dtype=bool)
    count_ones = np.sum(mask)
    
    if is_libero_or_oxe:
        return pad_libero_oxe_state_to_dim(x, target_dim, data_mask, axis)

    if x.shape[axis] != count_ones:
        raise ValueError(
            f"The length of x along axis {axis} ({x.shape[axis]}) does not match the number of ones in data_mask ({count_ones})"
        )
    # regular case
    new_shape = x.shape[:-1] + (target_dim,)
    result = np.zeros(new_shape, dtype=np.float32)

    # state dim is 2x action dim*(state, datamask)
    data_mask_x = data_mask + [0] * len(data_mask)
    mask_x = np.array(data_mask_x, dtype=bool)
    result[..., mask_x] = x

    data_mask_mask = [0] * len(data_mask) + data_mask
    mask_mask = np.array(data_mask_mask, dtype=bool)
    result[..., mask_mask] = 1

    return result


def pad_libero_oxe_state_to_dim(x: np.ndarray, target_dim: int, data_mask: list, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    
    new_shape = x.shape[:-1] + (target_dim,)
    result = np.zeros(new_shape, dtype=np.float32)

    result[..., :8] = x

    data_mask_mask = [0] * len(data_mask) + data_mask
    mask_mask = np.array(data_mask_mask, dtype=bool)
    result[..., mask_mask] = 1

    return result


def pad_calvin_state_to_dim(x: np.ndarray, target_dim: int, data_mask: list, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    # 将 data_mask 转换为布尔数组
    mask = np.array(data_mask, dtype=bool)
    if mask[0] == True:
        x = x[..., :7]
    else:
        x = x[..., 7:]

    count_ones = np.sum(mask)
    if x.shape[axis] != count_ones:
        raise ValueError(
                f"The length of x along axis {axis} ({x.shape[axis]}) does not match the number of ones in data_mask ({count_ones})"
            )

    new_shape = x.shape[:-1] + (target_dim,)
    result = np.zeros(new_shape, dtype=np.float32)

    data_mask_x = data_mask + [0] * len(data_mask)
    mask_x = np.array(data_mask_x, dtype=bool)
    result[..., mask_x] = x

    data_mask_mask = [0] * len(data_mask) + data_mask
    mask_mask = np.array(data_mask_mask, dtype=bool)
    result[..., mask_mask] = 1

    return result


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)
