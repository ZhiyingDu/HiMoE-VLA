from collections.abc import Sequence
import enum
import numpy as np
import torch
from typing import Dict
import torch.nn.functional as F  # noqa: N812
from torchvision import transforms


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)

def from_dict(data: Dict[str, np.ndarray]):
    """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
    # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
    if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
        raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
    # If images are uint8, convert them to [-1, 1] float32.
    for key in data["image"]:
        if data["image"][key].dtype == np.uint8:
            data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
    return {
        "images": data["image"],
        "image_masks": data["image_mask"],
        "state": data["state"],
        "data_mask": data["data_mask"],
        "tokenized_prompt": data.get("tokenized_prompt"),
        "tokenized_prompt_mask": data.get("tokenized_prompt_mask"),
    }


def preprocess_observation(
    observation: dict,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """
    
    if not set(image_keys).issubset(observation["images"]):
        image_list = list(observation["images"])
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {image_list}")

    batch_shape = observation["state"].shape[:-1]

    out_images = {}

    for key in image_keys:
        image = observation["images"][key].permute(0, 3, 1, 2)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5
            height, width = image.shape[2:4]
            transforms_list = []
            if "wrist" not in key:
                transforms_list += [
                    transforms.RandomCrop((int(height * 0.95), int(width * 0.95))),  
                    transforms.Resize((height, width)),  
                    transforms.RandomRotation(degrees=(-5, 5)),  
                ]
            transforms_list += [
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            transform_chain = transforms.Compose(transforms_list)


            image = transform_chain(image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    out_masks = {}
    for key in out_images.keys():
        if key not in observation["image_masks"]:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool)
        else:
            out_masks[key] = torch.as_tensor(observation["image_masks"][key])

    return dict(
        images=out_images,
        image_masks=out_masks,
        state=observation["state"].to(dtype=torch.float32),
        data_mask=observation["data_mask"],
        tokenized_prompt=observation["tokenized_prompt"],
        tokenized_prompt_mask=observation["tokenized_prompt_mask"],
    )


def to_device_recursive(data, device, dtype=None):
    if isinstance(data, torch.Tensor):
        if dtype is None:
            return data.to(device, non_blocking=True)
        else:
            return data.to(device, dtype=dtype, non_blocking=True)

    elif isinstance(data, dict):
        return {k: to_device_recursive(v, device, dtype) for k, v in data.items()}

    elif isinstance(data, list):
        return [to_device_recursive(v, device, dtype) for v in data]

    elif isinstance(data, tuple):
        return tuple(to_device_recursive(v, device, dtype) for v in data)

    else:
        return data

def preprocess_observation_and_to_device(
    observation: dict,
    *,
    train: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.bfloat16,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    observation = to_device_recursive(observation, device)
    if not set(image_keys).issubset(observation["images"]):
        image_list = list(observation["images"])
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {image_list}")

    batch_shape = observation["state"].shape[:-1]

    out_images = {}

    for key in image_keys:
        image = observation["images"][key].permute(0, 3, 1, 2)

        if train:
            image = image / 2.0 + 0.5
            height, width = image.shape[2:4]
            transforms_list = []
            if "wrist" not in key:
                transforms_list += [
                    transforms.RandomCrop((int(height * 0.95), int(width * 0.95))),  
                    transforms.Resize((height, width)),  
                    transforms.RandomRotation(degrees=(-5, 5)),  
                ]
            transforms_list += [
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            transform_chain = transforms.Compose(transforms_list)

            image = transform_chain(image)
            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image.to(device,dtype=dtype)

    out_masks = {}
    for key in out_images.keys():
        if key not in observation["image_masks"]:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool).to(device)
        else:
            out_masks[key] = torch.as_tensor(observation["image_masks"][key]).to(device)

    return dict(
        images=out_images,
        image_masks=out_masks,
        state=observation["state"].to(dtype=dtype),
        data_mask=observation["data_mask"],
        tokenized_prompt=observation["tokenized_prompt"],
        tokenized_prompt_mask=observation["tokenized_prompt_mask"],
    )

