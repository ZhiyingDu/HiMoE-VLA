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

def resize_with_pad(images, width, height, pad_value=0):
    # assume no-op when width height fits already
    has_batch_dim = images.ndimension() == 4
    if images.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {images.shape}")

    cur_height, cur_width = images.shape[1:3]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_images = F.interpolate(
        images, (images.shape[0], resized_height, resized_width, images.shape[3]), mode="bilinear", align_corners=False
    )

    # 填充图像到目标大小
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # pad on left and top of image
    padded_images = F.pad(resized_images, (pad_w0, pad_w1, pad_h0, pad_h1), mode='constant', value=pad_value)
    if not has_batch_dim:
        padded_images = padded_images.squeeze(0)  # 如果没有 batch 维度，移除它

    return padded_images


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
        image = observation["images"][key]
        if image.shape[1:3] != image_resolution:
            image = resize_with_pad(image, *image_resolution, pad_value=0)
            # image = image_tools.resize_with_pad(image, *image_resolution)

        image = image.permute(0, 3, 1, 2)

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
    
    if not set(image_keys).issubset(observation["images"]):
        image_list = list(observation["images"])
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {image_list}")

    batch_shape = observation["state"].shape[:-1]

    out_images = {}

    for key in image_keys:
        image = observation["images"][key]
        # print("image.shape[1:3] != image_resolution::", image.shape[1:3] != image_resolution)
        if image.shape[1:3] != image_resolution:
            image = resize_with_pad(image, *image_resolution, pad_value=0)

        image = image.permute(0, 3, 1, 2)

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
        state=observation["state"].to(device,dtype=dtype),
        data_mask=observation["data_mask"].to(device),
        tokenized_prompt=observation["tokenized_prompt"].to(device),
        tokenized_prompt_mask=observation["tokenized_prompt_mask"].to(device),
    )

