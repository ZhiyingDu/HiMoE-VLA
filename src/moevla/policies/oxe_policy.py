import dataclasses

import einops
import numpy as np

from moevla import transforms

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OxeInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int
    data_mask: list

    def __call__(self, data: dict) -> dict:

        # data["observation/state"] = np.zeros(7, dtype=np.float32)
        # state = transforms.pad_libero_oxe_state_to_dim(data["observation/state"], self.action_dim*2, self.data_mask)
        state = data["observation/state"]
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        if data.get("observation/image") is None:
            base_image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            base_image = _parse_image(data["observation/image"])
        if data.get("observation/wrist_image") is None:
            wrist_image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            wrist_image = _parse_image(data["observation/wrist_image"])
        if data.get("observation/wrist_image2") is None:
            wrist_image2 = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            wrist_image2 = _parse_image(data["observation/wrist_image2"])


        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": wrist_image2,
            },
            "image_mask": {
                "base_0_rgb": np.True_ if data.get("observation/image") is not None else np.False_,
                "left_wrist_0_rgb": np.True_ if data.get("observation/wrist_image") is not None else np.False_,
                "right_wrist_0_rgb": np.True_ if data.get("observation/wrist_image2") is not None else np.False_,
            },
            "data_mask": np.array(self.data_mask, dtype=bool)
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OxeOutputs(transforms.DataTransformFn):
    data_mask: list
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])}
