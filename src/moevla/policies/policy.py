from collections.abc import Sequence
from typing import Any, TypeAlias

from openpi_client import base_policy as _base_policy
from typing_extensions import override

from moevla.models.model import preprocess_observation_and_to_device
from moevla import transforms as _transforms
from moevla.models.model import from_dict
from moevla.models.moevla import MoEVLA

from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
import time

BasePolicy: TypeAlias = _base_policy.BasePolicy
import torch
def to_tensor_batch(x):
    return torch.as_tensor(x).unsqueeze(0)  # Adds batch dimension

# 假设 obs 是一个嵌套字典结构
def tree_map(fn, tree):
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(fn, v) for v in tree)
    else:
        return fn(tree)
    
class Policy(BasePolicy):
    def __init__(
        self,
        model: MoEVLA,
        *,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        torch.manual_seed(42)
        self.model = model
        self.model.eval()  # Set the model to evaluation mode
        self._sample_actions = self.model.sample_actions
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(42)  # Set seed for reproducibility
        

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = tree_map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        observation = from_dict(inputs)
        inputs = tree_map(to_tensor_batch, inputs)
        observation = tree_map(to_tensor_batch, observation)

        observation = preprocess_observation_and_to_device(observation, train=False)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                actions = self._sample_actions(observation["images"], observation["image_masks"], observation["tokenized_prompt"], observation["tokenized_prompt_mask"], observation["state"], observation["data_mask"])
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }

        # Unbatch and convert to np.ndarray.
        outputs = tree_map(lambda x: x[0].cpu().numpy(), outputs)
        return self._output_transform(outputs)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


