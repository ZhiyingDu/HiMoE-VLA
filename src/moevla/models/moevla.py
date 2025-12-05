from dataclasses import dataclass
import logging
import math
from pathlib import Path
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from .paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
import draccus

from typing_extensions import override

from moevla.models import model as _model
from moevla.shared import array_typing as at
from moevla.models.hub_mixin import CompatiblePyTorchModelHubMixin

from lerobot.configs.policies import PreTrainedConfig

logger = logging.getLogger("moevla")

def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)

def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = time.dtype
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb

def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

    [[1 1 1 1 1 1]]: pure causal attention.

    [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
    themselves and the last 3 tokens have a causal attention. The first
    entry could also be a 1 without changing behaviour.

    [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
    block can attend all previous blocks and all tokens on the same block.

    Args:
        input_mask: bool[B, N] true if its part of the input, false if padding.
        mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


@PreTrainedConfig.register_subclass("pi0")
@dataclass
class MoEVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    #  model_id
    paligemma_dir: str = "/mnt/blob/checkpoints/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c/"
    # paligemma_dir: str = "/home/azureuser/v-zhiyingdu/Code/pi_deepspeed/checkpoints/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c/"
    # Shorter state and action vectors will be padded
    max_action_dim: int = 24

    # Tokenizer
    tokenizer_max_length: int = 96

    # Projector
    proj_width: int = 1024

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # or fa2, flex

    # Finetuning settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_state_proj: bool = True


    def __init__(self, n_action_steps: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.n_action_steps = n_action_steps

    @property
    @override
    def model_type(self):
        return None
    
    def create(self) -> "MoEVLA":
        return MoEVLA(self)
    
    def __post_init__(self):
        super().__post_init__()

        # TODO(Steven): Validate device and amp? in all policy configs?
        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / "config.json", "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None
    
    def validate_features(self):
        print("Validating features...")  

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    

class MoEVLA(nn.Module, CompatiblePyTorchModelHubMixin):

    def __init__(self, config: MoEVLAConfig):
        super().__init__()
        self.config = config

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            paligemma_id=self.config.paligemma_dir,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_export_config)

        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_action_dim*2, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj
        if self.config.freeze_vision_encoder:
            self.paligemma_with_expert.paligemma.vision_tower.eval()
            for param in self.paligemma_with_expert.paligemma.vision_tower.parameters():
                param.requires_grad = False
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for param in self.paligemma_with_expert.paligemma.parameters():
                param.requires_grad = False

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    @at.typecheck
    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images.values(), img_masks.values(), strict=False):

            img_emb = self.paligemma_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, data_mask, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        state = state.to(dtype=torch.float32)
        actions = actions.to(dtype=torch.float32)
        
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        ## datamask
        # data_mask = (actions[:, 0:1, actions.shape[-1] // 2:] > 0.5).expand(-1, suffix_embs.shape[1], -1).contiguous()

        # Concatenate prefix and suffix masks
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # Create 2D attention masks
        pre_att_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        suf_att_2d_masks = make_att_2d_masks(pad_masks, att_masks)[:, prefix_pad_masks.shape[1]:, :]
        
        # Create position ids
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        pre_position_ids = position_ids[:, :prefix_pad_masks.shape[1]]
        suf_position_ids = position_ids[:, prefix_pad_masks.shape[1]:]
        
        # forward pass through PaliGemma with Expert
        suffix_out, _ = self.paligemma_with_expert(
            pre_attention_mask=pre_att_masks,
            pre_position_ids=pre_position_ids,
            suf_attention_mask=suf_att_2d_masks,
            suf_position_ids=suf_position_ids,
            past_key_values=None,
            prefix_embs=prefix_embs,
            suffix_embs=suffix_embs,
            data_mask=data_mask,
            use_cache=False,
            fill_kv_cache=False,
        )

        # Extract the last n_action_steps from the suffix output
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        # losses = F.mse_loss(u_t, v_t, reduction="mean")
        loss = F.mse_loss(u_t, v_t, reduction="none")  # shape: (B, N, D)
        mask = data_mask[:, None, :].expand(-1, u_t.shape[1], -1).float()  # shape: (B, 1, D) -> broadcast to (B, N, D)
        masked_loss = loss * mask  # shape: (B, N, D)
        final_loss = masked_loss.sum() / (mask.sum() + 1e-8)

        return final_loss

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, data_mask, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        state = state.to(dtype=torch.float32)
        
        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        pre_att_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pre_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # data_mask = (state[:, state.shape[-1] // 2:] > 0.5).expand(self.config.n_obs_steps+self.config.n_action_steps, -1).contiguous()

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            pre_attention_mask=pre_att_masks,
            pre_position_ids=pre_position_ids,
            suf_attention_mask=None,
            suf_position_ids=None,
            past_key_values=None,
            prefix_embs=prefix_embs,
            suffix_embs=None,
            data_mask=None,
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                prefix_att_masks,
                past_key_values,
                data_mask,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        prefix_att_masks,
        past_key_values,
        data_mask,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        suf_att_2d_masks = make_att_2d_masks(pad_masks, att_masks)[:, prefix_pad_masks.shape[1]:, :]

        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        suf_position_ids = position_ids[:, prefix_pad_masks.shape[1]:]

        suffix_out, _ = self.paligemma_with_expert.forward(
            pre_attention_mask=None,
            pre_position_ids=None,
            suf_attention_mask=suf_att_2d_masks,
            suf_position_ids=suf_position_ids,
            past_key_values=past_key_values,
            prefix_embs=None,
            suffix_embs=suffix_embs,
            data_mask=data_mask,
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        # suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
