from typing import List, Optional, Union

import torch
from pytest import Cache
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING

from moevla.models.himoe import HiMoEModel

def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)

class MoE1Config():
    def __init__(self):
        # moe configure
        self.moe_intermediate_size = 4096
        self.n_routed_experts = 2
        self.hidden_size = 1024
        self.hidden_act = "silu"
        self.pretraining_tp = 1

class MoE2Config():
    def __init__(self):
        # moe configure
        self.num_experts_per_tok = 2  # "num_experts_per_tok": 6
        self.n_shared_experts = 1
        self.moe_intermediate_size = 1024
        self.n_routed_experts = 8
        self.hidden_size = 1024
        self.condition_dim = 16
        self.hidden_act = "silu"
        self.pretraining_tp = 1
        self.scoring_func = "softmax"
        self.aux_loss_alpha = 0.001
        self.seq_aux = True
        self.norm_topk_prob = True
        self.dtype = "bfloat16"


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}
    def __init__(
        self,
        paligemma_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        paligemma_id: str = "google/paligemma-3b-pt-224",
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        **kwargs,
    ):  
        self.paligemma_id = paligemma_id
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        

        if paligemma_config is None:
            # Default config from Pi0
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257216,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257216,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        elif isinstance(self.paligemma_config, dict):
            # Override Pi0 default config for PaliGemma
            if "model_type" not in gemma_expert_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        if gemma_expert_config is None:
            # Default config from Pi0
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        elif isinstance(self.gemma_expert_config, dict):
            # Override Pi0 default config for Gemma Expert
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )


class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig
    embed_dtype = torch.bfloat16

    def __init__(self, config: PaliGemmaWithExpertConfig):
        super().__init__(config=config)
        self.config = config
        model_id = "google/paligemma-3b-pt-224"
        self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            token="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",
        )

        self.gemma_expert = HiMoEModel(config=config.gemma_expert_config)

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

        if self.config.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.model.embed_tokens(tokens).to(dtype=self.embed_dtype)

    def forward(
        self,
        pre_attention_mask: Optional[torch.Tensor] = None,
        pre_position_ids: Optional[torch.LongTensor] = None,
        suf_attention_mask: Optional[torch.Tensor] = None,
        suf_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        prefix_embs: Optional[torch.FloatTensor] = None, 
        suffix_embs:Optional[torch.FloatTensor] = None, 
        data_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        for layer_idx in range(num_layers):
            if prefix_embs is not None:
                pre_layer_outputs, k_proj, v_proj = self.paligemma.language_model.model.layers[layer_idx](hidden_states=prefix_embs, attention_mask=pre_attention_mask, position_ids=pre_position_ids, use_cache=False)
                prefix_embs = pre_layer_outputs[0]

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "k_proj": k_proj,
                        "v_proj": v_proj,
                    }
                else:
                    k_proj = past_key_values[layer_idx]["k_proj"]
                    v_proj = past_key_values[layer_idx]["v_proj"]

            if suffix_embs is not None:
                suf_layer_outputs = self.gemma_expert.layers[layer_idx](hidden_states=suffix_embs, data_mask=data_mask, condition_embeds=[k_proj, v_proj], attention_mask=suf_attention_mask, position_ids=suf_position_ids, use_cache=False)
                suffix_embs = suf_layer_outputs[0]
        
        suf_outputs_embeds = None
        if suffix_embs is not None:
            suf_outputs_embeds = self.gemma_expert.norm(suffix_embs)

        return suf_outputs_embeds, past_key_values
