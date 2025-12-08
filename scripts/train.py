# **********************
import logging
import os
from pathlib import Path
from functools import partial
import wandb
wandb.login(key="")

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState

from tqdm.auto import tqdm
import multiprocessing

from moevla.models.moevla import MoEVLA
from moevla.models.model import preprocess_observation
from moevla.training.utils import format_big_number
from moevla.training.config import TrainConfig, cli
from moevla.training.data_loader import create_data_loader
from moevla.training.utils import build_cosine_decay_schedule_with_wramup
import moevla.shared.normalize as _normalize

import transformers
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

def main(config: TrainConfig):
    logger = get_logger(__name__)
    logging_dir = Path(config.checkpoint_dir, config.logging_dir)
    accelerator_project_config = ProjectConfiguration(total_limit=config.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=config.deepspeed
        ) if config.deepspeed is not None else None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
        step_scheduler_with_optimizer=False, # set to False to use the scheduler step in the training loop, else it will step the scheduler automatically after each optimizer step
    )
    accelerator.init_trackers(project_name = config.exp_name)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if config.checkpoint_dir is not None:
            os.makedirs(config.checkpoint_dir, exist_ok=True)

        # if config.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=config.hub_model_id or Path(config.checkpoint_dir).name, exist_ok=True, token=config.hub_token
        #     ).repo_id
    

    # define model
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model = MoEVLA(config.model)
    num_total_params = sum(p.numel() for p in model.parameters())
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if config.pretrained_model_name_or_path is not None:
        logger.info("Constructing model from pretrained checkpoint.")
        model_file = config.pretrained_model_name_or_path
        state_dict = torch.load(model_file, map_location='cpu')

        state_dict = {k: v for k, v in state_dict.items() if "gate" not in k or "experts" not in k} 
        # assert 0 == 1
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("missing:", missing, " unexpected:", unexpected)
        del state_dict
        torch.cuda.empty_cache()
    

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # which ensure saving model in huggingface format (config.json + pytorch_model.bin)
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model  # type: ignore
                if isinstance(model_to_save, type(accelerator.unwrap_model(model))):
                    model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    if config.enable_gradient_checkpointing:
        non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        def check_fn(submodule: nn.Module) -> bool:
            return isinstance(submodule, GemmaDecoderLayer)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
    

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    optimizer_class = torch.optim.AdamW

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
        if "layernorm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    # Build Parameter Groups
    groups = [
        {"params": decay, "weight_decay": config.optimizer_weight_decay}, 
        {"params": no_decay, "weight_decay": 0.0}]
    
    # Optimizer creation
    optimizer = optimizer_class(
        groups,
        lr=config.optimizer_lr,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
    )
    lr_scheduler = build_cosine_decay_schedule_with_wramup(
                        optimizer, 
                        peak_lr=config.optimizer_lr, 
                        decay_lr=config.scheduler_decay_lr,
                        num_warmup_steps=config.scheduler_warmup_steps,
                        num_decay_steps=config.scheduler_decay_steps
                    )

    # prepare dataset
    dataset, num_frames, num_episodes = create_data_loader(config)
    
    logger.info("***** using sampler *****")
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        drop_last=True,
        seed=config.seed,
    )
    mp_context = None
    if config.num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
    )

    # Ensure 'train_micro_batch_size_per_gpu' is explicitly set to avoid DataLoader batch size being None,
    # which can cause training errors in DeepSpeed.
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.batch_size

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, data_loader, lr_scheduler           
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        config_dict = {k: v for k, v in vars(config).items() if k != 'total_configs'}
        accelerator.init_trackers("HiMoE_VLA", config=config_dict)

    # Train!
    total_batch_size = config.batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num episodes each epoch = {num_episodes}")
    logger.info(f"  Num frames each epoch = {num_frames}")
    logger.info(f"  Num train steps= ({config.num_train_steps})")

    logger.info(f"  Instantaneous batch size per device = {config.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")

    logger.info(f"  Num total params = ({format_big_number(num_total_params)})")
    logger.info(f"  Num learnable params= ({format_big_number(num_learnable_params)})")

    resume_global_step = 0
    # Potentially load in the weights and states from a previous save
    resume_from_checkpoint = config.checkpoint_dir
    if resume_from_checkpoint:
        # Get the mos recent checkpoint
        torch.cuda.empty_cache()
        dirs = os.listdir(config.checkpoint_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(config.checkpoint_dir, path, "accelerator")) # load_module_strict=False
            except:
                # load deepspeed's state_dict
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(os.path.join(config.checkpoint_dir, path, "accelerator", "pytorch_model", "mp_rank_00_model_states.pt"))
                model.module.load_state_dict(checkpoint["module"])
                del checkpoint
                
            resume_global_step = int(path.split("-")[1]) 
            first_epoch = resume_global_step // len(train_dataloader)

        torch.cuda.empty_cache()
    
    global_step = resume_global_step
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, config.num_train_steps), initial=global_step, total=config.num_train_steps, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    loss_for_log = {}
    device_type = "cuda" if "cuda" in str(accelerator.device) else "cpu"
    average_loss = 0.0
    model.train()
    while global_step < config.num_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                observation = batch[0]
                actions = batch[1]
                observation = preprocess_observation(observation, train=True)
                with torch.autocast(device_type=device_type, dtype=weight_dtype):
                    loss = model(
                        observation["images"], 
                        observation["image_masks"], 
                        observation["tokenized_prompt"], 
                        observation["tokenized_prompt_mask"], 
                        observation["state"], 
                        actions,
                        observation["data_mask"]
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=config.set_grads_to_none)
                average_loss += loss.detach().item()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # print(f"[DEBUG] global_step={global_step}, scheduler_step={lr_scheduler.scheduler.last_epoch}, lr={lr_scheduler.get_last_lr()[0]}")
                progress_bar.update(1)
                global_step += 1

                if global_step % config.checkpointing_period == 0:
                    save_path = os.path.join(config.checkpoint_dir, f"checkpoint-{global_step}")
                    torch.cuda.empty_cache()
                    if accelerator.is_main_process:
                        os.makedirs(save_path, exist_ok=True)
                        unwarp_model = accelerator.unwrap_model(model)
                        torch.save(unwarp_model.state_dict(), os.path.join(save_path, 'pytorch_model.pth'))
                        # save norm stats
                        for subconfig in config.total_configs:
                            data_config = subconfig.data
                            norm_stats = data_config._load_norm_stats(subconfig.assets_dirs, data_config.repo_id)
                            if norm_stats is not None:
                                _normalize.save(os.path.join(save_path, data_config.repo_id), norm_stats)

                    accelerator.save_state(os.path.join(save_path, "accelerator"))
                    logger.info(f"Saved state to {save_path}")
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            logs.update(loss_for_log)
            if global_step % 50 == 0:
                logs = {"loss": average_loss/(50*config.gradient_accumulation_steps), "lr": lr_scheduler.get_last_lr()[0]}
                # logger.info(logs)
                accelerator.log(logs, step=global_step)
                average_loss = 0.0

            if global_step >= config.num_train_steps:
                break        

    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    accelerator.save_state(os.path.join(config.checkpoint_dir, "accelerator"))
    logger.info(f"Saved Model to {config.checkpoint_dir}")
    torch.cuda.empty_cache()
    accelerator.end_training()


if __name__ == "__main__":
    main(cli())
