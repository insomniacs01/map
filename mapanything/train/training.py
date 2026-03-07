# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Training Code for MapAnything.

References:
DUSt3R: https://github.com/naver/dust3r
"""

import datetime
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sized

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import mapanything.utils.train_tools as train_tools
from mapanything.datasets import get_test_data_loader, get_train_data_loader
from mapanything.models import init_model
from mapanything.train.losses import *  # noqa
from mapanything.utils.inference import loss_of_one_batch_multi_view
from mapanything.utils.geometry import (
    closed_form_pose_inverse,
    geotrf,
    normalize_multiple_pointclouds,
    transform_pose_using_quats_and_trans_2_to_1,
)
from mapanything.utils.train_tools import NativeScalerWithGradNormCount as NativeScaler

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


def train(args):
    """
    Main training function that handles the entire training process.

    This function initializes the distributed training environment, sets up datasets,
    initializes the model, optimizer, and loss functions, and manages the training
    and evaluation loop across multiple epochs.

    In this training, an epoch is just a chunk of the entire dataset.

    Args:
        args: Configuration object containing all training parameters including
              dataset configs, model configs, training parameters, and loss functions.
    """
    # Initialize distributed training if required
    train_tools.init_distributed_mode(args.distributed)
    global_rank = train_tools.get_rank()
    world_size = train_tools.get_world_size()  # noqa
    print(
        f"[rank {global_rank}] init_distributed_mode done "
        f"(world_size={world_size}, gpu={getattr(args.distributed, 'gpu', None)})",
        force=True,
    )

    # Init output directory and device
    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Configure SDPA backends (Flash / Mem-Efficient / Math / cuDNN).
    # Some driver/kernel combos can trigger "CUDA driver error: invalid argument" during SDPA.
    sdpa_cfg = getattr(args.train_params, "sdpa", None)
    if torch.cuda.is_available() and sdpa_cfg is not None:
        try:
            enable_flash = bool(sdpa_cfg.get("enable_flash", True))
            enable_math = bool(sdpa_cfg.get("enable_math", True))
            enable_mem_efficient = bool(sdpa_cfg.get("enable_mem_efficient", True))
            enable_cudnn = bool(sdpa_cfg.get("enable_cudnn", True))
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(enable_flash)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(enable_math)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient)
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(enable_cudnn)
            if global_rank == 0:
                print(
                    "SDPA backends: "
                    f"flash={enable_flash}, "
                    f"mem_efficient={enable_mem_efficient}, "
                    f"math={enable_math}, "
                    f"cudnn={enable_cudnn}"
                )
        except Exception as sdpa_err:
            if global_rank == 0:
                print(f"Warning: failed to configure SDPA backends: {sdpa_err}")

    # Fix the seed
    seed = args.train_params.seed + train_tools.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.train_params.disable_cudnn_benchmark

    # Datasets and Dataloaders
    print(f"Building train dataset {args.dataset.train_dataset}")
    data_loader_train = build_dataset(
        dataset=args.dataset.train_dataset,
        num_workers=args.dataset.num_workers,
        test=False,
        max_num_of_imgs_per_gpu=args.train_params.max_num_of_imgs_per_gpu,
    )
    print(f"[rank {global_rank}] train dataloader built", force=True)
    print(f"Building test dataset {args.dataset.test_dataset}")
    # Since we don't have any backward overhead in test mode we default to a larger
    # batch size, but always clamp to >= 1 (important when num_views > max_imgs_per_gpu).
    eval_batch_size = getattr(args.train_params, "eval_batch_size", None)
    if eval_batch_size is None:
        test_batch_size = max(
            1,
            2 * (args.train_params.max_num_of_imgs_per_gpu // args.dataset.num_views),
        )
    else:
        test_batch_size = max(1, int(eval_batch_size))
    def _test_dataset_key(dataset_str: str, idx: int) -> str:
        ds = str(dataset_str).strip()
        prefix = ds.split("(", 1)[0].strip()
        split_val = None
        if "split='" in ds:
            split_val = ds.split("split='", 1)[1].split("'", 1)[0]
        elif 'split="' in ds:
            split_val = ds.split('split="', 1)[1].split('"', 1)[0]
        if split_val:
            return f"{prefix}:{split_val}"
        return f"{idx}:{prefix}"

    test_dataset_entries = [ds for ds in args.dataset.test_dataset.split("+") if "(" in ds]
    data_loader_test = {
        _test_dataset_key(dataset, idx): build_dataset(
            dataset=dataset,
            num_workers=args.dataset.num_workers,
            test=True,
            batch_size=test_batch_size,
        )
        for idx, dataset in enumerate(test_dataset_entries)
    }
    print(f"[rank {global_rank}] test dataloader(s) built", force=True)
    data_loader_test_full = {}
    test_dataset_full = getattr(args.dataset, "test_dataset_full", None)
    if test_dataset_full:
        test_dataset_full_entries = [
            ds for ds in str(test_dataset_full).split("+") if "(" in ds
        ]
        data_loader_test_full = {
            "full/" + _test_dataset_key(dataset, idx): build_dataset(
                dataset=dataset,
                num_workers=args.dataset.num_workers,
                test=True,
                batch_size=test_batch_size,
            )
            for idx, dataset in enumerate(test_dataset_full_entries)
        }
        print(f"[rank {global_rank}] test_full dataloader(s) built", force=True)

    # Load Model
    print(f"[rank {global_rank}] init_model start", force=True)
    if global_rank == 0:
        model = init_model(
            args.model.model_str,
            args.model.model_config,
            torch_hub_force_reload=args.model.torch_hub_force_reload,
        )
    # NOTE: Historically we used an NCCL barrier here so non-master ranks wait
    # for rank0 to finish `init_model()` (torch.hub + checkpoint IO). On some
    # hosts this barrier can hang indefinitely even though the process group
    # is healthy, blocking all training/eval. Since our entry scripts prefetch
    # the torch.hub weights (DINOv2) and each rank can safely initialize the
    # model from the local cache, we skip this barrier to avoid deadlocks.
    if global_rank != 0:
        model = init_model(
            args.model.model_str, args.model.model_config, torch_hub_force_reload=False
        )
    model.to(device)  # Move model to device
    print(f"[rank {global_rank}] init_model done + moved to device", force=True)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # Criterion
    print(f">> Creating train criterion = {args.loss.train_criterion}")
    train_criterion = eval(args.loss.train_criterion).to(device)
    print(
        f">> Creating test criterion = {args.loss.test_criterion or args.loss.train_criterion}"
    )
    test_criterion = eval(args.loss.test_criterion or args.loss.train_criterion).to(
        device
    )
    print(f"[rank {global_rank}] criteria ready", force=True)

    # Load pretrained model if provided
    if args.model.pretrained:
        print("Loading pretrained: ", args.model.pretrained)
        ckpt = torch.load(
            args.model.pretrained, map_location=device, weights_only=False
        )
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt  # in case it occupies memory

    # Init model for DDP training
    if args.distributed.distributed:
        print(f"[rank {global_rank}] before DDP wrap", force=True)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.distributed.gpu],
            find_unused_parameters=True,
            static_graph=False,
        )
        model_without_ddp = model.module
        print(f"[rank {global_rank}] after DDP wrap", force=True)

    # Optimizer and loss scaler for gradient accumulation
    # Following timm: set wd as 0 for bias and norm layers
    print(f"[rank {global_rank}] before get_parameter_groups", force=True)
    param_groups, param_groups_name_to_idx_map, param_groups_idx_to_name_map = (
        train_tools.get_parameter_groups(
            model_without_ddp,
            args.train_params.lr,
            args.train_params.weight_decay,
            submodule_configs=args.train_params.submodule_configs,
            warn_not_in_submodule=args.train_params.warn_not_in_submodule,
        )
    )
    print(f"[rank {global_rank}] after get_parameter_groups", force=True)
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.train_params.lr, betas=(0.9, 0.95)
    )
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        """
        Writes training and testing statistics to log files and TensorBoard.

        Args:
            epoch: Current epoch number.
            train_stats: Dictionary containing training metrics.
            test_stats: Dictionary containing testing metrics for each test dataset.
        """
        if train_tools.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )
            for test_name in data_loader_test_full:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        """
        Saves model checkpoint to disk.

        Args:
            epoch: Current epoch number.
            fname: Filename or identifier for the checkpoint.
            best_so_far: Best validation metric achieved so far.
        """
        train_tools.save_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    # Resume from a checkpoint if needed
    last_ckpt_fname = os.path.join(args.output_dir, "checkpoint-last.pth")
    if args.train_params.resume and os.path.isfile(last_ckpt_fname):
        args.train_params.resume_ckpt = last_ckpt_fname
    else:
        args.train_params.resume_ckpt = None
    best_so_far = train_tools.load_model(
        train_args=args.train_params,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    if best_so_far is None:
        best_so_far = float("inf")

    if (
        global_rank == 0
        and args.output_dir is not None
        and not getattr(args.train_params, "disable_tensorboard", False)
    ):
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.train_params.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.train_params.start_epoch, args.train_params.epochs + 1):
        # Save immediately the last checkpoint
        if epoch > args.train_params.start_epoch:
            if (
                args.train_params.save_freq
                and epoch % args.train_params.save_freq == 0
                or epoch == args.train_params.epochs
            ):
                save_model(epoch - 1, "last", best_so_far)

        # Test on multiple datasets
        new_best = False
        test_stats = {}
        ran_fast_eval = False
        ran_full_eval = False
        if (
            args.train_params.eval_freq > 0
            and epoch % args.train_params.eval_freq == 0
            and epoch > 0
        ):
            ran_fast_eval = True
            for test_name, testset in data_loader_test.items():
                print(f"Testing on {test_name} ...")
                stats = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=test_name,
                )
                test_stats[test_name] = stats

            # Calculate average test loss median (fast subset)
            fast_keys = [k for k in data_loader_test.keys() if k in test_stats]
            if fast_keys:
                avg_test_loss_med = np.mean(
                    [test_stats[k]["loss_med"] for k in fast_keys]
                )
                test_stats["Average Test Loss Median (fast)"] = avg_test_loss_med
                # Save best using fast eval unless full eval runs this epoch.
                if avg_test_loss_med < best_so_far:
                    best_so_far = avg_test_loss_med
                    new_best = True

        full_eval_freq = int(getattr(args.train_params, "full_eval_freq", 0) or 0)
        if (
            full_eval_freq > 0
            and data_loader_test_full
            and epoch % full_eval_freq == 0
            and epoch > 0
        ):
            ran_full_eval = True
            for test_name, testset in data_loader_test_full.items():
                print(f"Testing on {test_name} ...")
                stats = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=test_name,
                )
                test_stats[test_name] = stats

            full_keys = [k for k in data_loader_test_full.keys() if k in test_stats]
            if full_keys:
                avg_full_loss_med = np.mean(
                    [test_stats[k]["loss_med"] for k in full_keys]
                )
                test_stats["Average Test Loss Median (full)"] = avg_full_loss_med
                # Prefer full eval for best checkpoint tracking when available.
                if avg_full_loss_med < best_so_far:
                    best_so_far = avg_full_loss_med
                    new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.train_params.start_epoch:
            if args.train_params.keep_freq and epoch % args.train_params.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, "best", best_so_far)
        if epoch >= args.train_params.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            param_groups_name_to_idx_map=param_groups_name_to_idx_map,
            param_groups_idx_to_name_map=param_groups_idx_to_name_map,
            model_without_ddp=model_without_ddp,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    save_final_model(
        args, args.train_params.epochs, model_without_ddp, best_so_far=best_so_far
    )


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    """
    Saves the final model checkpoint after training completion.

    Args:
        args: Configuration object containing output directory information.
        epoch: Current epoch number.
        model_without_ddp: Model state dictionary or model instance without DistributedDataParallel wrapper.
        best_so_far: Optional; Best validation metric achieved during training.
    """
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": model_without_ddp
        if isinstance(model_without_ddp, dict)
        else model_without_ddp.cpu().state_dict(),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    train_tools.save_on_master(to_save, checkpoint_path)


def build_dataset(
    dataset, num_workers, test, batch_size=None, max_num_of_imgs_per_gpu=None
):
    """
    Builds data loaders for training or testing.

    Args:
        dataset: Dataset specification string.
        num_workers: Number of worker processes for data loading.
        test: Boolean flag indicating whether this is a test dataset.
        batch_size: Number of samples per batch. Defaults to None. Used only for testing.
        max_num_of_imgs_per_gpu: Maximum number of images per GPU. Defaults to None. Used only for training.

    Returns:
        DataLoader: PyTorch DataLoader configured for the specified dataset.
    """
    split = ["Train", "Test"][test]
    print(f"Building {split} Data loader for dataset: ", dataset)
    if test:
        assert batch_size is not None, (
            "batch_size must be specified for testing dataloader"
        )
        loader = get_test_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=False,
            drop_last=False,
        )
    else:
        assert max_num_of_imgs_per_gpu is not None, (
            "max_num_of_imgs_per_gpu must be specified for training dataloader"
        )
        loader = get_train_data_loader(
            dataset=dataset,
            max_num_of_imgs_per_gpu=max_num_of_imgs_per_gpu,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=True,
            drop_last=True,
        )

    print(f"{split} dataset length: ", len(loader))
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
    param_groups_name_to_idx_map=None,
    param_groups_idx_to_name_map=None,
    model_without_ddp=None,
):
    """
    Trains the model for one epoch.
    Epoch is just a chunk of the entire dataset.

    This function handles the training loop for a single epoch, including forward/backward passes,
    gradient accumulation, learning rate scheduling, and logging metrics.

    Args:
        model: The neural network model to train.
        criterion: Loss function to optimize.
        data_loader: DataLoader providing the training data.
        optimizer: Optimizer for updating model parameters.
        device: Device to run training on (CPU or GPU).
        epoch: Current epoch number.
        loss_scaler: Scaler for gradient accumulation and mixed precision training.
        args: Configuration object containing training parameters.
        log_writer: Optional; TensorBoard SummaryWriter for logging.
        param_groups_name_to_idx_map: Mapping from parameter group names to indices.
        param_groups_idx_to_name_map: Mapping from parameter group indices to names.
        model_without_ddp: Model without DistributedDataParallel wrapper for debugging.

    Returns:
        dict: Dictionary containing training metrics averaged over the epoch.
    """
    model.train(True)
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    for submodule_name in param_groups_name_to_idx_map:
        lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
        metric_logger.add_meter(
            lr_name, train_tools.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.train_params.accum_iter
    amp_enabled = bool(args.train_params.amp)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()
    bad_loss_count = 0
    max_bad_loss_count = int(getattr(args.train_params, "max_bad_loss_count", 20))
    max_loss_value = float(getattr(args.train_params, "max_loss_value", 1000.0))
    bad_loss_dump_max = int(getattr(args.train_params, "bad_loss_dump_max", 0))
    bad_loss_dump_stride = int(getattr(args.train_params, "bad_loss_dump_stride", 1))
    bad_loss_dump_suppressed = False
    # Track how many debug *dumps* we actually wrote. This must be separate from
    # `bad_loss_count` (number of bad batches), otherwise the combination
    # `bad_loss_dump_stride>1` and a small `bad_loss_dump_max` can suppress all dumps.
    bad_loss_dump_count = 0
    bad_loss_backoff = bool(getattr(args.train_params, "bad_loss_backoff", False))
    bad_loss_backoff_factor = float(
        getattr(args.train_params, "bad_loss_backoff_factor", 0.5)
    )
    bad_loss_backoff_min_lr = float(
        getattr(args.train_params, "bad_loss_backoff_min_lr", 1e-7)
    )
    bad_loss_backoff_max = int(getattr(args.train_params, "bad_loss_backoff_max", 3))
    bad_loss_backoff_count = 0
    bad_loss_disable_amp = bool(getattr(args.train_params, "bad_loss_disable_amp", False))
    bad_loss_reset_optim = bool(getattr(args.train_params, "bad_loss_reset_optim", False))

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch)
        epoch_f = epoch + data_iter_step / len(data_loader)

        # We use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            train_tools.adjust_learning_rate(
                optimizer,
                epoch_f,
                args.train_params,
                param_groups_idx_to_name_map,
                args.train_params.submodule_configs,
            )

        loss_tuple = loss_of_one_batch_multi_view(
            batch,
            model,
            criterion,
            device,
            use_amp=amp_enabled,
            amp_dtype=args.train_params.amp_dtype,
            ret="loss",
        )
        loss, loss_details = loss_tuple  # criterion returns two values
        if n_views > 2:
            loss = loss * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)
        loss_value = float(loss)

        bad_loss_local = (not math.isfinite(loss_value)) or (loss_value > max_loss_value)
        if torch.distributed.is_initialized():
            bad_loss_tensor = torch.tensor(
                1 if bad_loss_local else 0, device=device, dtype=torch.int32
            )
            torch.distributed.all_reduce(
                bad_loss_tensor, op=torch.distributed.ReduceOp.MAX
            )
            bad_loss = bool(bad_loss_tensor.item())
        else:
            bad_loss = bad_loss_local

        if bad_loss:
            bad_loss_count += 1
            if bad_loss_local:
                print(
                    f"[WARN] Bad loss={loss_value} at epoch={epoch} iter={data_iter_step} "
                    f"(max_loss_value={max_loss_value}); skipping batch "
                    f"[{bad_loss_count}/{max_bad_loss_count}].",
                    force=True,
                )
                print(f"[WARN] Loss Details: {loss_details}", force=True)
                dump_limit_reached = (
                    bad_loss_dump_max > 0 and bad_loss_dump_count >= bad_loss_dump_max
                )
                dump_stride_skip = (
                    bad_loss_dump_stride > 1
                    and (bad_loss_count % bad_loss_dump_stride) != 0
                )
                should_dump = not (dump_limit_reached or dump_stride_skip)
                if should_dump:
                    # Save debugging material for further inspection
                    debug_prefix = f"badloss_e{epoch}_it{data_iter_step}_n{bad_loss_count}"
                    for view_idx, view in enumerate(batch):
                        view_cpu = {}
                        for k, v in view.items():
                            view_cpu[k] = v.cpu() if isinstance(v, torch.Tensor) else v
                        with open(
                            os.path.join(
                                args.output_dir,
                                f"{debug_prefix}_view{view_idx}.pkl",
                            ),
                            "wb",
                        ) as f:
                            pickle.dump(view_cpu, f)
                    checkpoint_debug_path = os.path.join(
                        args.output_dir, f"{debug_prefix}_checkpoint.pth"
                    )
                    def _state_dict_to_cpu(module: torch.nn.Module) -> dict:
                        state = module.state_dict()
                        return {
                            key: (value.detach().cpu() if torch.is_tensor(value) else value)
                            for key, value in state.items()
                        }

                    to_save_debug = {
                        "args": args,
                        "model": (
                            model_without_ddp
                            if isinstance(model_without_ddp, dict)
                            else _state_dict_to_cpu(model_without_ddp)
                        ),
                        "epoch": epoch,
                        "data_iter_step": data_iter_step,
                        "loss_value": loss_value,
                        "loss_details": loss_details,
                    }
                    torch.save(to_save_debug, checkpoint_debug_path)
                    print(
                        f"[WARN] Saved debugging material to {args.output_dir}",
                        force=True,
                    )
                    bad_loss_dump_count += 1
                elif dump_limit_reached and not bad_loss_dump_suppressed:
                    print(
                        f"[WARN] Badloss debug dump capped at {bad_loss_dump_max}; "
                        "further dumps suppressed.",
                        force=True,
                    )
                    bad_loss_dump_suppressed = True
            elif train_tools.is_main_process():
                print(
                    f"[WARN] Bad loss detected on another rank at epoch={epoch} iter={data_iter_step}; skipping batch.",
                    force=True,
                )

            optimizer.zero_grad()
            del batch
            if bad_loss_count >= max_bad_loss_count:
                if bad_loss_backoff and bad_loss_backoff_count < bad_loss_backoff_max:
                    bad_loss_backoff_count += 1
                    # Reduce LR for all param groups to stabilize training.
                    for pg in optimizer.param_groups:
                        new_lr = max(
                            float(pg.get("lr", 0.0)) * bad_loss_backoff_factor,
                            bad_loss_backoff_min_lr,
                        )
                        pg["lr"] = new_lr
                    if bad_loss_disable_amp and amp_enabled:
                        amp_enabled = False
                        print(
                            "[WARN] Bad loss backoff: disabling AMP for stability.",
                            force=True,
                        )
                    if bad_loss_reset_optim:
                        optimizer.state.clear()
                        print(
                            "[WARN] Bad loss backoff: optimizer state cleared.",
                            force=True,
                        )
                    print(
                        f"[WARN] Bad loss backoff #{bad_loss_backoff_count}/{bad_loss_backoff_max}: "
                        f"reduced lr by factor={bad_loss_backoff_factor} (min_lr={bad_loss_backoff_min_lr}).",
                        force=True,
                    )
                    bad_loss_count = 0
                    torch.cuda.empty_cache()
                    continue
                print(
                    f"[ERROR] Reached max_bad_loss_count={max_bad_loss_count}; stopping training.",
                    force=True,
                )
                sys.exit(1)
            continue

        # Scale the loss by the number of gradient accumulation iterations
        loss /= accum_iter

        # Compute the scaled gradients (also clip the gradients to max norm of 1)
        gradient_norm = loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=1.0,
        )

        # Zero out the gradients to prepare for the next iteration of gradient descent
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        metric_logger.update(epoch=epoch_f)
        for submodule_name in param_groups_name_to_idx_map:
            lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
            log_lr = optimizer.param_groups[
                param_groups_name_to_idx_map[submodule_name][0]
            ]["lr"]
            metric_logger.meters[lr_name].update(log_lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and (
            (data_iter_step + 1) % (accum_iter * args.train_params.print_freq)
        ) == 0:
            loss_value_reduce = train_tools.all_reduce_mean(
                loss_value
            )  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            if gradient_norm is not None:
                log_writer.add_scalar("train_grad_norm", gradient_norm, epoch_1000x)
            for submodule_name in param_groups_name_to_idx_map:
                lr_name = (
                    f"train_lr_{submodule_name}"
                    if submodule_name != "default"
                    else "train_lr"
                )
                log_lr = optimizer.param_groups[
                    param_groups_name_to_idx_map[submodule_name][0]
                ]["lr"]
                log_writer.add_scalar(lr_name, log_lr, epoch_1000x)
            log_writer.add_scalar("train_iter", epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar("train_" + name, val, epoch_1000x)

    # # Gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):
    """
    Evaluates the model on a test dataset for one epoch.
    Epoch is just a chunk of the entire dataset.

    This function runs evaluation on the test dataset without computing gradients,
    and collects metrics for model performance assessment.

    Args:
        model: The neural network model to evaluate.
        criterion: Loss function for evaluation.
        data_loader: DataLoader providing the test data.
        device: Device to run evaluation on (CPU or GPU).
        epoch: Current epoch number.
        args: Configuration object containing evaluation parameters.
        log_writer: Optional; TensorBoard SummaryWriter for logging.
        prefix: String prefix for logging metrics.

    Returns:
        dict: Dictionary containing evaluation metrics (average and median values).
    """

    def _rotation_angle_deg_from_quats_xyzw(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
        dot = (q_pred * q_gt).sum(dim=-1).abs().clamp(max=1.0)
        ang = 2.0 * torch.acos(dot)
        return ang * (180.0 / math.pi)

    model.eval()
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(
        lambda: train_tools.SmoothedValue(window_size=9**9)
    )
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if args.train_params.freeze_val_samples_across_all_epochs:
        dataloader_epoch = 0
    else:
        dataloader_epoch = epoch
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(dataloader_epoch)

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch)
        result = loss_of_one_batch_multi_view(
            batch,
            model,
            criterion,
            device,
            use_amp=bool(args.train_params.amp),
            amp_dtype=args.train_params.amp_dtype,
            ret=None,
        )
        loss_value, loss_details = result["loss"]  # criterion returns two values
        if n_views > 2:
            loss_value = loss_value * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)

        preds = [result[f"pred{i + 1}"] for i in range(n_views)]

        # Metric monitoring for OPV2V-style datasets (only logs when pose+depth GT is available).
        try:
            depth_abs = 0.0
            depth_sq = 0.0
            depth_count = 0
            for view, pred in zip(batch, preds):
                gt_depth_z = view["pts3d_cam"][..., 2].float()
                valid = view["valid_mask"]
                pred_depth_z = pred["pts3d_cam"][..., 2].float()
                mask = valid & (gt_depth_z > 1e-6) & torch.isfinite(pred_depth_z)
                if not bool(mask.any()):
                    continue
                diff = (pred_depth_z[mask] - gt_depth_z[mask]).float()
                depth_abs += float(diff.abs().sum().detach().cpu())
                depth_sq += float((diff * diff).sum().detach().cpu())
                depth_count += int(mask.sum().detach().cpu())

            depth_mae = None
            depth_rmse = None
            if depth_count > 0:
                depth_mae = depth_abs / float(depth_count)
                depth_rmse = math.sqrt(depth_sq / float(depth_count))

            q0 = preds[0]["cam_quats"].float()
            t0 = preds[0]["cam_trans"].float()
            gt_q0 = batch[0]["camera_pose_quats"].float()
            gt_t0 = batch[0]["camera_pose_trans"].float()

            pose_trans_err = []
            pose_rot_err = []
            for view_idx in range(1, n_views):
                q_rel, t_rel = transform_pose_using_quats_and_trans_2_to_1(
                    q0,
                    t0,
                    preds[view_idx]["cam_quats"].float(),
                    preds[view_idx]["cam_trans"].float(),
                )
                gt_q_rel, gt_t_rel = transform_pose_using_quats_and_trans_2_to_1(
                    gt_q0,
                    gt_t0,
                    batch[view_idx]["camera_pose_quats"].float(),
                    batch[view_idx]["camera_pose_trans"].float(),
                )
                pose_trans_err.append(torch.linalg.norm((t_rel - gt_t_rel), dim=-1))
                pose_rot_err.append(_rotation_angle_deg_from_quats_xyzw(q_rel, gt_q_rel))

            pose_trans = None
            pose_rot = None
            if pose_trans_err:
                pose_trans = (
                    torch.stack(pose_trans_err, dim=0).mean(dim=0).mean().item()
                )
                pose_rot = torch.stack(pose_rot_err, dim=0).mean(dim=0).mean().item()

            metric_updates = {}
            if depth_mae is not None and math.isfinite(depth_mae):
                metric_updates["depth_z_mae_m"] = float(depth_mae)
            if depth_rmse is not None and math.isfinite(depth_rmse):
                metric_updates["depth_z_rmse_m"] = float(depth_rmse)
            if pose_trans is not None and math.isfinite(pose_trans):
                metric_updates["pose_trans_l2_m"] = float(pose_trans)
            if pose_rot is not None and math.isfinite(pose_rot):
                metric_updates["pose_rot_deg"] = float(pose_rot)
            # Ratio-space scale diagnostics (more human-friendly than raw loss).
            scale_vals = []
            pred_scale_per_sample = []
            for pred in preds:
                scale = pred.get("metric_scaling_factor")
                if not isinstance(scale, torch.Tensor):
                    continue
                scale = scale.detach().float()
                if scale.numel() == 0:
                    continue
                # Per-sample mean scale (for ratio-to-GT diagnostics).
                if scale.ndim == 1:
                    scale_mean = scale
                else:
                    scale_mean = scale.view(scale.shape[0], -1).mean(dim=1)
                pred_scale_per_sample.append(scale_mean.detach())

                # Flattened values (legacy ratio-to-1 diagnostics).
                flat = scale.view(-1)
                flat = flat[torch.isfinite(flat)]
                if flat.numel() > 0:
                    scale_vals.append(flat.cpu())
            if scale_vals:
                scale_all = torch.cat(scale_vals, dim=0)
                scale_all = torch.clamp(scale_all, min=1e-8)
                abs_rel = (scale_all - 1.0).abs()
                abs_log = scale_all.log().abs()
                scale_err_mean = abs_rel.mean().item()
                scale_log_err_mean = abs_log.mean().item()
                scale_eq_rel_err_mean = math.expm1(scale_log_err_mean)
                metric_updates["scale_err_mean"] = float(scale_err_mean)
                metric_updates["scale_log_err_mean"] = float(scale_log_err_mean)
                metric_updates["scale_eq_rel_err_mean"] = float(scale_eq_rel_err_mean)
                metric_updates["scale_ratio_mean"] = float(scale_all.mean().item())

            # OPV2V-correct ratio-to-GT diagnostics:
            # GT scale is the per-frame pointmap normalization factor `g` (avg distance to origin in view0 frame),
            # so the correct objective is `s/g -> 1` (not `s -> 1`).
            try:
                if pred_scale_per_sample:
                    pred_scale = torch.stack(pred_scale_per_sample, dim=0).mean(dim=0)  # (B,)
                    pred_scale = torch.clamp(pred_scale, min=1e-8)

                    # Compute GT normalization factor `g` from GT pointmaps in view0 camera frame.
                    if all(("pts3d" in b and "valid_mask" in b) for b in batch) and "camera_pose" in batch[0]:
                        in_camera0 = closed_form_pose_inverse(batch[0]["camera_pose"])
                        no_norm_gt_pts = [geotrf(in_camera0, b["pts3d"]) for b in batch]
                        valid_masks = [b["valid_mask"] for b in batch]
                        gt_norm_factor = normalize_multiple_pointclouds(
                            no_norm_gt_pts, valid_masks, "avg_dis", ret_factor=True
                        )[-1]
                        gt_scale = gt_norm_factor[:, 0, 0, 0].detach().float()

                        mask = torch.isfinite(gt_scale) & (gt_scale > 1e-8)
                        if "is_metric_scale" in batch[0]:
                            ms = batch[0]["is_metric_scale"].detach()
                            if isinstance(ms, torch.Tensor) and ms.shape == mask.shape:
                                mask = mask & ms.bool()

                        if mask.any():
                            ratio_to_gt = pred_scale[mask] / gt_scale[mask]
                            ratio_to_gt = torch.clamp(ratio_to_gt, min=1e-8)
                            abs_rel_gt = (ratio_to_gt - 1.0).abs().mean().item()
                            abs_log_gt = ratio_to_gt.log().abs().mean().item()
                            metric_updates["scale_gt_factor_mean"] = float(gt_scale[mask].mean().item())
                            metric_updates["scale_to_gt_err_mean"] = float(abs_rel_gt)
                            metric_updates["scale_to_gt_log_err_mean"] = float(abs_log_gt)
                            metric_updates["scale_to_gt_eq_rel_err_mean"] = float(math.expm1(abs_log_gt))
                            metric_updates["scale_to_gt_ratio_mean"] = float(ratio_to_gt.mean().item())
            except Exception:
                pass
            if metric_updates:
                metric_logger.update(**metric_updates)
        except Exception:
            pass

        metric_logger.update(loss=float(loss_value), **loss_details)

    # # Gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

    return results
