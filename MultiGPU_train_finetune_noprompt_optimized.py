#!/usr/bin/env python3
"""
Multi-GPU training script with optimized dataset loading and caching.
Supports both DataParallel and DistributedDataParallel training.
"""

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm
import monai
from pathlib import Path
import logging
import time
from datetime import datetime

import cfg
from models.sam import sam_model_registry
from utils.cached_dataset import CachedPublicDataset
from utils.dsc import dice_coeff_multi_class


def setup_logger(log_dir, rank=0):
    """Setup logger for training with file and console output."""
    # Create logs directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f'SAM_Training_Rank_{rank}')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - Rank%(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if rank == 0:  # Only main process writes to main log file
        log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler (all ranks can log to console, but we'll filter in practice)
    if rank == 0:  # Only main process logs to console to avoid spam
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def setup_distributed(rank, world_size, backend="nccl"):
    """Initialize distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def calculate_combined_loss(pred, msks, criterion1, criterion2):
    """Calculate combined dice and cross-entropy loss for multi-class logits."""
    # # For CrossEntropyLoss: use raw logits directly
    # msks_squeezed = torch.squeeze(msks.long(), 1)  # [B, H, W]
    # loss_ce = criterion2(pred, msks_squeezed)
    
    # # For DiceLoss: convert logits to probabilities using softmax
    # pred_softmax = torch.softmax(pred, dim=1)  # Convert logits to probabilities [B, 12, H, W]
    
    # # Convert integer masks to one-hot format for DiceLoss
    # msks_onehot = torch.nn.functional.one_hot(msks_squeezed, num_classes=pred.shape[1])  # [B, H, W, 12]
    # msks_onehot = msks_onehot.permute(0, 3, 1, 2).float()  # [B, 12, H, W]
    
    # # Calculate Dice loss with probabilities (not logits)
    # loss_dice = criterion1(pred_softmax, msks_onehot)
    
    # return loss_dice + loss_ce, loss_dice, loss_ce
    loss = criterion1(pred, msks.long())  # DiceCELoss handles logits directly
    return loss, loss * 0.5, loss * 0.5  # Return total, approximate dice component, ce component


def get_learning_rate_scheduler(optimizer, args, max_iterations):
    """Create and return appropriate learning rate scheduler."""
    if args.if_warmup:

        def lr_lambda(current_step):
            if current_step < args.warmup_period:
                return current_step / args.warmup_period
            else:
                shift_step = current_step - args.warmup_period
                remaining_steps = max_iterations - args.warmup_period
                return (1.0 - shift_step / remaining_steps) ** 0.9

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    dsc,
    dir_checkpoint,
    rank=0,
    is_best=False,
    is_periodic=False,
):
    """Save model checkpoint with metadata."""
    if rank == 0:  # Only main process saves checkpoints
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "dsc": dsc,
        }

        if is_best:
            torch.save(checkpoint, os.path.join(dir_checkpoint, "checkpoint_best.pth"))
            print(f"Best model saved with DSC: {dsc:.4f}")

        if is_periodic:
            torch.save(
                checkpoint,
                os.path.join(dir_checkpoint, f"checkpoint_epoch_{epoch}.pth"),
            )

        torch.save(checkpoint, os.path.join(dir_checkpoint, "checkpoint_latest.pth"))


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint for resuming training."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cuda")

        # Handle DDP models
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dsc = checkpoint.get("dsc", 0)
        print(f"Resumed from epoch {start_epoch}, best DSC: {best_dsc:.4f}")
        return start_epoch, best_dsc
    else:
        print("No checkpoint found, starting from scratch")
        return 0, 0


def train_epoch(
    model,
    trainloader,
    optimizer,
    scheduler,
    criterion1,
    criterion2,
    writer,
    epoch,
    iter_num,
    args,
    scaler=None,
    rank=0,
    logger=None,
):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    current_iter = iter_num
    
    if logger and rank == 0:
        logger.info(f"Starting training epoch {epoch}")

    if rank == 0:
        pbar = tqdm(trainloader, desc=f"Training Epoch {epoch}")
    else:
        pbar = trainloader

    for i, data in enumerate(pbar):
        batch_start_time = time.time()
        
        imgs = data["image"].cuda(non_blocking=True)
        msks = torchvision.transforms.Resize((args.out_size, args.out_size))(
            data["mask"]
        )
        msks = msks.cuda(non_blocking=True)

        optimizer.zero_grad()

        # Use mixed precision if scaler is provided
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                if args.if_update_encoder:
                    img_emb = (
                        model.module.image_encoder(imgs)
                        if hasattr(model, "module")
                        else model.image_encoder(imgs)
                    )
                else:
                    with torch.no_grad():
                        img_emb = (
                            model.module.image_encoder(imgs)
                            if hasattr(model, "module")
                            else model.image_encoder(imgs)
                        )

                # Get default embeddings
                sam_module = model.module if hasattr(model, "module") else model
                sparse_emb, dense_emb = sam_module.prompt_encoder(
                    points=None, boxes=None, masks=None
                )
                pred, _ = sam_module.mask_decoder(
                    image_embeddings=img_emb,
                    image_pe=sam_module.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                )
                loss, loss_dice, loss_ce = calculate_combined_loss(
                    pred, msks, criterion1, criterion2
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Update learning rate scheduler after optimizer step
            if scheduler is not None:
                scheduler.step()
        else:
            if args.if_update_encoder:
                img_emb = (
                    model.module.image_encoder(imgs)
                    if hasattr(model, "module")
                    else model.image_encoder(imgs)
                )
            else:
                with torch.no_grad():
                    img_emb = (
                        model.module.image_encoder(imgs)
                        if hasattr(model, "module")
                        else model.image_encoder(imgs)
                    )

            sam_module = model.module if hasattr(model, "module") else model
            sparse_emb, dense_emb = sam_module.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            pred, _ = sam_module.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam_module.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )
            loss, loss_dice, loss_ce = calculate_combined_loss(
                pred, msks, criterion1, criterion2
            )

            loss.backward()
            optimizer.step()
            # Update learning rate scheduler after optimizer step
            if scheduler is not None:
                scheduler.step()

        train_loss += loss.item()
        current_iter += 1
        
        batch_time = time.time() - batch_start_time

        # Log metrics (only on main process)
        if rank == 0 and writer is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("info/lr", current_lr, current_iter)
            writer.add_scalar("info/total_loss", loss, current_iter)
            writer.add_scalar("info/loss_ce", loss_ce, current_iter)
            writer.add_scalar("info/loss_dice", loss_dice, current_iter)
            writer.add_scalar("info/batch_time", batch_time, current_iter)
            
        # Log detailed info every 100 batches
        if logger and rank == 0 and i % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch}, Batch {i}/{len(trainloader)}: "
                       f"Loss={loss.item():.4f}, Dice={loss_dice.item():.4f}, "
                       f"CE={loss_ce.item():.4f}, LR={current_lr:.6f}, "
                       f"BatchTime={batch_time:.3f}s")

    train_loss /= i + 1
    
    if logger and rank == 0:
        logger.info(f"Epoch {epoch} training completed. Average loss: {train_loss:.4f}")
    
    return train_loss, current_iter


def validate_epoch(model, valloader, criterion1, criterion2, args, rank=0, logger=None):
    """Validate for one epoch."""
    model.eval()
    eval_loss = 0
    dsc = 0

    if logger and rank == 0:
        logger.info(f"Starting validation with {len(valloader)} batches")

    if rank == 0:
        pbar = tqdm(valloader, desc="Validating")
    else:
        pbar = valloader

    with torch.no_grad():
        for i, data in enumerate(pbar):
            imgs = data["image"].cuda(non_blocking=True)
            msks = torchvision.transforms.Resize((args.out_size, args.out_size))(
                data["mask"]
            )
            msks = msks.cuda(non_blocking=True)

            sam_module = model.module if hasattr(model, "module") else model
            img_emb = sam_module.image_encoder(imgs)
            sparse_emb, dense_emb = sam_module.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            pred, _ = sam_module.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam_module.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )

            loss, _, _ = calculate_combined_loss(pred, msks, criterion1, criterion2)
            eval_loss += loss.item()

            dsc_batch = dice_coeff_multi_class(
                pred.argmax(dim=1).cpu(),
                torch.squeeze(msks.long(), 1).cpu().long(),
                args.num_cls,
            )
            dsc += dsc_batch

    eval_loss /= i + 1
    dsc /= i + 1

    # Synchronize metrics across processes
    if dist.is_initialized():
        eval_loss_tensor = torch.tensor(eval_loss).cuda()
        dsc_tensor = torch.tensor(dsc).cuda()

        dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(dsc_tensor, op=dist.ReduceOp.SUM)

        eval_loss = eval_loss_tensor.item() / dist.get_world_size()
        dsc = dsc_tensor.item() / dist.get_world_size()

    if logger and rank == 0:
        logger.info(f"Validation completed. Loss: {eval_loss:.4f}, DSC: {dsc:.4f}")

    return eval_loss, dsc


def train_model_distributed(rank, world_size, args, train_dataset, eval_dataset):
    """Main training function for distributed training."""
    setup_distributed(rank, world_size)

    # Setup logging
    logger = setup_logger(args.dir_checkpoint, rank)
    
    if rank == 0:
        logger.info("=== Starting Distributed Training ===")
        logger.info(f"World size: {world_size}")
        logger.info(f"Architecture: {args.arch}")
        logger.info(f"Finetune type: {args.finetune_type}")
        logger.info(f"Batch size: {args.b}")
        logger.info(f"Number of classes: {args.num_cls}")

    # Initialize model
    if args.if_warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr

    sam = sam_model_registry[args.arch](
        args, checkpoint=os.path.join(args.sam_ckpt), num_classes=args.num_cls
    )

    # Configure fine-tuning strategy
    if args.finetune_type == "adapter":
        for n, value in sam.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
        if rank == 0:
            logger.info(f"Using adapter fine-tuning")
            logger.info(f"Update encoder: {args.if_update_encoder}")
            logger.info(f"Image encoder adapter: {args.if_encoder_adapter}")
            logger.info(f"Mask decoder adapter: {args.if_mask_decoder_adapter}")
            if args.if_encoder_adapter:
                logger.info(f"Added adapter layers: {args.encoder_adapter_depths}")

    elif args.finetune_type == "vanilla" and not args.if_update_encoder:
        if rank == 0:
            logger.info(f"Using vanilla fine-tuning")
            logger.info(f"Update encoder: {args.if_update_encoder}")
        for n, value in sam.image_encoder.named_parameters():
            value.requires_grad = False

    sam.cuda(rank)
    sam = DDP(sam, device_ids=[rank])

    # Setup data loaders with distributed sampling
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    val_sampler = DistributedSampler(
        eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=args.b,
        sampler=train_sampler,
        num_workers=args.num_workers // world_size,
        pin_memory=True,
    )
    valloader = DataLoader(
        eval_dataset,
        batch_size=args.b,
        sampler=val_sampler,
        num_workers=args.num_workers // world_size,
        pin_memory=True,
    )

    if rank == 0:
        logger.info(f"Training batches per epoch: {len(trainloader)}")
        logger.info(f"Validation batches per epoch: {len(valloader)}")

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        sam.parameters(),
        lr=b_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.1,
        amsgrad=False,
    )

    max_iterations = args.epochs * len(trainloader)
    scheduler = get_learning_rate_scheduler(optimizer, args, max_iterations)

    # Initialize mixed precision scaler
    use_amp = getattr(args, "mixed_precision", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Initialize loss functions
    criterion1 = monai.losses.DiceCELoss(
        sigmoid=False,  # For multi-class logits
        softmax=True,   # Apply softmax to logits
        to_onehot_y=True,  # Convert integer targets to one-hot
        reduction="mean",
        lambda_dice=1.0,
        lambda_ce=1.0
    )
    criterion2 = nn.CrossEntropyLoss()

    # Initialize tracking variables
    iter_num = 0
    writer = SummaryWriter(args.dir_checkpoint + "/log") if rank == 0 else None
    val_largest_dsc = 0
    last_update_epoch = 0
    start_epoch = 0

    # Resume from checkpoint if specified
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint:
        start_epoch, val_largest_dsc = load_checkpoint(
            sam, optimizer, args.resume_from_checkpoint
        )
        last_update_epoch = start_epoch
        if rank == 0:
            logger.info(f"Resumed training from epoch {start_epoch}")

    save_every_n_epochs = getattr(args, "save_every_n_epochs", 10)

    if rank == 0:
        logger.info(f"Starting distributed training on {world_size} GPUs for {args.epochs} epochs...")
        logger.info(f"Mixed precision training: {'Enabled' if use_amp else 'Disabled'}")
        logger.info(f"Learning rate: {b_lr}")
        logger.info(f"Max iterations: {max_iterations}")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling

        # Training phase
        train_loss, iter_num = train_epoch(
            sam,
            trainloader,
            optimizer,
            scheduler,
            criterion1,
            criterion2,
            writer,
            epoch,
            iter_num,
            args,
            scaler,
            rank,
            logger,
        )

        if rank == 0:
            logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # Validation phase (every 2 epochs)
        if epoch % 2 == 0:
            eval_loss, dsc = validate_epoch(
                sam, valloader, criterion1, criterion2, args, rank, logger
            )

            if rank == 0:
                # Log validation metrics
                if writer is not None:
                    writer.add_scalar("eval/loss", eval_loss, epoch)
                    writer.add_scalar("eval/dice", dsc, epoch)

                logger.info(f"Eval Epoch {epoch} | Val Loss: {eval_loss:.4f} | DSC: {dsc:.4f}")

                # Save best model
                if dsc > val_largest_dsc:
                    val_largest_dsc = dsc
                    last_update_epoch = epoch
                    logger.info(f"New best DSC: {dsc:.4f}")
                    save_checkpoint(
                        sam,
                        optimizer,
                        epoch,
                        eval_loss,
                        dsc,
                        args.dir_checkpoint,
                        rank,
                        is_best=True,
                    )

                # Early stopping check
                elif (epoch - last_update_epoch) > 20:
                    logger.info("Early stopping: No improvement for 20 epochs")
                    break

        # Periodic checkpoint saving
        if epoch % save_every_n_epochs == 0 and epoch > 0:
            save_checkpoint(
                sam,
                optimizer,
                epoch,
                train_loss,
                0,
                args.dir_checkpoint,
                rank,
                is_periodic=True,
            )
            if rank == 0:
                logger.info(f"Saved periodic checkpoint at epoch {epoch}")

        # Always save latest checkpoint
        save_checkpoint(sam, optimizer, epoch, train_loss, 0, args.dir_checkpoint, rank)

    if rank == 0:
        if writer is not None:
            writer.close()
        logger.info(f"Training completed. Best validation DSC: {val_largest_dsc:.4f}")
        logger.info("=== Training Finished ===")

    cleanup_distributed()
    return val_largest_dsc


def train_model_dataparallel(args, train_dataset, eval_dataset):
    """Training function for DataParallel (single-node multi-GPU)."""
    # Setup logging
    logger = setup_logger(args.dir_checkpoint, rank=0)
    
    logger.info("=== Starting DataParallel Training ===")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    logger.info(f"Architecture: {args.arch}")
    logger.info(f"Finetune type: {args.finetune_type}")
    logger.info(f"Batch size: {args.b}")
    logger.info(f"Number of classes: {args.num_cls}")

    # Initialize model
    if args.if_warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr

    sam = sam_model_registry[args.arch](
        args, checkpoint=os.path.join(args.sam_ckpt), num_classes=args.num_cls
    )

    # Configure fine-tuning strategy
    if args.finetune_type == "adapter":
        for n, value in sam.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
        logger.info("Using adapter fine-tuning")
        logger.info(f"Update encoder: {args.if_update_encoder}")
        logger.info(f"Image encoder adapter: {args.if_encoder_adapter}")
        logger.info(f"Mask decoder adapter: {args.if_mask_decoder_adapter}")
        if args.if_encoder_adapter:
            logger.info(f"Added adapter layers: {args.encoder_adapter_depths}")

    elif args.finetune_type == "vanilla" and not args.if_update_encoder:
        logger.info("Using vanilla fine-tuning")
        logger.info(f"Update encoder: {args.if_update_encoder}")
        for n, value in sam.image_encoder.named_parameters():
            value.requires_grad = False

    # Use DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        sam = nn.DataParallel(sam)
    sam.cuda()

    # Setup data loaders
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valloader = DataLoader(
        eval_dataset,
        batch_size=args.b,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Training batches per epoch: {len(trainloader)}")
    logger.info(f"Validation batches per epoch: {len(valloader)}")

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        sam.parameters(),
        lr=b_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.1,
        amsgrad=False,
    )

    max_iterations = args.epochs * len(trainloader)
    scheduler = get_learning_rate_scheduler(optimizer, args, max_iterations)

    # Initialize mixed precision scaler
    use_amp = getattr(args, "mixed_precision", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Initialize loss functions
    criterion1 = monai.losses.DiceLoss(
        sigmoid=True, squared_pred=True, to_onehot_y=True, reduction="mean"
    )
    criterion2 = nn.CrossEntropyLoss()

    # Initialize tracking variables
    iter_num = 0
    writer = SummaryWriter(args.dir_checkpoint + "/log")
    val_largest_dsc = 0
    last_update_epoch = 0
    start_epoch = 0

    # Resume from checkpoint if specified
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint:
        start_epoch, val_largest_dsc = load_checkpoint(
            sam, optimizer, args.resume_from_checkpoint
        )
        last_update_epoch = start_epoch
        logger.info(f"Resumed training from epoch {start_epoch}")

    save_every_n_epochs = getattr(args, "save_every_n_epochs", 10)

    logger.info(f"Starting DataParallel training for {args.epochs} epochs...")
    logger.info(f"Mixed precision training: {'Enabled' if use_amp else 'Disabled'}")
    logger.info(f"Learning rate: {b_lr}")
    logger.info(f"Max iterations: {max_iterations}")

    for epoch in range(start_epoch, args.epochs):
        # Training phase
        train_loss, iter_num = train_epoch(
            sam,
            trainloader,
            optimizer,
            scheduler,
            criterion1,
            criterion2,
            writer,
            epoch,
            iter_num,
            args,
            scaler,
            rank=0,
            logger=logger,
        )

        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # Validation phase (every 2 epochs)
        if epoch % 2 == 0:
            eval_loss, dsc = validate_epoch(
                sam, valloader, criterion1, criterion2, args, rank=0, logger=logger
            )

            # Log validation metrics
            writer.add_scalar("eval/loss", eval_loss, epoch)
            writer.add_scalar("eval/dice", dsc, epoch)

            logger.info(f"Eval Epoch {epoch} | Val Loss: {eval_loss:.4f} | DSC: {dsc:.4f}")

            # Save best model
            if dsc > val_largest_dsc:
                val_largest_dsc = dsc
                last_update_epoch = epoch
                logger.info(f"New best DSC: {dsc:.4f}")
                save_checkpoint(
                    sam,
                    optimizer,
                    epoch,
                    eval_loss,
                    dsc,
                    args.dir_checkpoint,
                    rank=0,
                    is_best=True,
                )

            # Early stopping check
            elif (epoch - last_update_epoch) > 20:
                logger.info("Early stopping: No improvement for 20 epochs")
                break

        # Periodic checkpoint saving
        if epoch % save_every_n_epochs == 0 and epoch > 0:
            save_checkpoint(
                sam,
                optimizer,
                epoch,
                train_loss,
                0,
                args.dir_checkpoint,
                rank=0,
                is_periodic=True,
            )
            logger.info(f"Saved periodic checkpoint at epoch {epoch}")

        # Always save latest checkpoint
        save_checkpoint(
            sam, optimizer, epoch, train_loss, 0, args.dir_checkpoint, rank=0
        )

    writer.close()
    logger.info(f"Training completed. Best validation DSC: {val_largest_dsc:.4f}")
    logger.info("=== Training Finished ===")
    return val_largest_dsc


def main():
    """Main function to handle different training modes."""
    import os

    args = cfg.parse_args(
        [
            "-num_cls",
            "12",
            "-if_warmup",
            "True",
            "-finetune_type",
            "adapter",
            "-arch",
            "vit_b",
            "-if_update_encoder",
            "True",
            "-if_encoder_adapter",
            "True",
            "-if_mask_decoder_adapter",
            "True",
            "-img_folder",
            "/home/admin/doku/datasets",
            "-mask_folder",
            "/home/admin/doku/datasets",
            "-sam_ckpt",
            "models/train_sam_hq/pretrained_checkpoint/sam_vit_b_01ec64.pth",
            "-targets",
            "multi_all",
            "-dataset_name",
            "cubicasa5k",
            "-dir_checkpoint",
            "2D-SAM_vit_b_encoderdecoder_adapter_cubicasa5k_noprompt",
            "-train_img_list",
            "/home/admin/doku/datasets/cubicasa5k/train.csv",
            "-val_img_list",
            "/home/admin/doku/datasets/cubicasa5k/val.csv",
            "-use_distributed",
            "-b",
            "1"
        ]
    )

    # Create checkpoint directory
    Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)

    # Setup initial logging for main process
    main_logger = setup_logger(args.dir_checkpoint, rank=0)
    main_logger.info("=== SAM Training Script Started ===")
    main_logger.info(f"Checkpoint directory: {args.dir_checkpoint}")
    main_logger.info(f"Dataset: {args.dataset_name}")
    main_logger.info(f"Architecture: {args.arch}")
    main_logger.info(f"Number of classes: {args.num_cls}")
    main_logger.info(f"Batch size: {args.b}")
    main_logger.info(f"Epochs: {args.epochs}")

    # Save arguments
    path_to_json = os.path.join(args.dir_checkpoint, "args.json")
    args_dict = vars(args)
    with open(path_to_json, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)
    main_logger.info(f"Saved training arguments to {path_to_json}")

    # Set number of workers
    num_workers = getattr(args, "num_workers", 8)
    args.num_workers = num_workers
    main_logger.info(f"Using {num_workers} workers for data loading")

    main_logger.info("Creating optimized datasets with caching...")

    # Create datasets with caching
    train_dataset = CachedPublicDataset(
        args,
        args.img_folder,
        args.mask_folder,
        args.train_img_list,
        phase="train",
        targets=[args.targets],
        normalize_type="sam",
        if_prompt=False,
        use_cache=True,
        num_workers_preprocessing=num_workers,
    )

    eval_dataset = CachedPublicDataset(
        args,
        args.img_folder,
        args.mask_folder,
        args.val_img_list,
        phase="val",
        targets=[args.targets],
        normalize_type="sam",
        if_prompt=False,
        use_cache=True,
        num_workers_preprocessing=num_workers,
    )

    main_logger.info(f"Training dataset size: {len(train_dataset)}")
    main_logger.info(f"Validation dataset size: {len(eval_dataset)}")

    # Determine training mode
    use_distributed = getattr(args, "use_distributed", False)
    world_size = torch.cuda.device_count()

    if use_distributed and world_size > 1:
        main_logger.info(f"Using DistributedDataParallel with {world_size} GPUs")
        mp.spawn(
            train_model_distributed,
            args=(world_size, args, train_dataset, eval_dataset),
            nprocs=world_size,
            join=True,
        )
    else:
        main_logger.info("Using DataParallel or single GPU training")
        train_model_dataparallel(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    main()
