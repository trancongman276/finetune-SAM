# from segment_anything import SamPredictor, sam_model_registry
import json

# from models.sam_LoRa import LoRA_Sam
# Scientific computing
import os
from pathlib import Path

import monai

# Pytorch packages
import torch
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.cuda.amp import autocast

# Visulization
# Others
from torch.utils.data import DataLoader
from tqdm import tqdm

import cfg
from models.sam import sam_model_registry
from utils.dataset import Public_dataset
from utils.cached_dataset import CachedPublicDataset
from utils.dsc import dice_coeff_multi_class

# Use the arguments
args = cfg.parse_args()
# you need to modify based on the layer of adapters you are choosing to add
# comment it if you are not using adapter
# args.encoder_adapter_depths = [0,1,2,3]

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def calculate_combined_loss(pred, msks, criterion1, criterion2):
    """Calculate combined dice and cross-entropy loss."""
    loss_dice = criterion1(pred, msks.float())
    loss_ce = criterion2(pred, torch.squeeze(msks.long(), 1))
    return loss_dice + loss_ce, loss_dice, loss_ce


def get_learning_rate_scheduler(optimizer, args, max_iterations):
    """Create and return appropriate learning rate scheduler."""
    if args.if_warmup:
        def lr_lambda(current_step):
            if current_step < args.warmup_period:
                # Warmup phase
                return current_step / args.warmup_period
            else:
                # Decay phase
                shift_step = current_step - args.warmup_period
                remaining_steps = max_iterations - args.warmup_period
                return (1.0 - shift_step / remaining_steps) ** 0.9
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # No warmup, just use a constant LR or simple decay
        return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


def save_checkpoint(model, optimizer, epoch, loss, dsc, dir_checkpoint, is_best=False, is_periodic=False):
    """Save model checkpoint with metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'dsc': dsc,
    }
    
    if is_best:
        torch.save(checkpoint, os.path.join(dir_checkpoint, "checkpoint_best.pth"))
        print(f"Best model saved with DSC: {dsc:.4f}")
    
    if is_periodic:
        torch.save(checkpoint, os.path.join(dir_checkpoint, f"checkpoint_epoch_{epoch}.pth"))
    
    # Always save latest checkpoint
    torch.save(checkpoint, os.path.join(dir_checkpoint, "checkpoint_latest.pth"))


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint for resuming training."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dsc = checkpoint.get('dsc', 0)
        print(f"Resumed from epoch {start_epoch}, best DSC: {best_dsc:.4f}")
        return start_epoch, best_dsc
    else:
        print("No checkpoint found, starting from scratch")
        return 0, 0


def train_epoch(sam, trainloader, optimizer, scheduler, criterion1, criterion2, 
               writer, epoch, iter_num, args, scaler=None):
    """Train for one epoch."""
    sam.train()
    train_loss = 0
    current_iter = iter_num
    
    for i, data in enumerate(tqdm(trainloader, desc=f"Training Epoch {epoch}")):
        imgs = data["image"].cuda()
        msks = torchvision.transforms.Resize((args.out_size, args.out_size))(data["mask"])
        msks = msks.cuda()

        optimizer.zero_grad()
        
        # Use mixed precision if scaler is provided
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                if args.if_update_encoder:
                    img_emb = sam.image_encoder(imgs)
                else:
                    with torch.no_grad():
                        img_emb = sam.image_encoder(imgs)

                # Get default embeddings
                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=None, boxes=None, masks=None
                )
                pred, _ = sam.mask_decoder(
                    image_embeddings=img_emb,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                )
                loss, loss_dice, loss_ce = calculate_combined_loss(pred, msks, criterion1, criterion2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            scaler.update()
        else:
            if args.if_update_encoder:
                img_emb = sam.image_encoder(imgs)
            else:
                with torch.no_grad():
                    img_emb = sam.image_encoder(imgs)

            # Get default embeddings
            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            pred, _ = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )
            loss, loss_dice, loss_ce = calculate_combined_loss(pred, msks, criterion1, criterion2)
            
            loss.backward()
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            optimizer.step()


        train_loss += loss.item()
        current_iter += 1
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("info/lr", current_lr, current_iter)
        writer.add_scalar("info/total_loss", loss, current_iter)
        writer.add_scalar("info/loss_ce", loss_ce, current_iter)
        writer.add_scalar("info/loss_dice", loss_dice, current_iter)

    train_loss /= (i + 1)
    return train_loss, current_iter


def validate_epoch(sam, valloader, criterion1, criterion2, args):
    """Validate for one epoch."""
    sam.eval()
    eval_loss = 0
    dsc = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(valloader, desc="Validating")):
            imgs = data["image"].cuda()
            msks = torchvision.transforms.Resize((args.out_size, args.out_size))(data["mask"])
            msks = msks.cuda()

            img_emb = sam.image_encoder(imgs)
            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            pred, _ = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
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

    eval_loss /= (i + 1)
    dsc /= (i + 1)
    return eval_loss, dsc


def train_model(trainloader, valloader, dir_checkpoint, epochs):
    """Main training function with modular design and improved features."""
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
            if "Adapter" not in n:  # only update parameters in adapter
                value.requires_grad = False
        print("if update encoder:", args.if_update_encoder)
        print("if image encoder adapter:", args.if_encoder_adapter)
        print("if mask decoder adapter:", args.if_mask_decoder_adapter)
        if args.if_encoder_adapter:
            print("added adapter layers:", args.encoder_adapter_depths)

    elif args.finetune_type == "vanilla" and not args.if_update_encoder:
        print("if update encoder:", args.if_update_encoder)
        for n, value in sam.image_encoder.named_parameters():
            value.requires_grad = False
    
    sam.to("cuda")

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        sam.parameters(),
        lr=b_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.1,
        amsgrad=False,
    )
    
    # Setup learning rate scheduler
    max_iterations = epochs * len(trainloader)
    scheduler = get_learning_rate_scheduler(optimizer, args, max_iterations)
    
    # Initialize mixed precision scaler if available and enabled
    use_amp = getattr(args, 'mixed_precision', True) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    # Initialize loss functions
    criterion1 = monai.losses.DiceLoss(
        sigmoid=True, squared_pred=True, to_onehot_y=True, reduction="mean"
    )
    criterion2 = nn.CrossEntropyLoss()

    # Initialize tracking variables
    iter_num = 0
    writer = SummaryWriter(dir_checkpoint + "/log")
    val_largest_dsc = 0
    last_update_epoch = 0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
        start_epoch, val_largest_dsc = load_checkpoint(sam, optimizer, args.resume_from_checkpoint)
        last_update_epoch = start_epoch
    
    # Add option to save periodic checkpoints every N epochs
    save_every_n_epochs = getattr(args, 'save_every_n_epochs', 10)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Mixed precision training: {'Enabled' if use_amp else 'Disabled'}")
    print(f"Learning rate scheduler: {'Warmup + Decay' if args.if_warmup else 'Constant'}")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")

    pbar = tqdm(range(start_epoch, epochs), desc="Training Progress")
    
    for epoch in pbar:
        # Training phase
        train_loss, iter_num = train_epoch(
            sam, trainloader, optimizer, scheduler, criterion1, criterion2,
            writer, epoch, iter_num, args, scaler
        )
        
        pbar.set_description(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # Validation phase (every 2 epochs)
        if epoch % 2 == 0:
            eval_loss, dsc = validate_epoch(sam, valloader, criterion1, criterion2, args)
            
            # Log validation metrics
            writer.add_scalar("eval/loss", eval_loss, epoch)
            writer.add_scalar("eval/dice", dsc, epoch)

            print(f"Eval Epoch {epoch} | Val Loss: {eval_loss:.4f} | DSC: {dsc:.4f}")
            
            # Save best model
            if dsc > val_largest_dsc:
                val_largest_dsc = dsc
                last_update_epoch = epoch
                print(f"New best DSC: {dsc:.4f}")
                save_checkpoint(sam, optimizer, epoch, eval_loss, dsc, dir_checkpoint, is_best=True)
            
            # Early stopping check
            elif (epoch - last_update_epoch) > 20:
                print("Early stopping: No improvement for 20 epochs")
                break
        
        # Periodic checkpoint saving
        if epoch % save_every_n_epochs == 0 and epoch > 0:
            save_checkpoint(sam, optimizer, epoch, train_loss, 0, dir_checkpoint, is_periodic=True)
        
        # Always save latest checkpoint
        save_checkpoint(sam, optimizer, epoch, train_loss, 0, dir_checkpoint)

    writer.close()
    print(f"Training completed. Best validation DSC: {val_largest_dsc:.4f}")
    return val_largest_dsc


if __name__ == "__main__":
    dataset_name = args.dataset_name
    print("train dataset: {}".format(dataset_name))
    train_img_list = args.train_img_list
    val_img_list = args.val_img_list

    num_workers = 8
    if_vis = True
    Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
    path_to_json = os.path.join(args.dir_checkpoint, "args.json")
    args_dict = vars(args)
    with open(path_to_json, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)
    print(args.targets)

    # Determine whether to use cached dataset
    use_cached = getattr(args, 'use_cached_dataset', False)
    dataset_class = CachedPublicDataset if use_cached else Public_dataset
    
    print(f"Using {'CachedPublicDataset' if use_cached else 'Public_dataset'} for data loading")

    train_dataset = dataset_class(
        args,
        args.img_folder,
        args.mask_folder,
        train_img_list,
        phase="train",
        targets=[args.targets],
        normalize_type="sam",
        if_prompt=False,
        use_cache=use_cached,
        cache_dir=getattr(args, 'cache_dir', None),
        num_workers_preprocessing=getattr(args, 'num_workers_preprocessing', None)
    )
    eval_dataset = dataset_class(
        args,
        args.img_folder,
        args.mask_folder,
        val_img_list,
        phase="val",
        targets=[args.targets],
        normalize_type="sam",
        if_prompt=False,
        use_cache=use_cached,
        cache_dir=getattr(args, 'cache_dir', None),
        num_workers_preprocessing=getattr(args, 'num_workers_preprocessing', None)
    )
    trainloader = DataLoader(
        train_dataset, batch_size=args.b, shuffle=True, num_workers=num_workers
    )
    valloader = DataLoader(
        eval_dataset, batch_size=args.b, shuffle=False, num_workers=num_workers
    )

    train_model(trainloader, valloader, args.dir_checkpoint, args.epochs)
