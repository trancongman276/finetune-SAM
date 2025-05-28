from pathlib import Path

from torch.utils.data import DataLoader

import cfg
from utils.dataset import Public_dataset

arch = "vit_b"
target = "multi_all"
img_folder = "/home/admin/doku/datasets"
dataset_name = "cubicasa5k"
finetune_type = "adapter"
train_img_list = f"{img_folder}/{dataset_name}/train.csv"
val_img_list = f"{img_folder}/{dataset_name}/val.csv"
dir_checkpoint = f"2D-SAM_{arch}_encoderdecoder_{finetune_type}_{dataset_name}_noprompt"
args = cfg.parse_args(
    [
        "-if_warmup",
        "True",
        "-finetune_type",
        finetune_type,
        "-arch",
        arch,
        "-if_mask_decoder_adapter",
        "True",
        "-img_folder",
        img_folder,
        "-mask_folder",
        img_folder,
        "-sam_ckpt",
        "ckpts/sam_hq_vit_b.pth",
        "-targets",
        target,
        "-dataset_name",
        dataset_name,
        "-dir_checkpoint",
        dir_checkpoint,
        "-train_img_list",
        train_img_list,
        "-val_img_list",
        val_img_list,
    ]
)


print("train dataset: {}".format(args.dataset_name))

num_workers = 8
if_vis = True
Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

# train_dataset = Public_dataset(
#     args,
#     args.img_folder,
#     args.mask_folder,
#     train_img_list,
#     phase="train",
#     targets=[args.targets],
#     normalize_type="sam",
#     if_prompt=False,
# )
eval_dataset = Public_dataset(
    args,
    args.img_folder,
    args.mask_folder,
    val_img_list,
    phase="val",
    targets=[args.targets],
    normalize_type="sam",
    if_prompt=False,
)
# trainloader = DataLoader(
#     train_dataset, batch_size=args.b, shuffle=True, num_workers=num_workers
# )
valloader = DataLoader(
    eval_dataset, batch_size=args.b, shuffle=False, num_workers=num_workers
)

# print("train dataset size: {}".format(len(train_dataset)))
print("val dataset size: {}".format(len(eval_dataset)))
# print("train dataset: {}".format(train_dataset))
print("val dataset: {}".format(eval_dataset))

# print("Example item from train dataset:")
# for i, item in enumerate(trainloader):
#     if i == 0:
#         print(item)
#         break
print("Example item from val dataset:")
item = next(iter(valloader))
print(item)
print("Data loading completed successfully.")