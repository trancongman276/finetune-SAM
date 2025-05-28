import os

import cfg
from models.sam import sam_model_registry

arch = "vit_hq_b"
target = "multi_all"
img_folder = "./datasets"
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

room_classes = [
    "Background",
    "Outdoor",
    "Wall",
    "Kitchen",
    "Living Room",
    "Bed Room",
    "Bath",
    "Entry",
    "Railing",
    "Storage",
    "Garage",
    "Undefined",
]
# icon_classes = [
#     "No Icon",
#     "Window",
#     "Door",
#     "Closet",
#     "Electrical Applience",
#     "Toilet",
#     "Sink",
#     "Sauna Bench",
#     "Fire Place",
#     "Bathtub",
#     "Chimney",
# ]


sam = sam_model_registry["vit_b"](
    args,
    checkpoint=os.path.join("ckpts/sam_hq_vit_b.pth"),
    num_classes=len(room_classes),
)

print(sam)