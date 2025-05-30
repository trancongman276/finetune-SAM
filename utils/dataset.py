import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
from utils.funcs import get_first_prompt, get_top_boxes
from torchvision.transforms import InterpolationMode


class Public_dataset(Dataset):

    def __init__(
        self,
        args,
        img_folder,
        mask_folder,
        img_list,
        phase="train",
        sample_num=50,
        channel_num=1,
        normalize_type="sam",
        crop=False,
        crop_size=1024,
        targets=["femur", "hip"],
        part_list=["all"],
        cls=-1,
        if_prompt=True,
        prompt_type="point",
        region_type="largest_3",
        label_mapping=None,
        if_spatial=True,
        delete_empty_masks=True,
    ):
        """
        target: 'combine_all': combine all the targets into binary segmentation
                'multi_all': keep all targets as multi-cls segmentation
                f'{one_target_name}': segmentation specific one type of target, such as 'hip'

        normalzie_type: 'sam' or 'medsam', if sam, using transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]); if medsam, using [0,1] normalize
        cls: the target cls for segmentation
        prompt_type: point or box
        if_patial: if add spatial transformations or not

        """
        super(Public_dataset, self).__init__()
        self.args = args
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase
        self.normalize_type = normalize_type
        self.targets = targets
        self.part_list = part_list
        self.cls = cls
        self.delete_empty_masks = delete_empty_masks
        self.if_prompt = if_prompt
        self.prompt_type = prompt_type
        self.region_type = region_type
        self.label_dic = {}
        self.data_list = []
        self.label_mapping = label_mapping
        self.load_label_mapping()
        self.load_data_list(img_list)
        self.if_spatial = if_spatial
        self.setup_transformations()

    def load_label_mapping(self):
        # Load the predefined label mappings from a pickle file
        # the format is {'label_name1':cls_idx1, 'label_name2':,cls_idx2}
        if self.label_mapping:
            with open(self.label_mapping, "rb") as handle:
                self.segment_names_to_labels = pickle.load(handle)
            self.label_dic = {seg[1]: seg[0] for seg in self.segment_names_to_labels}
            self.label_name_list = [seg[0] for seg in self.segment_names_to_labels]
            print(self.label_dic)
        else:
            self.segment_names_to_labels = {}
            self.label_dic = {value: "all" for value in range(1, 256)}

    def load_data_list(self, img_list):
        """
        Load and filter the data list based on the existence of the mask and its relevance to the specified parts and targets.
        """
        with open(img_list, "r") as file:
            lines = file.read().strip().split("\n")
        for line in lines:
            img_path, mask_path = line.split(",")
            mask_path = mask_path.strip()
            # if mask_path.startswith("/"):
            #     mask_path = mask_path[1:]
            # msk = Image.open(os.path.join(self.mask_folder, mask_path)).convert("L")
            msk = np.load(mask_path)[0]
            if self.should_keep(msk, mask_path):
                self.data_list.append(line)

        print(f"Filtered data list to {len(self.data_list)} entries.")

    def should_keep(self, msk, mask_path):
        """
        Determine whether to keep an image based on the mask and part list conditions.
        """
        if self.delete_empty_masks:
            mask_array = np.array(msk, dtype=int)
            # print(np.unique(mask_array))
            if "combine_all" in self.targets:
                return np.any(mask_array > 0)
            elif "multi_all" in self.targets:
                return np.any(mask_array > 0)
            elif any(target in self.targets for target in self.segment_names_to_labels):
                target_classes = [
                    self.segment_names_to_labels[target][1]
                    for target in self.targets
                    if target in self.segment_names_to_labels
                ]
                return any(mask_array == cls for cls in target_classes)
            elif self.cls > 0:
                return np.any(mask_array == self.cls)
            if self.part_list[0] != "all":
                return any(part in mask_path for part in self.part_list)
            return False
        else:
            return True

    def setup_transformations(self):
        if self.phase == "train":
            transformations = [
                transforms.RandomEqualize(p=0.1),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
                ),
            ]
            # if add spatial transform
            if self.if_spatial:
                self.transform_spatial = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            self.crop_size,
                            scale=(0.5, 1.5),
                            interpolation=InterpolationMode.NEAREST,
                        ),
                        transforms.RandomRotation(
                            45, interpolation=InterpolationMode.NEAREST
                        ),
                    ]
                )
        else:
            transformations = []
        transformations.append(transforms.ToTensor())
        if self.normalize_type == "sam":
            transformations.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )
        elif self.normalize_type == "medsam":
            transformations.append(
                transforms.Lambda(
                    lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                )
            )
        self.transform_img = transforms.Compose(transformations)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        img_path, mask_path = data.split(",")
        # if mask_path.startswith("/"):
        #     mask_path = mask_path[1:]
        # img = Image.open(os.path.join(self.img_folder, img_path.strip())).convert("RGB")
        # msk = Image.open(os.path.join(self.mask_folder, mask_path.strip())).convert("L")
        img = np.load(img_path.strip())
        msk = np.load(mask_path.strip())[0]

        # Scale img back to [0, 255] if necessary
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)

        img = Image.fromarray(img).convert("RGB")
        msk = Image.fromarray(msk).convert("L")

        img = transforms.Resize((self.args.image_size, self.args.image_size))(img)
        msk = transforms.Resize(
            (self.args.image_size, self.args.image_size), InterpolationMode.NEAREST
        )(msk)

        img, msk = self.apply_transformations(img, msk)

        if "combine_all" in self.targets:  # combine all targets as single target
            msk = np.array(np.array(msk, dtype=int) > 0, dtype=int)
        elif "multi_all" in self.targets:
            msk = np.array(msk, dtype=int)
        elif self.cls > 0:
            msk = np.array(msk == self.cls, dtype=int)
        return self.prepare_output(img, msk, img_path, mask_path)

    def apply_transformations(self, img, msk):
        if self.crop:
            img, msk = self.apply_crop(img, msk)
        img = self.transform_img(img)
        msk = torch.tensor(np.array(msk, dtype=int), dtype=torch.long)

        if self.phase == "train" and self.if_spatial:
            mask_cls = np.array(msk, dtype=int)
            mask_cls = np.repeat(mask_cls[np.newaxis, :, :], 3, axis=0)
            both_targets = torch.cat(
                (img.unsqueeze(0), torch.tensor(mask_cls).unsqueeze(0)), 0
            )
            transformed_targets = self.transform_spatial(both_targets)
            img = transformed_targets[0]
            mask_cls = np.array(transformed_targets[1][0].detach(), dtype=int)
            msk = torch.tensor(mask_cls)
        return img, msk

    def apply_crop(self, img, msk):
        t, l, h, w = transforms.RandomCrop.get_params(
            img, (self.crop_size, self.crop_size)
        )
        img = transforms.functional.crop(img, t, l, h, w)
        msk = transforms.functional.crop(msk, t, l, h, w)
        return img, msk

    def prepare_output(self, img, msk, img_path, mask_path):
        if len(msk.shape) == 2:
            msk = torch.unsqueeze(torch.tensor(msk, dtype=torch.long), 0)
        output = {"image": img, "mask": msk, "img_name": img_path}
        if self.if_prompt:
            # Assuming get_first_prompt and get_top_boxes functions are defined and handle prompt creation
            if self.prompt_type == "point":
                prompt, mask_now = get_first_prompt(
                    msk.numpy()[0], region_type=self.region_type
                )
                pc = torch.tensor(prompt[:, :2], dtype=torch.float)
                pl = torch.tensor(prompt[:, -1], dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now, dtype=torch.long), 0)
                output.update({"point_coords": pc, "point_labels": pl, "mask": msk})
            elif self.prompt_type == "box":
                print(msk.shape, msk.numpy()[0].shape)
                prompt, mask_now = get_top_boxes(
                    msk.numpy()[0], region_type=self.region_type
                )
                box = torch.tensor(prompt, dtype=torch.float)
                # the ground truth are only the selected masks
                msk = torch.unsqueeze(torch.tensor(mask_now, dtype=torch.long), 0)
                output.update({"boxes": box, "mask": msk})
            elif self.prompt_type == "hybrid":
                point_prompt, _ = get_first_prompt(msk[0].numpy(), self.region_type)
                box_prompt, _ = get_top_boxes(msk.numpy(), self.region_type)
                pc = torch.tensor(point_prompt[:, :2], dtype=torch.float)
                pl = torch.tensor(point_prompt[:, -1], dtype=torch.float)
                box = torch.tensor(box_prompt, dtype=torch.float)
                output.update({"point_coords": pc, "point_labels": pl, "boxes": box})
        return output
