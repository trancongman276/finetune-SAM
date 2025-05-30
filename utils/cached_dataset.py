import os
import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import warnings
from tqdm.auto import tqdm

class CachedPublicDataset(Dataset):
    """
    Optimized version of Public_dataset with caching and faster loading.
    """
    
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
        delete_empty_masks=False,
        use_cache=True,
        cache_dir=None,
        num_workers_preprocessing=None
    ):
        super(CachedPublicDataset, self).__init__()
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
        self.if_spatial = if_spatial
        self.use_cache = use_cache
        
        # Set up caching
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(img_list), "cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up number of workers for preprocessing
        if num_workers_preprocessing is None:
            self.num_workers_preprocessing = min(8, mp.cpu_count())
        else:
            self.num_workers_preprocessing = num_workers_preprocessing
        
        # Load components
        self.load_label_mapping()
        self.load_data_list_optimized(img_list)
        self.setup_transformations()
        

    def _get_cache_path(self, data_key):
        """Generate cache file path for a data entry."""
        cache_filename = f"cache_{hash(data_key)}.pt"
        return os.path.join(self.cache_dir, cache_filename)


    def load_label_mapping(self):
        """Load label mapping from pickle file."""
        if self.label_mapping:
            with open(self.label_mapping, "rb") as handle:
                self.segment_names_to_labels = pickle.load(handle)
            self.label_dic = {seg[1]: seg[0] for seg in self.segment_names_to_labels}
            self.label_name_list = [seg[0] for seg in self.segment_names_to_labels]
        else:
            self.segment_names_to_labels = {}
            self.label_dic = {value: "all" for value in range(1, 256)}

    def load_data_list_optimized(self, img_list):
        """Optimized data list loading with parallel processing."""
        print(f"Loading data list from {img_list}...")
        
        with open(img_list, "r") as file:
            lines = file.read().strip().split("\n")
        
        print(f"Filtering {len(lines)} entries...")
        
        def check_entry(line):
            try:
                img_path, mask_path = line.split(",")
                mask_path = mask_path.strip()
                
                # Quick check if files exist
                if not os.path.exists(mask_path):
                    return None
                    
                if self.delete_empty_masks:
                    msk = np.load(mask_path)[0]
                    if self.should_keep(msk, mask_path):
                        return line
                else:
                    return line
            except Exception as e:
                warnings.warn(f"Error processing line {line}: {e}")
                return None
            return None
        
        # Use parallel processing for filtering
        with ThreadPoolExecutor(max_workers=self.num_workers_preprocessing) as executor:
            results = list(tqdm(
                executor.map(check_entry, lines),
                total=len(lines),
                desc="Filtering data"
            ))
        
        self.data_list = [result for result in results if result is not None]
        print(f"Filtered data list to {len(self.data_list)} entries.")

    def should_keep(self, msk, mask_path):
        """Determine whether to keep an image based on conditions."""
        if self.delete_empty_masks:
            mask_array = np.array(msk, dtype=int)
            
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
        """Setup image transformations."""
        if self.phase == "train":
            transformations = [
                transforms.RandomEqualize(p=0.1),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
                ),
            ]
            
            if self.if_spatial:
                self.transform_spatial = transforms.Compose([
                    transforms.RandomResizedCrop(
                        self.crop_size,
                        scale=(0.5, 1.5),
                        interpolation=InterpolationMode.NEAREST,
                    ),
                    transforms.RandomRotation(
                        45, interpolation=InterpolationMode.NEAREST
                    ),
                ])
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
        data_key = self.data_list[index]
        
        # Try to load from cache first
        # if self.use_cache:
        #     cached_data = self._load_cached_item(data_key)
        #     if cached_data is not None:
        #         img_array = cached_data['image']
        #         msk_array = cached_data['mask']
        #         img_path = cached_data['img_path']
        #         mask_path = cached_data['mask_path']
                
        #         # Convert back to PIL
        #         img = Image.fromarray(img_array)
        #         msk = Image.fromarray(msk_array)
        #     else:
        #         # Fallback to regular loading
        #         img_path, mask_path = data_key.split(",")
        #         img_path = img_path.strip()
        #         mask_path = mask_path.strip()
                
        #         img = np.load(img_path)
        #         msk = np.load(mask_path)[0]
                
        #         if img.dtype == np.float32 or img.dtype == np.float64:
        #             img = (img * 255).astype(np.uint8)
                
        #         img = Image.fromarray(img).convert("RGB")
        #         msk = Image.fromarray(msk).convert("L")
                
        #         img = transforms.Resize((self.args.image_size, self.args.image_size))(img)
        #         msk = transforms.Resize(
        #             (self.args.image_size, self.args.image_size), InterpolationMode.NEAREST
        #         )(msk)
        # else:
        # Regular loading without cache
        img_path, mask_path = data_key.split(",")
        img_path = img_path.strip()
        mask_path = mask_path.strip()
        
        img = np.load(img_path)
        msk = np.load(mask_path)[0]
        
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        
        img = Image.fromarray(img).convert("RGB")
        msk = Image.fromarray(msk).convert("L")
        
        img = transforms.Resize((self.args.image_size, self.args.image_size))(img)
        msk = transforms.Resize(
            (self.args.image_size, self.args.image_size), InterpolationMode.NEAREST
        )(msk)

        # Apply transformations
        img, msk = self.apply_transformations(img, msk)

        # Process mask based on targets
        if "combine_all" in self.targets:
            msk = np.array(np.array(msk, dtype=int) > 0, dtype=int)
        elif "multi_all" in self.targets:
            msk = np.array(msk, dtype=int)
        elif self.cls > 0:
            msk = np.array(msk == self.cls, dtype=int)

        return self.prepare_output(img, msk, img_path, mask_path)

    def apply_transformations(self, img, msk):
        """Apply transformations to image and mask."""
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
        """Apply random crop to image and mask."""
        t, l, h, w = transforms.RandomCrop.get_params(
            img, (self.crop_size, self.crop_size)
        )
        img = transforms.functional.crop(img, t, l, h, w)
        msk = transforms.functional.crop(msk, t, l, h, w)
        return img, msk

    def prepare_output(self, img, msk, img_path, mask_path):
        """Prepare final output dictionary."""
        if len(msk.shape) == 2:
            msk = torch.unsqueeze(torch.tensor(msk, dtype=torch.long), 0)
        
        output = {"image": img, "mask": msk, "img_name": img_path}
        
        # Add prompts if needed (simplified for performance)
        if self.if_prompt:
            # Note: For performance, you might want to pre-compute prompts during caching
            # This is left as-is for compatibility
            from utils.funcs import get_first_prompt, get_top_boxes
            
            if self.prompt_type == "point":
                prompt, mask_now = get_first_prompt(
                    msk.numpy()[0], region_type=self.region_type
                )
                pc = torch.tensor(prompt[:, :2], dtype=torch.float)
                pl = torch.tensor(prompt[:, -1], dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now, dtype=torch.long), 0)
                output.update({"point_coords": pc, "point_labels": pl, "mask": msk})
            elif self.prompt_type == "box":
                prompt, mask_now = get_top_boxes(
                    msk.numpy()[0], region_type=self.region_type
                )
                box = torch.tensor(prompt, dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now, dtype=torch.long), 0)
                output.update({"boxes": box, "mask": msk})
        
        return output
