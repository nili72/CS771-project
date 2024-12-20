import os
import numpy as np
import cv2
import albumentations
import glob
import random
from PIL import Image
from torch.utils.data import Dataset




class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root, 
                 size=None, random_crop=False, interpolation="bicubic",
           
                 ):
        self.data_csv = data_csv
        self.data_root = data_root
        
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            
        }

        size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        
        
            processed = {"image": image,
                        
                         }

        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        
        return example


class Examples(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="data/sflckr_examples.txt",
                         data_root="data/sflckr_images",
                         
                         size=size, random_crop=random_crop, interpolation=interpolation)


class cropTrain(SegmentationBase): 
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"): 
        super().__init__(data_csv="/ocean/projects/mat240020p/nli1/839/train.txt",
                         data_root="/ocean/projects/mat240020p/nli1/839/augmented_data/", 
                         
                         size=size, random_crop=random_crop, interpolation=interpolation, 
                         )


class cropValidation(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="/ocean/projects/mat240020p/nli1/839/val.txt",
                         #'/jet/home/nli1/diffusion/loops/crops/val/val_256_crop_5_images.txt',#'data/kvasir/kvasir_eval.txt',
                         data_root="/ocean/projects/mat240020p/nli1/839/augmented_data/",

                         #'/jet/home/nli1/diffusion/loops/crops/val/masks/256_crop_5/',#'data/kvasir/masks',
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         )

