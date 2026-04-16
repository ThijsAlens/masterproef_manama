import torch
from torchvision import transforms
from PIL import Image

import json
import os


class CustomDataset(torch.utils.data.Dataset):
    """
    This class is used for loading the custom dataset for regression task.
    The dataset is organized in the following structure:
        ./source_dir_data/
            image_001_color.png
            image_001_depth.png
            image_001.json
        
    Each image has a corresponding JSON file with the following structure:
        {
            "pixel": {
                "x": 123,
                "y": 456
            }
            "world": {
                "x": 1.23,
                "y": 4.56
            }
        }

        Args:
            source_dir (str): The directory containing the dataset.
            include_depth (bool): Whether to include depth information as an additional channel (default: True).
            world (bool): Whether to use world coordinates as targets instead of pixel coordinates (default: True).
        """
    
    def __init__(self, source_dir: str, include_depth: bool = True, world: bool = True):
        self.include_depth = include_depth
        self.world = world
        self.data: list[tuple] = []
        for file in os.listdir(source_dir):
            if file.endswith(".json"):
                id = file[:-5]
                color_path = os.path.join(source_dir, f"{id}_color.png")
                depth_path = os.path.join(source_dir, f"{id}_depth.png")
                json_path = os.path.join(source_dir, f"{id}.json")
                if not os.path.exists(color_path): 
                    print(f"Warning: Missing color image for {id}, path: {color_path}")
                    continue
                if not os.path.exists(depth_path):
                    print(f"Warning: Missing depth image for {id}, path: {depth_path}")
                    continue
                if not os.path.exists(json_path):
                    print(f"Warning: Missing JSON file for {id}, path: {json_path}")
                    continue
                self.data.append((color_path, depth_path, json_path))
                

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        color_path, depth_path, json_path = self.data[idx]

        image = Image.open(color_path).convert("RGB")
        color_tensor = transforms.ToTensor()(image)
        if self.include_depth:
            depth_image = Image.open(depth_path).convert("L")
            depth_tensor = transforms.ToTensor()(depth_image)
            image = torch.cat((color_tensor, depth_tensor), dim=0)
        else:
            image = color_tensor

        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        if self.world:
            target = torch.tensor([json_data["world"]["x"], json_data["world"]["y"]], dtype=torch.float32)
        else:
            target = torch.tensor([json_data["pixel"]["x"], json_data["pixel"]["y"]], dtype=torch.float32)

        return image, target