import os
import pandas as pd
import torchvision.transforms as transforms
import torch
import mgrs
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class FTWDataset(Dataset):
    def __init__(self, path_to_gt, path_to_images, task='both'):
        super().__init__()
        self.task = task
        self.path_to_gt = path_to_gt
        self.path_to_images = path_to_images
        
        self.images = []
        dataframe = pd.read_csv(self.path_to_gt)
        
        # Create dictionaries to store season information for each scene_id
        self.harvest_labels = set(dataframe[dataframe['season'] == 'harvest']['s2_scene_id'])
        self.planting_labels = set(dataframe[dataframe['season'] == 'planting']['s2_scene_id'])
        
        if self.task == 'both':
            self.true_labels = self.harvest_labels.union(self.planting_labels)
        elif self.task in {'harvest', 'planting'}:
            self.true_labels = dataframe[dataframe['season'] == self.task]['s2_scene_id']
            
        self.true_labels = set(self.true_labels)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        for file in os.listdir(self.path_to_images):
            if file.lower().endswith('png'):
                self.images.append(os.path.join(self.path_to_images, file))
        
        files = set([os.path.basename(image_path).split('.')[0] for image_path in self.images])
        print(len(self.true_labels - files)/len(self.true_labels))
        
        if self.true_labels.issubset(files):
            print("True labels is actually a subset of the images")
        else:
            print("True labels is not a subset of the images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        scene_id = os.path.basename(img_path).split('.')[0]
        time = scene_id.split('_')[2].split('T')[0]
        year = int(time[:4])
        month = int(time[4:6])
        date = int(time[6:])
        
        if scene_id in self.harvest_labels:
            label = torch.tensor(1)
        elif scene_id in self.planting_labels:
            label = torch.tensor(2)
        else:
            label = torch.tensor(0)
        
        mgrs_tile = scene_id.split('_')[4][1:]
        (lat, lon) = mgrs.MGRS().toLatLon(mgrs_tile)
        
        return image, label, scene_id, (year, month, date), (lat, lon)
    
    def group_by_mgrs_tiles(self):
        mgrs_groups = {}
        for img_path in self.images:
            scene_id = os.path.basename(img_path).split('.')[0]
            mgrs_tile = scene_id.split('_')[4][1:]
            
            if mgrs_tile not in mgrs_groups:
                mgrs_groups[mgrs_tile] = []
                
            mgrs_groups[mgrs_tile].append(img_path)
            
        return mgrs_groups

    
        








