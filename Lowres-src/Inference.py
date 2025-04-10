import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.utils import save_image
from sklearn.metrics import f1_score, classification_report
import numpy as np
from Dataset import FTWDataset
from SimpleModel import FrozenResNet18
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from colorama import Fore, Back, Style, init

init(autoreset=True)

def load_and_process_image(img_path, transform, device='cuda'):
    """Load and preprocess a single image."""
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def get_probabilities_from_model(image_tensor, model):
    """Get planting and harvest probabilities from model."""
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.sigmoid(logits[0][1:])
        harvest_prob, planting_prob = probs.tolist()
    return harvest_prob, planting_prob

def get_image_scores(image_paths, model, transform, device='cuda'):
    """Process a list of images and return sorted planting and harvest scores."""
    model = model.to(device)
    planting_scores, harvest_scores = [], []
    
    for img_path in image_paths:
        image_tensor = load_and_process_image(img_path, transform, device)
        harvest_prob, planting_prob = get_probabilities_from_model(image_tensor, model)
        
        planting_scores.append((img_path, planting_prob))        
        harvest_scores.append((img_path, harvest_prob))
    
    planting_scores.sort(key=lambda x: x[1], reverse=True)
    harvest_scores.sort(key=lambda x: x[1], reverse=True)
    return planting_scores, harvest_scores

def find_paired_probability(img_path, score_list):
    """Find the probability for a specific image path in a list of scores."""
    for path, prob in score_list:
        if path == img_path:
            return prob
    return 0

def classify_image(planting_prob, harvest_prob):
    """Classify image based on planting and harvest probabilities."""
    if planting_prob > 0.7 and harvest_prob < 0.3:
        return "Strong planting signal"
    elif harvest_prob > 0.7 and planting_prob < 0.3:
        return "Strong harvest signal"
    elif planting_prob > 0.5 and harvest_prob > 0.5:
        return "Both signals present"
    elif planting_prob < 0.3 and harvest_prob < 0.3:
        return "Weak signals overall"
    else:
        return "Mixed signals"

def get_comment_color(comment):
    """Return appropriate color code based on the comment."""
    if "Strong planting" in comment:
        return Fore.GREEN
    elif "Strong harvest" in comment:
        return Fore.BLUE
    elif "Both signals" in comment:
        return Fore.MAGENTA
    elif "Weak signals" in comment:
        return Fore.YELLOW
    else:
        return Fore.WHITE

def get_probability_color(prob):
    """Return appropriate color code based on probability value."""
    if prob > 0.7:
        return Fore.GREEN
    elif prob > 0.5:
        return Fore.CYAN
    elif prob > 0.3:
        return Fore.YELLOW
    else:
        return Fore.RED

def print_section_header(title):
    """Print a formatted section header."""
    border = "=" * (len(title) + 10)
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{border}")
    print(f"{Fore.CYAN}{Style.BRIGHT}===  {title}  ===")
    print(f"{Fore.CYAN}{Style.BRIGHT}{border}{Style.RESET_ALL}")

def print_top_images(scores, paired_scores, category, tile, top_n=5):
    """Print table of top images for a category (planting or harvest)."""
    title = f"Top {top_n} {category.upper()} images for MGRS tile {tile}"
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}{title}")
    
    if category == "planting":
        print(f"{Fore.WHITE}{'Image':<25} {Fore.GREEN}{'Planting Prob':<15} {Fore.BLUE}{'Harvest Prob':<15} {'Comment':<30}")
    else:
        print(f"{Fore.WHITE}{'Image':<25} {Fore.BLUE}{'Harvest Prob':<15} {Fore.GREEN}{'Planting Prob':<15} {'Comment':<30}")
    
    print(f"{Style.DIM}{'-' * 85}{Style.RESET_ALL}")
    
    for i, (img_path, primary_prob) in enumerate(scores[:top_n]):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        secondary_prob = find_paired_probability(img_path, paired_scores)
        
        if category == "planting":
            planting_prob, harvest_prob = primary_prob, secondary_prob
        else:
            harvest_prob, planting_prob = primary_prob, secondary_prob
            
        comment = classify_image(planting_prob, harvest_prob)
        comment_color = get_comment_color(comment)
        
        primary_color = get_probability_color(primary_prob)
        secondary_color = get_probability_color(secondary_prob)
        
        # Format with colors
        if category == "planting":
            print(f"{Fore.WHITE}{base_name:<25} {primary_color}{primary_prob:.4f}{Style.RESET_ALL}        {secondary_color}{secondary_prob:.4f}{Style.RESET_ALL}        {comment_color}{comment:<30}{Style.RESET_ALL}")
        else:
            print(f"{Fore.WHITE}{base_name:<25} {primary_color}{primary_prob:.4f}{Style.RESET_ALL}        {secondary_color}{secondary_prob:.4f}{Style.RESET_ALL}        {comment_color}{comment:<30}{Style.RESET_ALL}")

def analyze_mgrs_tiles(mgrs_groups, model, transform, top_pictures=5, max_tiles=3):
    """Analyze and display results for a set of MGRS tiles."""
    for i, tile in enumerate(list(mgrs_groups.keys())[:max_tiles]):
        # Clear visual separation between tiles
        if i > 0:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}{'*' * 100}{Style.RESET_ALL}")
        
        print_section_header(f"ANALYSIS FOR MGRS TILE: {tile}")
        
        planting_scores, harvest_scores = get_image_scores(mgrs_groups[tile], model, transform)
        
        print_top_images(planting_scores, harvest_scores, "planting", tile, top_pictures)
        
        # Visual separator between planting and harvest sections
        print(f"\n{Fore.CYAN}{Style.DIM}{'-' * 50}{Style.RESET_ALL}")
        
        print_top_images(harvest_scores, planting_scores, "harvest", tile, top_pictures)

def main():
    model = FrozenResNet18(num_classes=3)
    model_path = '/home/p.vinh/Auto-FTW/Lowres-src/predictions/both/resnet18/trial_19/best_model.pth'
    state_dict = torch.load(model_path, map_location=torch.device('cuda'))
    model.load_state_dict(state_dict)
    model.eval()
    
    dataset = FTWDataset(path_to_gt='/home/p.vinh/Auto-FTW/combined.csv', 
                        path_to_images='/data/p.vinh/Auto-FTW/Lowres-Images')
    
    mgrs_groups = dataset.group_by_mgrs_tiles()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),          
        transforms.Normalize(            
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 100}")
    print(f"{Fore.CYAN}{Style.BRIGHT}=== AGRICULTURAL IMAGE ANALYSIS - PLANTING & HARVEST DETECTION ===")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 100}{Style.RESET_ALL}")
    
    top_pictures = 5
    analyze_mgrs_tiles(mgrs_groups, model, transform, top_pictures)
    
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 100}")
    print(f"{Fore.CYAN}{Style.BRIGHT}=== ANALYSIS COMPLETE ===")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 100}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()