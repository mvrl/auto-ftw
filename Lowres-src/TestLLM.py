import os
import json
import torch
import argparse
from datasets import Dataset
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    LlavaNextConfig,
    Trainer, 
    TrainingArguments,
    DefaultDataCollator
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from PIL import Image
from sklearn.model_selection import train_test_split
from Dataset import FTWDataset
import wandb

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='both')
parser.add_argument('--gt', type=str, default='/home/p.vinh/Auto-FTW/combined.csv')
parser.add_argument('--images', type=str, default='/data/p.vinh/Auto-FTW/Lowres-Images')
parser.add_argument('--output_dir', type=str, default='./llava-ftw-lora')
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--per_device_train_batch_size', type=int, default=4)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--lora_rank', type=int, default=16)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.05)
args = parser.parse_args()

# Setup WandB for logging
wandb.init(project="llava-ftw-lora", config=vars(args))

# Helper functions from your code
def convert_date_to_string(date):
    return f'{date[0]}-{date[1]}-{date[2]}'

def get_location_name(coordinate):
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="mgrs_region_finder")
        lat, lon = coordinate
        location = geolocator.reverse((lat, lon), language="en")
        if location is None:
            return 'Unknown location'
        address = location.raw.get('address', {})
        region = address.get('state') or address.get('county') or address.get('region') or 'Unknown region'
        country = address.get('country') or 'Unknown country'
        return f'{region}, {country}'
    except Exception:
        return 'Unknown location'

# First prepare the dataset before loading the model
print("Preparing dataset...")

# Load the dataset
dataset = FTWDataset(
    path_to_gt=args.gt,
    path_to_images=args.images,
    task=args.task
)

task_prompt = args.task
if args.task == 'both':
    task_prompt = 'harvesting or planting'

# Prepare the data for fine-tuning
def prepare_data_for_training():
    data_items = []
    
    for idx, (image_path, label, scene_id, date_info, latlon) in enumerate(dataset):
        date_str = convert_date_to_string(date_info)
        location = get_location_name(latlon)
        
        # Create the conversation for fine-tuning
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This picture was taken at {date_str}, location: {location}. Is this picture depicting {task_prompt} day?"},
                    {"type": "image_url", "image_url": {"url": f"file:{image_path}"}}
                ]
            },
            {
                "role": "assistant", 
                "content": "Yes" if label == 1 else "No"
            }
        ]
        
        data_items.append({
            "id": str(scene_id) if scene_id else f"example_{idx}",
            "image": image_path,
            "conversations": conversation,
            "label": label
        })
    
    return data_items

# Process the data
all_data = prepare_data_for_training()
train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)

# Save the conversation data to files
os.makedirs("./finetune_data", exist_ok=True)
with open("./finetune_data/train.json", "w") as f:
    json.dump(train_data, f)
with open("./finetune_data/val.json", "w") as f:
    json.dump(val_data, f)

# Convert to HF Dataset format
def create_hf_dataset(data_items):
    dataset_dict = {
        "id": [],
        "image_path": [],
        "conversations": [],
        "label": []
    }
    
    for item in data_items:
        dataset_dict["id"].append(item["id"])
        dataset_dict["image_path"].append(item["image"])
        dataset_dict["conversations"].append(item["conversations"])
        dataset_dict["label"].append(item["label"])
    
    return Dataset.from_dict(dataset_dict)

train_dataset = create_hf_dataset(train_data)
val_dataset = create_hf_dataset(val_data)

print(f"Dataset prepared with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

# Load just the processor first - it's much smaller
print("Loading processor...")
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_name)

# Load model configuration only first
print("Loading model configuration...")
config = LlavaNextConfig.from_pretrained(
    model_name, 
    trust_remote_code=True
)

# Now load the model with specific parameters to avoid full preload
print("Loading model architecture...")
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,      # Reduces initial CPU memory usage
    offload_folder="offload",    # Offloads weights to disk temporarily
    offload_state_dict=True      # Offloads state dict to disk during loading
)

# Set up LoRA configuration
print("Configuring LoRA...")
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
    "gate_proj", "up_proj", "down_proj",     # MLP modules
    "lm_head",                               # Output head
]

# Special handling for vision-language projection layers
for name, param in model.named_parameters():
    if "multi_modal_projector" in name:
        target_modules.append(name.split(".")[-1])

# Configure LoRA
lora_config = LoraConfig(
    r=args.lora_rank,                       # Rank of LoRA matrices
    lora_alpha=args.lora_alpha,             # Scaling factor
    target_modules=target_modules,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Prepare the model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print percentage of trainable parameters

# Freeze vision encoder completely
for param in model.model.vision_tower.parameters():
    param.requires_grad = False

# Data preprocessing function - designed to process one example at a time to save memory
def process_single_example(example):
    try:
        # Get the image
        image_path = example["image_path"]
        image = Image.open(image_path).convert("RGB")
        
        # Extract conversation
        conv = example["conversations"]
        
        # Extract text prompt
        prompt_text = None
        for content in conv[0]["content"]:
            if content["type"] == "text":
                prompt_text = content["text"]
                break
        
        # Extract target (Yes/No)
        target = conv[1]["content"]
        
        # Tokenize inputs
        tokenized_input = processor(
            text=prompt_text,
            images=image,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize labels
        tokenized_target = processor.tokenizer(
            text=target,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Set -100 for padding tokens in targets to ignore them in loss calculation
        tokenized_target[tokenized_target == processor.tokenizer.pad_token_id] = -100
        
        # Combine inputs and targets
        processed_example = {
            "input_ids": tokenized_input.input_ids[0],
            "attention_mask": tokenized_input.attention_mask[0],
            "labels": tokenized_target[0]
        }
        
        if "pixel_values" in tokenized_input:
            processed_example["pixel_values"] = tokenized_input.pixel_values[0]
            
        return processed_example
        
    except Exception as e:
        print(f"Error processing example: {e}")
        return None

# Process datasets sequentially to save memory
def process_dataset(dataset):
    processed_examples = []
    
    for i in range(len(dataset)):
        example = dataset[i]
        processed = process_single_example(example)
        if processed:
            processed_examples.append(processed)
    
    return Dataset.from_list(processed_examples)

print("Processing datasets...")
processed_train_dataset = process_dataset(train_dataset)
processed_val_dataset = process_dataset(val_dataset)

# Custom data collator
class LlavaDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, examples):
        # Group by keys
        batch = {
            "input_ids": torch.stack([example["input_ids"] for example in examples]),
            "attention_mask": torch.stack([example["attention_mask"] for example in examples]),
            "labels": torch.stack([example["labels"] for example in examples]),
        }
        
        # Add pixel_values if present
        if "pixel_values" in examples[0]:
            batch["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
        
        return batch

# Training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",
    remove_unused_columns=False,  # Important for custom data processing
    # Memory optimization parameters
    gradient_checkpointing=True,  # Saves memory by recomputing gradients
    optim="adamw_torch",         # Memory-efficient optimizer
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_val_dataset,
    data_collator=LlavaDataCollator(processor),
)

# Fine-tune the model
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model - only save the LoRA adapter weights
print("Saving LoRA adapter...")
model.save_pretrained(args.output_dir)
processor.save_pretrained(args.output_dir)

print("Fine-tuning completed successfully!")