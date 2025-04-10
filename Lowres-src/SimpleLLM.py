from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from Dataset import FTWDataset
import wandb
import argparse
from sklearn.metrics import classification_report
from tqdm import tqdm
import geopy.geocoders
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='both')
parser.add_argument('--gt', type=str, default='/home/p.vinh/Auto-FTW/combined.csv')
parser.add_argument('--images', type=str, default='/data/p.vinh/Auto-FTW/Lowres-Images')
args = parser.parse_args()

def get_location_name(coordinate):
    print(coordinate)
    geolocator = Nominatim(user_agent="mgrs_region_finder")
    lat, lon = coordinate
    location = geolocator.reverse((lat, lon), language="en")
    if location == None:
        return ''
    address  = location.raw.get('address', {})
    region   = address.get('state') or address.get('county') or address.get('region')
    country  = address.get('country')
    return f'{region}, {country}'

def convert_date_to_string(date):
    return f'{date[0]}-{date[1]}-{date[2]}'



llava_model ="llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(llava_model)

model = LlavaNextForConditionalGeneration.from_pretrained(
    llava_model,
    device_map="auto", 
    torch_dtype=torch.float16
)


task_prompt = args.task

if args.task == 'both':
    task_prompt = 'haversting or planting'



dataset = FTWDataset(
    path_to_gt=args.gt,
    path_to_images=args.images,
    task = args.task
)

predictions = []
true_labels = []

for image, label, scene_id, date_info, latlon in tqdm(dataset):
    print(get_location_name(latlon))
    print(date_info)
    date_str = convert_date_to_string(date_info)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : f"This picture was taken at {date_str}, location: {latlon}. Is this picture depicting {task_prompt} day?. Answer in one word: Yes or No"},
                {"type": "image"}
            ]
        }
    ]


    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=10)
    LLM_prediction = processor.decode(output[0], skip_special_tokens=True).split(' ')[-2]
    pred = 1 if LLM_prediction == "Yes" else 0
    predictions.append(pred)
    true_labels.append(label)


    

report = classification_report(true_labels, predictions, target_names=["No", "Yes"])
print("\nClassification Report:")
print(report)