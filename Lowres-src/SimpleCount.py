import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, classification_report

class WhitePixelCountDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_gt, path_to_images, threshold=0.95):
        super().__init__()
        self.path_to_gt = path_to_gt
        self.path_to_images = path_to_images
        self.true_labels = pd.read_csv(self.path_to_gt)['s2_scene_id'].unique()
        self.images = []
        self.threshold = threshold

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()  
        ])

        for file in os.listdir(self.path_to_images):
            if file.lower().endswith('png'):
                self.images.append(os.path.join(self.path_to_images, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)  # shape: (1, 224, 224)
        
        white_count = (image > self.threshold).float().sum()  # Scalar count
        feature = white_count.view(1)  # Ensure shape is (1,)
        
        scene_id = os.path.basename(img_path).split('.')[0]
        label = torch.tensor(int(scene_id in self.true_labels))
        
        return feature, label

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature → one output logit

    def forward(self, x):
        return self.linear(x)  # Return raw logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_dataset = WhitePixelCountDataset(
    path_to_gt='/home/p.vinh/Auto-FTW/latvia.csv',
    path_to_images='/home/p.vinh/Auto-FTW/Lowres-Images',
    threshold=0.95  # Adjust as needed
)

test_ratio = 0.4
total_size = len(full_dataset)
test_size = int(total_size * test_ratio)
train_size = total_size - test_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

model = LogisticRegressionModel().to(device)
criterion = nn.BCEWithLogitsLoss()  # Combines a Sigmoid with binary cross entropy loss.
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

print("Starting training...\n")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_train_preds = []
    all_train_labels = []

    for features, labels in train_loader:
        features = features.to(device)  # shape: (batch_size, 1)
        labels = labels.float().to(device).unsqueeze(1)  # shape: (batch_size, 1)
        
        optimizer.zero_grad()
        outputs = model(features)  # Raw logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).int().cpu().numpy().flatten()
        all_train_preds.extend(preds)
        all_train_labels.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / train_size
    train_f1 = f1_score(all_train_labels, all_train_preds)
    print(f'Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}, Training F1 Score: {train_f1:.4f}')
    
    model.eval()
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            outputs = model(features)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).int().cpu().numpy().flatten()
            all_test_preds.extend(preds)
            all_test_labels.extend(labels.cpu().numpy().flatten())

    test_f1 = f1_score(all_test_labels, all_test_preds)
    print(f'Epoch [{epoch+1}/{num_epochs}] Test F1 Score: {test_f1:.4f}\n')

print("Final Evaluation on Test Set:")
print(classification_report(all_test_labels, all_test_preds, digits=4))