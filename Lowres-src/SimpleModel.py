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
import wandb
import random
from Dataset import FTWDataset


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', num_classes=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        
        if num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        if self.num_classes == 1:
            BCE_loss = self.criterion(inputs, targets)
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        else:
            CE_loss = self.criterion(inputs, targets)
            pt = torch.exp(-CE_loss)
            F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class FrozenResNet18(nn.Module):
    def __init__(self, pretrained=True, dropout=0.2, num_classes=1):
        super(FrozenResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.num_classes = num_classes
        
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        in_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, image):
        return self.resnet(image)


class FrozenVGG11(nn.Module):
    def __init__(self, pretrained=True, dropout=0.2, num_classes=1):
        super(FrozenVGG11, self).__init__()
        self.vgg = models.vgg11(pretrained=pretrained)
        self.num_classes = num_classes
        
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        in_features = self.vgg.classifier[6].in_features
        
        self.vgg.classifier[6] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.vgg.classifier[6].parameters():
            param.requires_grad = True

    def forward(self, image):
        return self.vgg(image)


class FrozenVisionTransformer(nn.Module):
    def __init__(self, pretrained=True, dropout=0.2, num_classes=1):
        super(FrozenVisionTransformer, self).__init__()
        self.vit = models.vit_b_16(pretrained=pretrained)
        self.num_classes = num_classes
        
        for param in self.vit.parameters():
            param.requires_grad = False
            
        in_features = self.vit.heads.head.in_features
        
        self.vit.heads.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.vit.heads.head.parameters():
            param.requires_grad = True

    def forward(self, image):
        return self.vit(image)


class FrozenSwinTransformer(nn.Module):
    def __init__(self, pretrained=True, dropout=0.2, num_classes=1):
        super(FrozenSwinTransformer, self).__init__()
        self.swin = models.swin_t(pretrained=pretrained)
        self.num_classes = num_classes
        
        for param in self.swin.parameters():
            param.requires_grad = False
            
        in_features = self.swin.head.in_features
        
        self.swin.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.swin.head.parameters():
            param.requires_grad = True

    def forward(self, image):
        return self.swin(image)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-ratio', type=float, default=0.4)
    parser.add_argument('--gt', type=str, default='/home/p.vinh/Auto-FTW/combined.csv')
    parser.add_argument('--images', type=str, default='/data/p.vinh/Auto-FTW/Lowres-Images')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--save-dir', type=str, default='predictions')
    parser.add_argument('--trials', type=int, default=20, 
                        help='Number of random hyperparameter trials per model')
    parser.add_argument('--wandb-project', type=str, default='FTW-Classification',
                        help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='WandB entity (username or team name)')
    return parser.parse_args()


def prepare_data(args, task):
    full_dataset = FTWDataset(path_to_gt=args.gt, path_to_images=args.images, task=task)
    
    total_size = len(full_dataset)
    test_size = int(total_size * args.test_ratio)
    train_size = total_size - test_size
    
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    from collections import Counter
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    test_labels = [test_dataset[i][1].item() for i in range(len(test_dataset))]
    
    print("Train set label distribution:", Counter(train_labels))
    print("Test set label distribution:", Counter(test_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, train_size, test_size


def train_epoch(model, train_loader, criterion, optimizer, device, train_size, num_classes):
    model.train()
    running_loss = 0.0
    all_train_labels = []
    all_train_preds = []
    
    for images, labels, _, _, _ in train_loader:
        images = images.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if num_classes == 1:
            labels = labels.float().to(device).unsqueeze(1)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int().cpu().numpy().flatten()
        else:
            labels = labels.long().to(device)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        all_train_labels.extend(labels.cpu().numpy().flatten())
        all_train_preds.extend(preds)
    
    epoch_loss = running_loss / train_size
    
    return epoch_loss, all_train_labels, all_train_preds


def evaluate(model, loader, criterion, device, data_size, num_classes, is_final_epoch=False, save_dir=None):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    all_ids = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, ids, _, _) in enumerate(loader):
            images = images.to(device)
            
            outputs = model(images)
            
            if num_classes == 1:
                labels = labels.float().to(device).unsqueeze(1)
                loss = criterion(outputs, labels)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).int().cpu().numpy().flatten()
                probs_np = probs.cpu().numpy().flatten()
            else:
                labels = labels.long().to(device)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                probs_np = probs.cpu().numpy()  # Shape: [batch_size, num_classes]
            
            total_loss += loss.item() * images.size(0)
            
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds)
            all_probs.append(probs_np)
            all_ids.extend(ids)
            
            if is_final_epoch and save_dir:
                os.makedirs(save_dir, exist_ok=True)
                if num_classes == 1:
                    for i, (pred, label) in enumerate(zip(preds, labels.cpu().numpy().flatten())):
                        if pred == 1:  # Save only positive predictions
                            img = images[i].cpu()
                            filename = f"{save_dir}/pos_pred_{batch_idx}_{i}_true_{int(label)}.png"
                            save_image(img, filename)
                else:
                    for i, (pred, label) in enumerate(zip(preds, labels.cpu().numpy().flatten())):
                        if pred > 0:  # Save non-"neither" predictions
                            img = images[i].cpu()
                            class_name = "harvest" if pred == 1 else "planting"
                            true_class = "harvest" if label == 1 else "planting" if label == 2 else "neither"
                            filename = f"{save_dir}/pred_{class_name}_true_{true_class}_{batch_idx}_{i}.png"
                            save_image(img, filename)
    
    avg_loss = total_loss / data_size
    
    if num_classes == 1:
        all_probs = np.concatenate(all_probs)
    else:
        all_probs = np.vstack(all_probs)
    
    return avg_loss, all_labels, all_preds, all_probs, all_ids


def get_f1_score(labels, preds, num_classes=1):
    if num_classes == 1:
        return f1_score(labels, preds)
    else:
        return f1_score(labels, preds, average='macro')


def get_model(model_name, dropout, num_classes):
    if model_name == 'resnet18':
        return FrozenResNet18(pretrained=True, dropout=dropout, num_classes=num_classes)
    elif model_name == 'vgg11':
        return FrozenVGG11(pretrained=True, dropout=dropout, num_classes=num_classes)
    elif model_name == 'vit':
        return FrozenVisionTransformer(pretrained=True, dropout=dropout, num_classes=num_classes)
    elif model_name == 'swin':
        return FrozenSwinTransformer(pretrained=True, dropout=dropout, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate_model(model_name, task, args, train_loader, test_loader, train_size, test_size, gamma, alpha, trial_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = 3 if task == 'both' else 1
    
    run_name = f"{model_name}_task_{task}_g{gamma:.2f}_a{alpha:.2f}_trial{trial_num}"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model": model_name,
            "task": task,
            "gamma": gamma,
            "alpha": alpha,
            "lr": args.lr,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "num_classes": num_classes,
            "trial": trial_num
        },
        reinit=True,
    )
    
    model_save_dir = os.path.join(args.save_dir, task, model_name, f"trial_{trial_num}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"\n------ Training {model_name} for task '{task}' with gamma={gamma:.2f}, alpha={alpha:.2f}, trial={trial_num} ------")
    
    model = get_model(model_name, args.dropout, num_classes).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    
    best_test_f1 = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        is_final_epoch = (epoch == args.epochs - 1)
        
        train_loss, train_labels, train_preds = train_epoch(
            model, train_loader, criterion, optimizer, device, train_size, num_classes
        )
        
        train_f1 = get_f1_score(train_labels, train_preds, num_classes)
        
        test_loss, test_labels, test_preds, test_probs, test_ids = evaluate(
            model, test_loader, criterion, device, test_size, num_classes,
            is_final_epoch, 
            os.path.join(model_save_dir, f"epoch_{epoch+1}") if is_final_epoch else None
        )
        
        test_f1 = get_f1_score(test_labels, test_preds, num_classes)
        
        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "test_loss": test_loss,
            "test_f1": test_f1,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        
        if num_classes > 1:
            test_report = classification_report(test_labels, test_preds, output_dict=True)
            
            for class_id in range(num_classes):
                class_name = f"class_{class_id}"
                if str(class_id) in test_report:
                    log_data[f"test_f1_{class_name}"] = test_report[str(class_id)]['f1-score']
                    log_data[f"test_precision_{class_name}"] = test_report[str(class_id)]['precision']
                    log_data[f"test_recall_{class_name}"] = test_report[str(class_id)]['recall']
        
        wandb.log(log_data)
        
        print(f'Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'Epoch [{epoch+1}/{args.epochs}] Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}')
        
        scheduler.step(test_loss)
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_epoch = epoch + 1
            model_save_path = os.path.join(model_save_dir, f'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            wandb.log({"best_model_saved": epoch + 1})
    
    model.load_state_dict(torch.load(os.path.join(model_save_dir, f'best_model.pth')))
    final_loss, final_labels, final_preds, final_probs, final_ids = evaluate(
        model, test_loader, criterion, device, test_size, num_classes, True, 
        os.path.join(model_save_dir, f"final_best")
    )
    
    report = classification_report(final_labels, final_preds, output_dict=True)
    
    wandb.log({
        "final_test_loss": final_loss,
        "final_test_f1": report['macro avg']['f1-score'],
        "final_test_precision": report['macro avg']['precision'],
        "final_test_recall": report['macro avg']['recall'],
        "best_epoch": best_epoch
    })
    
    wandb.sklearn.plot_confusion_matrix(final_labels, final_preds)
    
    if num_classes > 1:
        print("\nClassification Report:")
        print(classification_report(final_labels, final_preds))
        
        for class_id in range(1, num_classes):  # Skip 'neither' class (0)
            class_name = "harvest" if class_id == 1 else "planting"
            print(f"\nTop 5 {class_name} predictions:")
            
            combined = list(zip(final_ids, final_probs[:, class_id]))
            top_5 = sorted(combined, key=lambda x: x[1], reverse=True)[:5]
            
            for scene_id, prob in top_5:
                print(f"ID: {scene_id}, {class_name} Probability: {prob:.4f}")
    
    run.finish()
    
    return {
        'model': model_name,
        'task': task,
        'gamma': gamma,
        'alpha': alpha,
        'trial': trial_num,
        'f1_score': best_test_f1,
        'best_epoch': best_epoch,
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall']
    }


def main():
    args = parse_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    
    tasks = ['both']
    
    models_to_test = ['resnet18', 'vgg11', 'vit', 'swin']
    
    gamma_range = (0.5, 5.0)  # min, max
    alpha_range = (0.25, 1.0)  # min, max
    
    all_results = []
    
    print("\nExperiment plan:")
    print(f"{'Task':<10} {'Model':<10} {'Trials':<10}")
    print("-" * 30)
    for task in tasks:
        for model_name in models_to_test:
            print(f"{task:<10} {model_name:<10} {args.trials:<10}")
    print(f"\nTotal number of experiments: {len(tasks) * len(models_to_test) * args.trials}")
    
    for task in tasks:
        print(f"\n\n{'='*80}")
        print(f"Starting experiments for task: {task}")
        print(f"{'='*80}")
        
        train_loader, test_loader, train_size, test_size = prepare_data(args, task)
        
        task_results = []
        
        for model_name in models_to_test:
            model_results = []
            
            for trial in range(1, args.trials + 1):
                gamma = random.uniform(gamma_range[0], gamma_range[1])
                alpha = random.uniform(alpha_range[0], alpha_range[1])
                
                result = train_and_evaluate_model(
                    model_name, task, args, train_loader, test_loader, 
                    train_size, test_size, gamma, alpha, trial
                )
                
                model_results.append(result)
                task_results.append(result)
                all_results.append(result)
            
            best_model_result = sorted(model_results, key=lambda x: x['f1_score'], reverse=True)[0]
            print(f"\nBest trial for {model_name} on {task}:")
            print(f"Trial: {best_model_result['trial']}, Gamma: {best_model_result['gamma']:.2f}, "
                  f"Alpha: {best_model_result['alpha']:.2f}, F1: {best_model_result['f1_score']:.4f}")
        
        best_task_result = sorted(task_results, key=lambda x: x['f1_score'], reverse=True)[0]
        print(f"\nBEST MODEL FOR TASK '{task}':")
        print(f"Model: {best_task_result['model']}, Trial: {best_task_result['trial']}, "
              f"Gamma: {best_task_result['gamma']:.2f}, Alpha: {best_task_result['alpha']:.2f}, "
              f"F1: {best_task_result['f1_score']:.4f}")
    
    best_overall = sorted(all_results, key=lambda x: x['f1_score'], reverse=True)[0]
    print(f"\n\nOVERALL BEST RESULT ACROSS ALL TASKS:")
    print(f"Task: {best_overall['task']}, Model: {best_overall['model']}, "
          f"Trial: {best_overall['trial']}, Gamma: {best_overall['gamma']:.2f}, "
          f"Alpha: {best_overall['alpha']:.2f}, F1: {best_overall['f1_score']:.4f}")
    
    summary_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="all_tasks_summary",
        reinit=True,
    )
    
    data = []
    for result in sorted(all_results, key=lambda x: (x['task'], x['model'], -x['f1_score'])):
        data.append([
            result['task'],
            result['model'], 
            result['trial'],
            result['gamma'], 
            result['alpha'], 
            result['f1_score'],
            result['precision'],
            result['recall'],
            result['best_epoch']
        ])
    
    comparison_table = wandb.Table(
        columns=["Task", "Model", "Trial", "Gamma", "Alpha", "F1 Score", "Precision", "Recall", "Best Epoch"],
        data=data
    )
    wandb.log({"all_results": comparison_table})
    
    best_per_task_model = []
    for task in tasks:
        task_results = [r for r in all_results if r['task'] == task]
        for model_name in models_to_test:
            model_results = [r for r in task_results if r['model'] == model_name]
            if model_results:
                best_result = sorted(model_results, key=lambda x: x['f1_score'], reverse=True)[0]
                best_per_task_model.append([
                    task,
                    model_name,
                    best_result['trial'],
                    best_result['gamma'],
                    best_result['alpha'],
                    best_result['f1_score']
                ])
    
    best_models_table = wandb.Table(
        columns=["Task", "Model", "Best Trial", "Gamma", "Alpha", "F1 Score"],
        data=best_per_task_model
    )
    wandb.log({"best_models_per_task": best_models_table})
    
    best_f1_data = []
    for task in tasks:
        for model_name in models_to_test:
            model_task_results = [r for r in all_results if r['task'] == task and r['model'] == model_name]
            if model_task_results:
                best_result = sorted(model_task_results, key=lambda x: x['f1_score'], reverse=True)[0]
                best_f1_data.append([f"{task}_{model_name}", best_result['f1_score']])
    
    wandb.log({
        "best_f1_per_task_model": wandb.plot.bar(
            wandb.Table(columns=["Task_Model", "F1 Score"], data=best_f1_data),
            "Task_Model", "F1 Score",
            title="Best F1 Score by Task and Model"
        )
    })
    
    summary_run.finish()


if __name__ == "__main__":
    main()