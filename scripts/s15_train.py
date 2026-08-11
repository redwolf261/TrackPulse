"""
Sections 15-18: train MobileNetV3-Small on RSCD 3-class (DRY/DAMP/WET), GPU + AMP,
early stopping on val macro-F1, then evaluate on held-out test manifest.
Augmentation applied only via train-time transform in DataLoader (never baked into
files, never applied before split) - satisfies Section 17.
"""
import os, time, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATA_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data/manifests'
EXP_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp00_rscd_baseline'
CKPT_DIR = os.path.join(EXP_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

CLASSES = ['DRY', 'DAMP', 'WET']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
if device.type == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])
eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

class RSCDDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        img = self.transform(img)
        label = CLASS_TO_IDX[row['label']]
        return img, label

train_ds = RSCDDataset(os.path.join(DATA_DIR, 'split_manifest_train.csv'), train_tf)
val_ds = RSCDDataset(os.path.join(DATA_DIR, 'split_manifest_val.csv'), eval_tf)
test_ds = RSCDDataset(os.path.join(DATA_DIR, 'split_manifest_test.csv'), eval_tf)

print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

train_labels = [CLASS_TO_IDX[l] for l in train_ds.df['label']]
counts = [train_labels.count(i) for i in range(3)]
print("train class counts:", dict(zip(CLASSES, counts)))
weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)
weights = weights / weights.sum() * 3  # normalize
weights = weights.to(device)
print("class weights:", weights.cpu().tolist())

model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 3)
model = model.to(device)

for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-3:].parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
EPOCHS = 15
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(weight=weights)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

PATIENCE = 4
best_val_f1 = -1
best_state = None
patience_ctr = 0
history = []

for epoch in range(EPOCHS):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
    scheduler.step()
    train_loss = running_loss / len(train_ds)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    elapsed = time.time() - t0
    print(f"epoch {epoch+1}/{EPOCHS} train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f} time={elapsed:.1f}s lr={scheduler.get_last_lr()[0]:.2e}")
    history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_acc': val_acc, 'val_macro_f1': val_f1, 'time_s': elapsed, 'lr': scheduler.get_last_lr()[0]})

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'final_model.pth'))

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, os.path.join(CKPT_DIR, 'best_model.pth'))
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"early stopping at epoch {epoch+1} (best val_macro_f1={best_val_f1:.4f})")
            break

pd.DataFrame(history).to_csv(os.path.join(EXP_DIR, 'training_log.csv'), index=False)
print("saved training_log.csv")

# load BEST checkpoint (by val macro-F1) for final test evaluation - model selection uses val only, never test
model.load_state_dict(best_state)
model.eval()

# also save legacy path for backend compatibility
torch.save(best_state, 'c:/Users/Rivan/Projects/AI_Grand_Prix/models/trackpulse_classifier.pt')
print("also saved to models/trackpulse_classifier.pt (backend-compatible path)")

# ---- final test evaluation (Section 18) - test set touched ONLY here, once ----
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(imgs)
        probs = torch.softmax(out.float(), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.tolist())

test_acc = accuracy_score(all_labels, all_preds)
test_macro_f1 = f1_score(all_labels, all_preds, average='macro')
prec, rec, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=[0,1,2])
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)

print("\n=== TEST SET RESULTS (held out, never used in training/tuning/model-selection) ===")
print("accuracy:", test_acc)
print("macro F1:", test_macro_f1)
print(report)
print("confusion matrix (rows=true, cols=pred), order", CLASSES)
print(cm)
print(f"\nWET recall (safety-critical): {rec[2]:.4f}")
print(f"WATER... (same as WET here) precision: {prec[2]:.4f}")
print(f"DRY precision: {prec[0]:.4f}")

results = {
    'test_accuracy': float(test_acc),
    'test_macro_f1': float(test_macro_f1),
    'per_class': {
        CLASSES[i]: {'precision': float(prec[i]), 'recall': float(rec[i]), 'f1': float(f1[i]), 'support': int(support[i])}
        for i in range(3)
    },
    'confusion_matrix': cm.tolist(),
    'classes_order': CLASSES,
    'wet_recall': float(rec[2]),
    'dry_precision': float(prec[0]),
    'best_val_macro_f1': float(best_val_f1),
    'device': str(device),
    'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
    'seed': SEED,
}
with open(os.path.join(EXP_DIR, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)

# save raw test probs for calibration step
test_manifest = pd.read_csv(os.path.join(DATA_DIR, 'split_manifest_test.csv'))
probs_df = pd.DataFrame(all_probs, columns=[f'prob_{c}' for c in CLASSES])
probs_df['true_label'] = [CLASSES[l] for l in all_labels]
probs_df['pred_label'] = [CLASSES[p] for p in all_preds]
probs_df['filename'] = test_manifest['filename'].values
probs_df.to_csv(os.path.join(EXP_DIR, 'test_predictions_with_probs.csv'), index=False)

print("\nsaved metrics.json and test_predictions_with_probs.csv")
