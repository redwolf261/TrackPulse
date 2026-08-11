"""
exp01: fine-tune MobileNetV3-Small, initialized from exp00's frozen best_model.pth,
on a COMBINED dataset = RSCD train (2582 imgs, existing manifest) + new racing-domain
training pool (115 imgs, sourced separately from the 49-image eval set, zero overlap
verified by filename + sha256 hash).

Racing images are oversampled (repeated) in the combined train set so they are not
drowned out by the 2582 RSCD images - otherwise gradient signal from the domain-gap-closing
data would be ~4% of batches.

Validation = RSCD val manifest (group-aware, from exp00) + held-back racing val images
(18 images never used in training). Model selection (early stopping / best checkpoint)
uses combined val macro-F1.
"""
import os, time, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small
from PIL import Image
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

ROOT = 'c:/Users/Rivan/Projects/AI_Grand_Prix'
DATA_DIR = f'{ROOT}/data/manifests'
RACING_DIR = f'{ROOT}/data/racing_train_pool'
EXP00_CKPT = f'{ROOT}/experiments/exp00_rscd_baseline/checkpoints/best_model.pth'
EXP_DIR = f'{ROOT}/experiments/exp01_racing_finetune'
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

class RacingDataset(Dataset):
    def __init__(self, json_path, transform, repeat=1):
        with open(json_path, encoding='utf-8') as f:
            items = json.load(f)
        self.items = items * repeat
        self.transform = transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        row = self.items[idx]
        path = os.path.join(ROOT, row['filepath']) if not os.path.isabs(row['filepath']) else row['filepath']
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = CLASS_TO_IDX[row['label']]
        return img, label

# ---- datasets ----
rscd_train = RSCDDataset(os.path.join(DATA_DIR, 'split_manifest_train.csv'), train_tf)
rscd_val = RSCDDataset(os.path.join(DATA_DIR, 'split_manifest_val.csv'), eval_tf)

RACING_REPEAT = 6  # oversample racing pool so it contributes a meaningful fraction of gradient signal
racing_train = RacingDataset(os.path.join(RACING_DIR, 'racing_train_split.json'), train_tf, repeat=RACING_REPEAT)
racing_val = RacingDataset(os.path.join(RACING_DIR, 'racing_val_split.json'), eval_tf, repeat=1)

train_ds = ConcatDataset([rscd_train, racing_train])
val_ds = ConcatDataset([rscd_val, racing_val])

print(f"RSCD train={len(rscd_train)} val={len(rscd_val)}")
print(f"racing train(raw)={len(racing_train)//RACING_REPEAT} train(oversampled x{RACING_REPEAT})={len(racing_train)} val={len(racing_val)}")
print(f"COMBINED train={len(train_ds)} val={len(val_ds)}")

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# class weights computed on the combined train set (post-oversampling) actual composition
rscd_train_df = rscd_train.df
rscd_labels = [CLASS_TO_IDX[l] for l in rscd_train_df['label']]
with open(os.path.join(RACING_DIR, 'racing_train_split.json'), encoding='utf-8') as f:
    racing_train_raw = json.load(f)
racing_labels = [CLASS_TO_IDX[r['label']] for r in racing_train_raw] * RACING_REPEAT
all_train_labels = rscd_labels + racing_labels
counts = [all_train_labels.count(i) for i in range(3)]
print("combined train class counts:", dict(zip(CLASSES, counts)))
weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)
weights = weights / weights.sum() * 3
weights = weights.to(device)
print("class weights:", weights.cpu().tolist())

# ---- model: init from exp00 frozen checkpoint (NOT ImageNet scratch) ----
model = mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 3)
state = torch.load(EXP00_CKPT, map_location='cpu')
model.load_state_dict(state)
model = model.to(device)
print("initialized from exp00 checkpoint:", EXP00_CKPT)

# same fine-tune scope as exp00: classifier head + last 3 feature blocks unfrozen
for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-3:].parameters():
    param.requires_grad = True

# lower LR than exp00's from-ImageNet run, since we're adapting an already-converged
# checkpoint and want to avoid catastrophic forgetting of the RSCD source domain
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
EPOCHS = 12
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
print(f"\nBEST val_macro_f1={best_val_f1:.4f}, checkpoint saved to {CKPT_DIR}/best_model.pth")

summary = {
    'experiment': 'exp01_racing_finetune',
    'init_checkpoint': EXP00_CKPT,
    'rscd_train_n': len(rscd_train), 'rscd_val_n': len(rscd_val),
    'racing_train_n_raw': len(racing_train_raw), 'racing_train_oversample_factor': RACING_REPEAT,
    'racing_val_n': len(racing_train_raw) and len(json.load(open(os.path.join(RACING_DIR,'racing_val_split.json'),encoding='utf-8'))),
    'combined_train_n': len(train_ds), 'combined_val_n': len(val_ds),
    'combined_train_class_counts': dict(zip(CLASSES, counts)),
    'lr': 3e-4, 'epochs_max': EPOCHS, 'epochs_run': len(history),
    'best_val_macro_f1': float(best_val_f1),
    'device': str(device), 'gpu_name': torch.cuda.get_device_name(0) if device.type=='cuda' else None,
    'seed': SEED,
}
with open(os.path.join(EXP_DIR, 'training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("saved training_summary.json")
