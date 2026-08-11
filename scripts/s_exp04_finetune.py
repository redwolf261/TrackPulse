"""
exp04: fix the DAMP/DRY trade-off from exp03 by trying gentler training
strategies, on the EXACT same training set exp03 used (same holdout, same
pools). Initialized from exp02's checkpoint each time (same as exp03).

Usage:
    python s_exp04_finetune.py optionA   # lighter DAMP oversample (6x)
    python s_exp04_finetune.py optionB   # class-weighted loss, natural (non-oversampled) data
    python s_exp04_finetune.py optionC   # hybrid: mild oversample (3x) + mild class weighting

Same fine-tune scope / LR / schedule philosophy as exp01/exp02/exp03.
"""
import os, sys, time, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

OPTION = sys.argv[1] if len(sys.argv) > 1 else 'optionA'
assert OPTION in ('optionA', 'optionB', 'optionC'), f"unknown option {OPTION}"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

ROOT = 'c:/Users/Rivan/Projects/AI_Grand_Prix'
DATA_DIR = f'{ROOT}/data/manifests'
RACING_DIR_V1 = f'{ROOT}/data/racing_train_pool'
RACING_DIR_V2 = f'{ROOT}/data/racing_train_pool_v2'
EXP02_CKPT = f'{ROOT}/experiments/exp02_racing_v2/checkpoints/best_model.pth'
EXP_DIR = f'{ROOT}/experiments/exp04_damp_rebalance'
CKPT_DIR = os.path.join(EXP_DIR, f'checkpoints_{OPTION}')
os.makedirs(CKPT_DIR, exist_ok=True)

HOLDOUT_LIST = f'{ROOT}/data/racing_spotcheck_v2/damp_holdout_source_filenames.json'
with open(HOLDOUT_LIST, encoding='utf-8') as f:
    HOLDOUT_FILENAMES = set(json.load(f))
print(f"[{OPTION}] excluding {len(HOLDOUT_FILENAMES)} held-out DAMP filenames from training/val pools")

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
    def __init__(self, items, transform):
        self.items = items
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

def per_class_oversample(items, factors):
    out = []
    for it in items:
        f = factors[it['label']]
        out.extend([it] * f)
    return out

def exclude_holdout(items):
    return [it for it in items if it['filename'] not in HOLDOUT_FILENAMES]

# ---- load racing pools, excluding held-out DAMP filenames (identical to exp03) ----
with open(os.path.join(RACING_DIR_V1, 'racing_train_split.json'), encoding='utf-8') as f:
    racing_v1_train_raw = exclude_holdout(json.load(f))
with open(os.path.join(RACING_DIR_V1, 'racing_val_split.json'), encoding='utf-8') as f:
    racing_v1_val_raw = exclude_holdout(json.load(f))
with open(os.path.join(RACING_DIR_V2, 'racing_train_split.json'), encoding='utf-8') as f:
    racing_v2_train_raw = exclude_holdout(json.load(f))
with open(os.path.join(RACING_DIR_V2, 'racing_val_split.json'), encoding='utf-8') as f:
    racing_v2_val_raw = exclude_holdout(json.load(f))

combined_racing_train_raw = racing_v1_train_raw + racing_v2_train_raw
combined_racing_val_raw = racing_v1_val_raw + racing_v2_val_raw

raw_counts = Counter(r['label'] for r in combined_racing_train_raw)
print("combined racing train (raw, post-holdout, pre-oversample) class counts:", dict(raw_counts))

# ---- strategy-specific oversample factors / class weighting ----
if OPTION == 'optionA':
    # Lighter DAMP oversample factor (6x instead of exp03's 10x). DRY/WET kept at exp02/exp03's 3x.
    OVERSAMPLE_FACTORS = {'DRY': 3, 'DAMP': 6, 'WET': 3}
    USE_EXTRA_CLASS_WEIGHT = False
elif OPTION == 'optionB':
    # No oversampling at all (natural/raw data); rely purely on inverse-frequency class weights in CE loss.
    OVERSAMPLE_FACTORS = {'DRY': 1, 'DAMP': 1, 'WET': 1}
    USE_EXTRA_CLASS_WEIGHT = True
else:  # optionC: hybrid - mild oversample + mild extra class weighting
    OVERSAMPLE_FACTORS = {'DRY': 1, 'DAMP': 3, 'WET': 1}
    USE_EXTRA_CLASS_WEIGHT = True

print(f"[{OPTION}] per-class oversample factors:", OVERSAMPLE_FACTORS, "| extra inverse-freq class weight:", USE_EXTRA_CLASS_WEIGHT)

racing_train_oversampled = per_class_oversample(combined_racing_train_raw, OVERSAMPLE_FACTORS)
oversampled_counts = Counter(r['label'] for r in racing_train_oversampled)
print("combined racing train (post-strategy) class counts:", dict(oversampled_counts))

# ---- datasets ----
rscd_train = RSCDDataset(os.path.join(DATA_DIR, 'split_manifest_train.csv'), train_tf)
rscd_val = RSCDDataset(os.path.join(DATA_DIR, 'split_manifest_val.csv'), eval_tf)

racing_train = RacingDataset(racing_train_oversampled, train_tf)
racing_val = RacingDataset(combined_racing_val_raw, eval_tf)

train_ds = ConcatDataset([rscd_train, racing_train])
val_ds = ConcatDataset([rscd_val, racing_val])

print(f"RSCD train={len(rscd_train)} val={len(rscd_val)}")
print(f"racing train(raw,post-holdout)={len(combined_racing_train_raw)} train(post-strategy)={len(racing_train)} val={len(racing_val)}")
print(f"COMBINED train={len(train_ds)} val={len(val_ds)}")

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

rscd_train_df = rscd_train.df
rscd_labels = [CLASS_TO_IDX[l] for l in rscd_train_df['label']]
racing_labels = [CLASS_TO_IDX[r['label']] for r in racing_train_oversampled]
all_train_labels = rscd_labels + racing_labels
counts = [all_train_labels.count(i) for i in range(3)]
print("combined train class counts:", dict(zip(CLASSES, counts)))

# base inverse-frequency weights (same formula as exp02/exp03 used, applied to *actual* loader composition)
weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)
weights = weights / weights.sum() * 3

if USE_EXTRA_CLASS_WEIGHT:
    # For optionB/C: since we removed/reduced oversampling, the natural inverse-frequency
    # weights already do the heavy lifting of correcting for DAMP scarcity relative to DRY/WET
    # and to RSCD's much larger volume. No further multiplier needed beyond the standard
    # 1/count formula - this IS "class-weighted loss instead of oversampling".
    pass

weights = weights.to(device)
print("class weights (used in CrossEntropyLoss):", weights.cpu().tolist())

# ---- model: init from exp02 checkpoint (same fine-tune scope as exp03: last 3 feature blocks unfrozen) ----
model = mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 3)
state = torch.load(EXP02_CKPT, map_location='cpu')
model.load_state_dict(state)
model = model.to(device)
print("initialized from exp02 checkpoint:", EXP02_CKPT)

for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-3:].parameters():
    param.requires_grad = True

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
    print(f"[{OPTION}] epoch {epoch+1}/{EPOCHS} train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f} time={elapsed:.1f}s lr={scheduler.get_last_lr()[0]:.2e}")
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
            print(f"[{OPTION}] early stopping at epoch {epoch+1} (best val_macro_f1={best_val_f1:.4f})")
            break

pd.DataFrame(history).to_csv(os.path.join(EXP_DIR, f'training_log_{OPTION}.csv'), index=False)
print(f"saved training_log_{OPTION}.csv")
print(f"\n[{OPTION}] BEST val_macro_f1={best_val_f1:.4f}, checkpoint saved to {CKPT_DIR}/best_model.pth")

summary = {
    'experiment': f'exp04_damp_rebalance_{OPTION}',
    'strategy': OPTION,
    'init_checkpoint': EXP02_CKPT,
    'rscd_train_n': len(rscd_train), 'rscd_val_n': len(rscd_val),
    'racing_v1_pool_dir': RACING_DIR_V1, 'racing_v1_train_n_raw': len(racing_v1_train_raw), 'racing_v1_val_n': len(racing_v1_val_raw),
    'racing_v2_pool_dir': RACING_DIR_V2, 'racing_v2_train_n_raw': len(racing_v2_train_raw), 'racing_v2_val_n': len(racing_v2_val_raw),
    'n_damp_held_out': len(HOLDOUT_FILENAMES),
    'combined_racing_train_n_raw': len(combined_racing_train_raw),
    'combined_racing_train_raw_class_counts': dict(raw_counts),
    'oversample_factors_per_class': OVERSAMPLE_FACTORS,
    'use_extra_inverse_freq_class_weight': USE_EXTRA_CLASS_WEIGHT,
    'combined_racing_train_n_poststrategy': len(racing_train_oversampled),
    'combined_racing_train_poststrategy_class_counts': dict(oversampled_counts),
    'combined_racing_val_n': len(combined_racing_val_raw),
    'combined_train_n': len(train_ds), 'combined_val_n': len(val_ds),
    'combined_train_class_counts': dict(zip(CLASSES, counts)),
    'class_weights_in_loss': weights.cpu().tolist(),
    'lr': 3e-4, 'epochs_max': EPOCHS, 'epochs_run': len(history),
    'best_val_macro_f1': float(best_val_f1),
    'device': str(device), 'gpu_name': torch.cuda.get_device_name(0) if device.type=='cuda' else None,
    'seed': SEED,
}
with open(os.path.join(EXP_DIR, f'training_summary_{OPTION}.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"saved training_summary_{OPTION}.json")
