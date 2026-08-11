"""
exp02: fine-tune MobileNetV3-Small, initialized from exp01's checkpoint (which was
itself initialized from exp00, then adapted to the exp01 racing pool), on a COMBINED
dataset = RSCD train (2582 imgs) + exp01 racing pool (115 imgs, existing split) +
NEW exp02 racing pool v2 (178 imgs, sourced separately, zero overlap with the 49-image
eval set and the exp01 pool verified by sha256 in scripts/_v2_leakage_check.py).

Oversampling strategy (documented reasoning):
  exp01 used a single flat 6x oversample factor on its 115-image racing pool
  (44 DRY / 12 DAMP / 59 WET), because DAMP was so scarce (12 imgs) that a uniform
  factor was the simplest way to avoid drowning gradient signal from the whole pool.

  For exp02 we now have substantially more racing-domain data across BOTH pools
  combined (train-split counts):
    exp01 pool (all used as train, no re-split): DRY=44  DAMP=12  WET=59
    v2 pool (own 85/15 split):                    DRY=99  DAMP=33  WET=19
    combined racing train raw:                     DRY=143 DAMP=45  WET=78  (266 total)

  DAMP is still the scarcest class in absolute terms (45 raw) even though it grew
  ~3.75x versus exp01 (12 raw). To keep pushing the model to learn DAMP specifically
  (exp00/exp01 showed DAMP was consistently the weakest class - lowest recall/F1),
  we use PER-CLASS oversample factors instead of exp01's flat factor:
    DRY:  3x   (143 -> 429)   - diversity is valuable but DRY is already well
                                 represented in RSCD (dominant RSCD class), so a
                                 lighter oversample avoids overwhelming batches
                                 with easy examples
    DAMP: 8x   (45  -> 360)   - highest oversample: scarcest class, historically
                                 weakest metric, most valuable new data
    WET:  3x   (78  -> 234)   - WET is already reasonably well covered by exp01
                                 (recall likely already decent), light oversample
                                 mainly to keep it competitive with DRY/DAMP volume

  This gives combined oversampled racing train ~= 429+360+234 = 1023 images versus
  RSCD's 2582, i.e. racing-domain data is ~28% of the combined train set (compare
  to exp01's racing contribution of 115*6=690 / (2582+690) = ~21%). This is a
  deliberate increase given genuinely more/better data is now available, especially
  for DAMP, while still keeping RSCD as the majority signal to avoid catastrophic
  forgetting of the base domain.

Validation = RSCD val manifest + exp01 racing val (18 imgs, held out) + v2 racing
val (27 imgs, held out, its own stratified split). Model selection uses combined
val macro-F1, matching exp01's protocol.
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
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

ROOT = 'c:/Users/Rivan/Projects/AI_Grand_Prix'
DATA_DIR = f'{ROOT}/data/manifests'
RACING_DIR_V1 = f'{ROOT}/data/racing_train_pool'
RACING_DIR_V2 = f'{ROOT}/data/racing_train_pool_v2'
EXP01_CKPT = f'{ROOT}/experiments/exp01_racing_finetune/checkpoints/best_model.pth'
EXP_DIR = f'{ROOT}/experiments/exp02_racing_v2'
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
    """Per-item repeat: pass a list of (item, repeat_count) OR a flat items list with
    a uniform repeat. Here we build the repeated list explicitly per class outside."""
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

# ---- load racing pools ----
with open(os.path.join(RACING_DIR_V1, 'racing_train_split.json'), encoding='utf-8') as f:
    racing_v1_train_raw = json.load(f)
with open(os.path.join(RACING_DIR_V1, 'racing_val_split.json'), encoding='utf-8') as f:
    racing_v1_val_raw = json.load(f)
with open(os.path.join(RACING_DIR_V2, 'racing_train_split.json'), encoding='utf-8') as f:
    racing_v2_train_raw = json.load(f)
with open(os.path.join(RACING_DIR_V2, 'racing_val_split.json'), encoding='utf-8') as f:
    racing_v2_val_raw = json.load(f)

combined_racing_train_raw = racing_v1_train_raw + racing_v2_train_raw
combined_racing_val_raw = racing_v1_val_raw + racing_v2_val_raw

from collections import Counter
raw_counts = Counter(r['label'] for r in combined_racing_train_raw)
print("combined racing train (raw, pre-oversample) class counts:", dict(raw_counts))

OVERSAMPLE_FACTORS = {'DRY': 3, 'DAMP': 8, 'WET': 3}
print("per-class oversample factors:", OVERSAMPLE_FACTORS)

racing_train_oversampled = per_class_oversample(combined_racing_train_raw, OVERSAMPLE_FACTORS)
oversampled_counts = Counter(r['label'] for r in racing_train_oversampled)
print("combined racing train (oversampled) class counts:", dict(oversampled_counts))

# ---- datasets ----
rscd_train = RSCDDataset(os.path.join(DATA_DIR, 'split_manifest_train.csv'), train_tf)
rscd_val = RSCDDataset(os.path.join(DATA_DIR, 'split_manifest_val.csv'), eval_tf)

racing_train = RacingDataset(racing_train_oversampled, train_tf)
racing_val = RacingDataset(combined_racing_val_raw, eval_tf)

train_ds = ConcatDataset([rscd_train, racing_train])
val_ds = ConcatDataset([rscd_val, racing_val])

print(f"RSCD train={len(rscd_train)} val={len(rscd_val)}")
print(f"racing train(raw)={len(combined_racing_train_raw)} train(oversampled)={len(racing_train)} val={len(racing_val)}")
print(f"COMBINED train={len(train_ds)} val={len(val_ds)}")

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# class weights computed on the combined train set (post-oversampling) actual composition
rscd_train_df = rscd_train.df
rscd_labels = [CLASS_TO_IDX[l] for l in rscd_train_df['label']]
racing_labels = [CLASS_TO_IDX[r['label']] for r in racing_train_oversampled]
all_train_labels = rscd_labels + racing_labels
counts = [all_train_labels.count(i) for i in range(3)]
print("combined train class counts:", dict(zip(CLASSES, counts)))
weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)
weights = weights / weights.sum() * 3
weights = weights.to(device)
print("class weights:", weights.cpu().tolist())

# ---- model: init from exp01 checkpoint (NOT exp00, NOT ImageNet) ----
model = mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 3)
state = torch.load(EXP01_CKPT, map_location='cpu')
model.load_state_dict(state)
model = model.to(device)
print("initialized from exp01 checkpoint:", EXP01_CKPT)

# same fine-tune scope as exp00/exp01: classifier head + last 3 feature blocks unfrozen
for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-3:].parameters():
    param.requires_grad = True

# keep exp01's LR/schedule philosophy: same lr/epochs/patience unless there's a
# reason to change. No strong reason here, so keep them identical for comparability.
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
    'experiment': 'exp02_racing_v2',
    'init_checkpoint': EXP01_CKPT,
    'rscd_train_n': len(rscd_train), 'rscd_val_n': len(rscd_val),
    'racing_v1_pool_dir': RACING_DIR_V1, 'racing_v1_train_n_raw': len(racing_v1_train_raw), 'racing_v1_val_n': len(racing_v1_val_raw),
    'racing_v2_pool_dir': RACING_DIR_V2, 'racing_v2_train_n_raw': len(racing_v2_train_raw), 'racing_v2_val_n': len(racing_v2_val_raw),
    'combined_racing_train_n_raw': len(combined_racing_train_raw),
    'combined_racing_train_raw_class_counts': dict(raw_counts),
    'oversample_factors_per_class': OVERSAMPLE_FACTORS,
    'combined_racing_train_n_oversampled': len(racing_train_oversampled),
    'combined_racing_train_oversampled_class_counts': dict(oversampled_counts),
    'combined_racing_val_n': len(combined_racing_val_raw),
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
