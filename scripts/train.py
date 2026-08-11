"""
Step 5: Fine-tune MobileNetV3-Small on DRY/DAMP/WET, 3-class classification.
"""
import os, time, json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report

DATA_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data'
MODELS_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/models'
os.makedirs(MODELS_DIR, exist_ok=True)

CLASSES = ['DRY', 'DAMP', 'WET']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
if device.type == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))

IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

train_ds = RSCDDataset(os.path.join(DATA_DIR, 'manifest_train.csv'), train_tf)
val_ds = RSCDDataset(os.path.join(DATA_DIR, 'manifest_val.csv'), eval_tf)
test_ds = RSCDDataset(os.path.join(DATA_DIR, 'manifest_test.csv'), eval_tf)

print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

BATCH_SIZE = 32
NUM_WORKERS = 4
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# class weights for imbalance
train_labels = [CLASS_TO_IDX[l] for l in train_ds.df['label']]
counts = [train_labels.count(i) for i in range(3)]
print("train class counts:", dict(zip(CLASSES, counts)))
weights = torch.tensor([1.0 / c if c > 0 else 0.0 for c in counts], dtype=torch.float32).to(device)

model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 3)
model = model.to(device)

# Fine-tune classifier + last few feature blocks
for param in model.features.parameters():
    param.requires_grad = False
# unfreeze last 3 feature blocks for a bit more capacity
for param in model.features[-3:].parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
criterion = nn.CrossEntropyLoss(weight=weights)

EPOCHS = 15
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
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    scheduler.step()
    train_loss = running_loss / len(train_ds)

    # validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    elapsed = time.time() - t0
    print(f"epoch {epoch+1}/{EPOCHS} train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f} time={elapsed:.1f}s")
    history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_acc': val_acc, 'val_macro_f1': val_f1, 'time_s': elapsed})

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"early stopping at epoch {epoch+1}")
            break

# load best model
model.load_state_dict(best_state)
model.eval()

torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'trackpulse_classifier.pt'))
print("saved checkpoint to", os.path.join(MODELS_DIR, 'trackpulse_classifier.pt'))

# ---- final test evaluation ----
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

test_acc = accuracy_score(all_labels, all_preds)
test_macro_f1 = f1_score(all_labels, all_preds, average='macro')
prec, rec, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=[0,1,2])
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)

print("\n=== TEST SET RESULTS ===")
print("accuracy:", test_acc)
print("macro F1:", test_macro_f1)
print(report)
print("confusion matrix (rows=true, cols=pred), order", CLASSES)
print(cm)

results = {
    'test_accuracy': float(test_acc),
    'test_macro_f1': float(test_macro_f1),
    'per_class': {
        CLASSES[i]: {'precision': float(prec[i]), 'recall': float(rec[i]), 'f1': float(f1[i]), 'support': int(support[i])}
        for i in range(3)
    },
    'confusion_matrix': cm.tolist(),
    'classes_order': CLASSES,
    'history': history,
    'best_val_macro_f1': float(best_val_f1),
    'device': str(device),
}
with open(os.path.join(DATA_DIR, 'train_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nsaved train_results.json")
