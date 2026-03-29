import re
import numpy as np
from tqdm import tqdm


from sklearn.model_selection import (
    train_test_split)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score)

from transformers import CamembertTokenizer, CamembertForSequenceClassification

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

FILE_PRESIDENTS = "/tempory/presidents.utf8"

MAX_LENGTH = 110 # TODO: tweak it

TRAIN_MODE = "last_n"  # "mlp_only" | "last_n" | "full" | "llrd"
N_UNFREEZE = 2           # used when TRAIN_MODE == "last_n"
LLRD_DECAY = 0.9         # used when TRAIN_MODE == "llrd"

USE_WEIGHTS = True
LOAD_CHECKPOINT = False
SAVE_CHECKPOINT = True
FNAME = f"/tempory/pres_results/checkpoint-presidents-{TRAIN_MODE + ('_balanced_' if USE_WEIGHTS else '_unbalanced_')}camembert-"
MAX_EPOCHS = 100

print(f"Train mode: {TRAIN_MODE} | Balanced: {'yes' if USE_WEIGHTS else 'no'} | N unfreeze = {N_UNFREEZE} | llrd decay = {LLRD_DECAY}")

def configure_finetuning(model, mode, n_unfreeze=2, llrd_decay=0.9, base_lr=2e-5):
    """
    Returns an optimizer configured according to the fine-tuning strategy.

    Modes:
      mlp_only  — freeze the entire encoder, train only the classification head
      last_n    — freeze all encoder layers except the last n transformer blocks
      full      — unfreeze the entire model
      llrd      — unfreeze all, but apply layer-wise LR decay toward the input
    """

    # Start from fully frozen
    for param in model.parameters():
        param.requires_grad = False

    if mode == "mlp_only":
        for param in model.classifier.parameters():
            param.requires_grad = True
        return AdamW(model.classifier.parameters(), lr=base_lr)

    elif mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return AdamW(model.parameters(), lr=base_lr)

    elif mode == "last_n":
        # Always train the head
        for param in model.classifier.parameters():
            param.requires_grad = True

        # CamemBERT's transformer blocks live at model.roberta.encoder.layer
        layers = model.roberta.encoder.layer  # ModuleList of 12 blocks
        for layer in layers[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

        return AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)

    elif mode == "llrd":
        # Unfreeze everything — LLRD controls the update magnitude per layer,
        # not which layers are frozen
        for param in model.parameters():
            param.requires_grad = True

        layers = model.roberta.encoder.layer  # 12 blocks, index 0 = closest to input
        n_layers = len(layers)

        # Build parameter groups from top (head) to bottom (embedding),
        # each multiplied by llrd_decay relative to the one above it
        param_groups = []

        # Head gets full base_lr
        param_groups.append({"params": model.classifier.parameters(), "lr": base_lr})

        # Transformer layers: layer[-1] gets base_lr * decay^1, layer[-2] * decay^2, ...
        for depth, layer in enumerate(reversed(layers)):
            lr = base_lr * (llrd_decay ** (depth + 1))
            param_groups.append({"params": layer.parameters(), "lr": lr})

        # Embeddings get the smallest lr (decay^(n_layers+1))
        emb_lr = base_lr * (llrd_decay ** (n_layers + 1))
        param_groups.append({"params": model.roberta.embeddings.parameters(), "lr": emb_lr})

        return AdamW(param_groups)

    else:
        raise ValueError(f"Unknown TRAIN_MODE: '{mode}'. Choose from: mlp_only, last_n, full, llrd")




def load_presidents(file=FILE_PRESIDENTS) -> tuple[np.ndarray, np.ndarray]:
    """
    0 for Chirac
    1 for Mitterrand
    """
    texts = []
    labels = []
    with open(file) as f:
        for line in f.readlines():
            speaker, sentence = re.match(r"<\d+:\d+:(.)>\s*(.*)\n", line).groups()
            if speaker == "C":
                speaker = 0
            elif speaker == "M":
                speaker = 1
            else:
                # Something went wrong
                raise ValueError
            texts.append(sentence)
            labels.append(speaker)
    return np.array(texts), np.array(labels)



X_train_full, y_train_full = load_presidents()


# Keep track of indices to restore the time order
indices = np.arange(len(X_train_full))

# Step 1: randomly select which indices go to train/val
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    # random_state=1,
    shuffle=False,
    # stratify=y_train_full
)
print("No shuffling for train test split")

train_idx_sorted = np.sort(train_idx)
val_idx_sorted = np.sort(val_idx)

X_train = X_train_full[train_idx_sorted]
y_train = y_train_full[train_idx_sorted]
X_val = X_train_full[val_idx_sorted]
y_val = y_train_full[val_idx_sorted]

num_pos = np.sum(y_train_full == 1)
num_neg = np.sum(y_train_full == 0)
num_min = np.minimum(num_neg, num_pos)
total = num_pos + num_neg


# UNDERSAMPLE = False
# if UNDERSAMPLE:
#     num_pos = np.sum(y_train == 1)
#     num_neg = np.sum(y_train== 0)
#     num_min = np.minimum(num_neg, num_pos)
#     idx_neg_under_sampled = np.random.choice(np.where(y_train==0)[0], num_min, replace=False)
#     idx_pos_under_sampled = np.random.choice(np.where(y_train==1)[0], num_min, replace=False)# np.where(y_train==1)[0]
#     idx_under_sampled = np.sort(np.concat([idx_neg_under_sampled, idx_pos_under_sampled]))
#     X_train = X_train[idx_under_sampled]
#     y_train = y_train[idx_under_sampled]

y_train = torch.from_numpy(y_train).long()
y_val = torch.from_numpy(y_val).long()

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

class PresidentsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove the batch dimension added by return_tensors="pt"
        return {key: val.squeeze(0) for key, val in encoding.items()}, self.labels[idx]

train_dataset = PresidentsDataset(X_train, y_train, tokenizer, MAX_LENGTH)
val_dataset = PresidentsDataset(X_val, y_val, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4,pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4,pin_memory=torch.cuda.is_available())


model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)

model.to(device)

# weights = torch.tensor([total/num_neg, total/num_pos]).float().to(device)

weights = torch.tensor([np.sqrt(total/num_neg), np.sqrt(total/num_pos)]).float().to(device)

# weight_pos = num_neg / num_pos 
# weights = torch.tensor([1.0, weight_pos]).float().to(device)

        
loss_fn = nn.CrossEntropyLoss(weight=weights if USE_WEIGHTS else None
                              )

optimizer = configure_finetuning(model, TRAIN_MODE, N_UNFREEZE, LLRD_DECAY, base_lr=2e-5)

if LOAD_CHECKPOINT:
    checkpoint = torch.load(FNAME+"_last.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0

for epoch in range(start_epoch,MAX_EPOCHS):
    model.train()
    train_pred = []
    train_true = []
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch + 1}"):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        with torch.no_grad():
            train_pred.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            train_true.extend(labels.cpu().numpy())
        
    model.eval()
    val_scores = [] 
    val_pred = []
    val_true = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluation epoch {epoch + 1}"):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1] 
            val_scores.extend(probs.cpu().numpy())
            val_pred.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    TP = sum((t == 1 and p == 1) for t, p in zip(val_true, val_pred))
    TN = sum((t == 0 and p == 0) for t, p in zip(val_true, val_pred))
    FP = sum((t == 0 and p == 1) for t, p in zip(val_true, val_pred))
    FN = sum((t == 1 and p == 0) for t, p in zip(val_true, val_pred))
    f1 = f1_score(val_true, val_pred)
    ap = average_precision_score(val_true, val_scores)
    roc_auc = roc_auc_score(val_true, val_scores)
    
    print(
    f"[Epoch {epoch+1:03d}] "
    f"Loss: {train_loss/len(train_loader):.4f} | "
    f"F1: {f1:.4f} | AP: {ap:.4f} | AUC: {roc_auc:.4f} || "
    f"TP: {TP:5d} FP: {FP:5d} | TN: {TN:5d} FN: {FN:5d}")
    
    if SAVE_CHECKPOINT:
        checkpoint = {
        'epoch': epoch,
        'roc_auc': roc_auc,
        'ap': ap,
        'f1': f1,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, FNAME+f"last.pth")
        torch.save(checkpoint, FNAME+f"{epoch}.pth")