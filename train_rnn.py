"""
train_rnn.py — Standalone training script for Assignment 4
RNN-based handwritten digit classifier on MNIST.

Mirrors train_cnn.py from Assignment 3, with the CNN replaced by an LSTM-based RNN.
The key idea: treat each MNIST image as a sequence of 28 rows (timesteps),
each row being a 28-pixel feature vector. The LSTM processes the sequence
row-by-row and classifies the digit from the final hidden state.

Run:
    python3 train_rnn.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import numpy as np
import seaborn as sns
import time
import os

# ── reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── device ───────────────────────────────────────────────────────────────────
# NOTE: PyTorch LSTM on MPS has been unreliable for some versions.
# We fall back to CPU if MPS is available but LSTM is being used — safe fallback.
if torch.backends.mps.is_available():
    # MPS available — use it; if LSTM errors arise, swap to cpu
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# ── hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 64       # same as A2 & A3 — stable gradient estimates
EPOCHS       = 10       # same as A2 & A3 — fair comparison
LR           = 0.001    # Adam default — consistent with previous assignments
INPUT_SIZE   = 28       # each row of the image = 28 pixels  (sequence feature dim)
HIDDEN_SIZE  = 128      # LSTM hidden state dimension
NUM_LAYERS   = 2        # stacked LSTM layers for more expressive representations
NUM_CLASSES  = 10       # digits 0–9
SEQUENCE_LEN = 28       # 28 rows per image = 28 timesteps

os.makedirs("outputs", exist_ok=True)

# ── data loading ──────────────────────────────────────────────────────────────
# Same transform as A2 & A3:  ToTensor gives [1, 28, 28]; Normalize standardises
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTraining samples : {len(train_dataset):,}")
print(f"Test samples     : {len(test_dataset):,}")
print(f"Batches/epoch    : {len(train_loader)}")

# ── model definition ──────────────────────────────────────────────────────────
class DigitRNN(nn.Module):
    """
    LSTM-based RNN for MNIST digit classification.

    Architecture:
        Each 28×28 image is fed as a sequence of 28 rows (timesteps).
        Each timestep carries 28 features (one pixel row).
        The LSTM processes this sequence; only the final hidden state
        is passed to the fully-connected classifier head.

    Input  : [batch, 1, 28, 28]
    Reshape: [batch, 28, 28]     — 28 timesteps × 28 features
    LSTM   : [batch, 28, 128]    — hidden state at each timestep
    Slice  : [batch, 128]        — take the last timestep only
    FC     : [batch, 10]         — logits for 10 digit classes

    Parameters:
        input_size  = 28   (one row of the image)
        hidden_size = 128  (LSTM memory dimension)
        num_layers  = 2    (stacked / deep LSTM)
        num_classes = 10
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
        super(DigitRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── LSTM core ────────────────────────────────────────────────────────
        # batch_first=True → input shape is (batch, seq_len, input_size)
        # dropout between stacked layers acts as regularisation (only if num_layers > 1)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2    # inter-layer dropout for depth > 1
        )

        # ── classifier head ──────────────────────────────────────────────────
        # Similar to A3's FC head; maps final hidden state → class logits
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),             # output-level dropout — same role as A3
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = x.squeeze(1)               # [batch, 28, 28]  — remove channel dim
        # x is now: [batch, seq_len=28, input_size=28]

        # LSTM forward — h0 and c0 default to zeros (stateless per batch)
        out, _ = self.lstm(x)          # out: [batch, 28, hidden_size]

        # Take only the last timestep's output — it has seen the full sequence
        out = out[:, -1, :]            # [batch, hidden_size]

        # Classify
        out = self.classifier(out)     # [batch, num_classes]
        return out


model = DigitRNN().to(device)
print(f"\nModel architecture:\n{model}\n")

total_params    = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ── loss & optimiser ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()                          # same as A2 & A3
optimizer = optim.Adam(model.parameters(), lr=LR)          # same as A2 & A3

# ── helper: evaluate on a DataLoader ─────────────────────────────────────────
def evaluate(loader):
    """Return (avg_loss, accuracy%) for the given dataloader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs      = model(imgs)
            loss         = criterion(outputs, labels)
            total_loss  += loss.item() * imgs.size(0)
            preds        = outputs.argmax(dim=1)
            correct     += (preds == labels).sum().item()
            total       += imgs.size(0)
    return total_loss / total, correct / total * 100

# ── training loop ─────────────────────────────────────────────────────────────
train_losses, train_accs = [], []
test_losses,  test_accs  = [], []

print("\n" + "=" * 65)
print("Training")
print("=" * 65)

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)              # forward through LSTM → FC
        loss    = criterion(outputs, labels)
        loss.backward()                    # BPTT through all 28 timesteps
        # gradient clipping: prevents exploding gradients (key for RNNs)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += imgs.size(0)

    tr_loss = running_loss / total
    tr_acc  = correct / total * 100
    te_loss, te_acc = evaluate(test_loader)

    train_losses.append(tr_loss)
    train_accs.append(tr_acc)
    test_losses.append(te_loss)
    test_accs.append(te_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS}  |  "
          f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%  |  "
          f"Test Loss: {te_loss:.4f}  Test Acc: {te_acc:.2f}%")

elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed:.1f}s")
print(f"Best test accuracy: {max(test_accs):.2f}%")

# ── save model ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "outputs/rnn_digit_model.pth")
print("Model saved to outputs/rnn_digit_model.pth")

# ── plot 1: training curves ───────────────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("RNN (LSTM) Training Curves — MNIST", fontsize=14, fontweight="bold")

axes[0].plot(epochs_range, train_losses, "b-o", label="Train Loss", linewidth=2, markersize=5)
axes[0].plot(epochs_range, test_losses,  "r-o", label="Test Loss",  linewidth=2, markersize=5)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Loss per Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, train_accs, "b-o", label="Train Acc", linewidth=2, markersize=5)
axes[1].plot(epochs_range, test_accs,  "r-o", label="Test Acc",  linewidth=2, markersize=5)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Accuracy per Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([90, 100])

plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/training_curves.png")

# ── plot 2: confusion matrix ──────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs    = imgs.to(device)
        outputs = model(imgs)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

num_classes = 10
cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(all_labels, all_preds):
    cm[t][p] += 1

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10), ax=ax)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label",      fontsize=12)
ax.set_title("Confusion Matrix — RNN (LSTM) on MNIST Test Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/confusion_matrix.png")

# ── plot 3: sample predictions ────────────────────────────────────────────────
test_dataset_raw = datasets.MNIST(root="./data", train=False, download=False,
                                   transform=transform)
indices = np.random.choice(len(test_dataset_raw), 25, replace=False)

fig, axes_grid = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle("Sample Predictions — RNN (LSTM) on MNIST", fontsize=14, fontweight="bold")

model.eval()
with torch.no_grad():
    for i, idx in enumerate(indices):
        img, true_label = test_dataset_raw[idx]
        img_tensor = img.unsqueeze(0).to(device)
        output     = model(img_tensor)
        pred_label = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item() * 100

        ax    = axes_grid[i // 5][i % 5]
        color = "green" if pred_label == true_label else "red"
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"Pred: {pred_label}\n({confidence:.1f}%)", color=color, fontsize=9)
        ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/sample_predictions.png")

# ── per-class accuracy ────────────────────────────────────────────────────────
per_class_correct = np.diag(cm)
per_class_total   = cm.sum(axis=1)
per_class_acc     = per_class_correct / per_class_total * 100

print("\n" + "=" * 65)
print("Per-Class Accuracy")
print("=" * 65)
for digit in range(10):
    print(f"  Digit {digit}: {per_class_correct[digit]:4d}/{per_class_total[digit]:4d}  "
          f"({per_class_acc[digit]:.1f}%)")

# ── final summary ─────────────────────────────────────────────────────────────
final_test_loss, final_test_acc = evaluate(test_loader)
print("\n" + "=" * 65)
print("FINAL RESULTS — Homework 4 (hr4625)")
print("=" * 65)
print(f"Test Accuracy : {final_test_acc:.2f}%")
print(f"Test Loss     : {final_test_loss:.4f}")
print(f"Training Time : {elapsed:.1f}s")
print(f"Device        : {device}")
print()
print("vs. Assignment 3 (CNN)         : 99.36%")
print("vs. Assignment 2 (Feedforward) : 98.09%")
print("=" * 65)
