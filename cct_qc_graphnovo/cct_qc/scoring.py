import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm

class PsmDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    def __init__(self, in_dim=33, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 1),
            nn.Sigmoid()            
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with tqdm(total=len(loader)) as pbar:
        for X, y in loader:
            X = X.to(device)
            y_hat = model(X).cpu()
            y_true.append(y)
            y_pred.append(y_hat)
            pbar.update(1)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred > 0.5))
    return auc, acc, y_true, y_pred


def train_mlp(X, y,
         lr=1e-4, batch_size=256, epochs=50, dropout=0.1,
         test_size=0.2, random_state=42, patience=8, device='cuda:0', ckpt_path=None):
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    if any(np.unique(labels).size != 2 for labels in (y_train, y_val)):
        raise ValueError(
            "QC training requires both label classes in the training and validation splits. "
            "Use more reference-labeled PSMs; peptide-level training also requires "
            "incorrect references after PSM filtering."
        )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)

    # 4.3 DataLoader
    train_loader = DataLoader(PsmDataset(X_train, y_train), batch_size=batch_size,
                              shuffle=True, drop_last=len(y_train) % batch_size == 1)
    val_loader   = DataLoader(PsmDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

    # 4.4 mode;, optimizer, loss
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(in_dim=np.array(X).shape[1], dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    # 4.5 train loop (early stopping)
    best_auc, epochs_no_improve = -float("inf"), 0
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        auc, acc, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Val AUC {auc:.4f} | Val Acc {acc:.4f}")

        # early stopping
        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(),
                        "scaler": scaler}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # 4.6 load best ckpt and report 
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    scaler = ckpt["scaler"]
    _, _, y_true, y_pred = evaluate(model, val_loader, device)
    print("\nFinal classification report (Val):")
    print(classification_report(y_true, (y_pred > 0.5)))
    return model


def gamma_fdr(df_fdr, fdr_thresh=0.05, min_gamma=0.01, max_gamma=100.0, eps=1e-4,
              conservative_fdr_ratio = 0.5, interval_margin = 0.1):
    """calculation of adjustment factor gamma."""
    true_fdr = np.array(df_fdr['observed_fdp'].tolist())
    estimated_fdr = np.array(df_fdr['estimated_fdr'].tolist())

    target_idx_list = np.where(np.abs(true_fdr - estimated_fdr)<=eps)[0] # score threshold (s_c) of conservatism starting point 
    if len(target_idx_list) == 0:
        target_idx = 0
    else:
        target_idx = max(target_idx_list)
    conservative_start_point = estimated_fdr[target_idx]  
    print(f"conservative start point: {conservative_start_point:.4f}, target fdr: {fdr_thresh:.4f}, start idx: {target_idx}")
    
    if conservative_start_point <= fdr_thresh * conservative_fdr_ratio or max(true_fdr)<=fdr_thresh:
        # Interval conservativeness theorem
        expected_interval_low = fdr_thresh * (1 - interval_margin)
        expected_interval_high = fdr_thresh * (1 + interval_margin)
        low_idx = np.argmin(np.abs(true_fdr - expected_interval_low))
        high_idx = np.argmin(np.abs(true_fdr - expected_interval_high))
        gamma_low = true_fdr[low_idx] / estimated_fdr[low_idx]
        gamma_high = true_fdr[high_idx] / estimated_fdr[high_idx]
        gamma = max([gamma_low, gamma_high])

        gamma = np.clip(gamma, min_gamma, max_gamma)
        df_fdr['estimated_fdr_update'] = gamma * df_fdr['estimated_fdr']
        if gamma == gamma_low:
            optimal_idx = high_idx
        else:
            optimal_idx = low_idx
        exact_low, exact_high = true_fdr[low_idx], true_fdr[high_idx]
        print(f"gamma of interval: interval[{expected_interval_low:.4f}, {expected_interval_high:.4f}], gamma={gamma:.4f}")
        print((f"conservative interval: interval[{exact_low:.4f}, {exact_high:.4f}], gamma={gamma:.4f}"))
    else:
        # Adjustment-factor theorem
        target_idx_list = np.where(np.abs(true_fdr - fdr_thresh)<=eps)[0] 
        if len(target_idx_list) == 0:
            target_idx = 0
        else:
            target_idx = max(target_idx_list)
        gamma = true_fdr[target_idx] / estimated_fdr[target_idx]  
        print(f"according to Adjustment-factor theorem: gamma={gamma:.4f} (nearby target fdr)")
        gamma = np.clip(gamma, min_gamma, max_gamma)
        df_fdr['estimated_fdr_update'] = gamma * df_fdr['estimated_fdr']
        nearby = np.flatnonzero(np.abs(df_fdr['estimated_fdr_update'] - fdr_thresh) <= eps)
        optimal_idx = int(nearby[-1]) if len(nearby) else None
        exact_low, exact_high = (true_fdr[optimal_idx], true_fdr[-1]) if optimal_idx is not None else (np.nan, np.nan)
        if optimal_idx is not None:
            print((f"conservative interval: interval[{exact_low:.4f}, {exact_high:.4f}], gamma={gamma:.4f}"))
    observed_at_threshold = true_fdr[optimal_idx] if optimal_idx is not None else np.nan
    return df_fdr, gamma, observed_at_threshold, optimal_idx, [exact_low, exact_high]


def calculate_fdr(df_file, score_name):
    df_sorted_psm = df_file.copy()
    df_sorted_psm = df_sorted_psm.sort_values(by=score_name, ascending=False).reset_index(drop=True)
    cum_target = (df_sorted_psm['is_decoy'] == 0).cumsum()
    df_sorted_psm['cum_decoy']  = (df_sorted_psm['is_decoy'] == 1).cumsum()
    df_sorted_psm['cum_decoy_true'] = ((df_sorted_psm['is_decoy'] == 1) & (df_sorted_psm['denovo_label'] == 1)).cumsum()
    df_sorted_psm['cum_decoy_false'] = ((df_sorted_psm['is_decoy'] == 1) & (df_sorted_psm['denovo_label'] == 0)).cumsum()
    df_sorted_psm['cum_true'] = ((df_sorted_psm['is_decoy'] == 0) & (df_sorted_psm['denovo_label'] == 1)).cumsum()
    df_sorted_psm['cum_false']  = ((df_sorted_psm['is_decoy'] == 0) & (df_sorted_psm['denovo_label'] == 0)).cumsum()
    df_sorted_psm['estimated_fdr'] = df_sorted_psm['cum_decoy'] / np.maximum(cum_target, 1)
    df_sorted_psm['observed_fdp']  = df_sorted_psm['cum_false'] / np.maximum((df_sorted_psm['cum_true']+df_sorted_psm['cum_false']), 1)
    return df_sorted_psm

