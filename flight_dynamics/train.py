import os
import sys
import json
import argparse
import itertools
import math
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

# Import custom modules
from normalizer import NormalizerFactory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from models.s4.s4d import S4D

# Fix for PyTorch 1.11+ Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


# ==========================================
# 1. DIRECTORY & CONFIG MANAGEMENT
# ==========================================
def create_directory_structure(working_dir, experiment_name, conf_idx=None):
    # Formatted without colons to avoid filesystem errors
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    
    if conf_idx is not None:
        dir_name = f"{experiment_name}{conf_idx}_{timestamp}"
    else:
        dir_name = f"{experiment_name}_{timestamp}"
    
    run_dir = os.path.join(working_dir, "runs", dir_name)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return run_dir, checkpoint_dir


# ==========================================
# 2. DATASET & COLLATION
# ==========================================
def pad_collate(batch):
    Xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    
    # Record original true lengths
    lengths = torch.tensor([len(x) for x in Xs])
    max_len_actual = lengths.max().item()
    
    # Calculate the next power of 2 above the actual max length
    # e.g., if max_len is 853, next_pow_2 becomes 1024
    next_pow_2 = 2 ** math.ceil(math.log2(max_len_actual))
    
    # Pad to max_len_actual first to stack them into a batch
    X_padded = pad_sequence(Xs, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)
    
    # Pad the remaining difference up to next_pow_2
    pad_amount = next_pow_2 - max_len_actual
    if pad_amount > 0:
        # F.pad format for 3D tensor (Batch, Length, Features): 
        # (pad_features_left, pad_features_right, pad_length_left, pad_length_right)
        X_padded = F.pad(X_padded, (0, 0, 0, pad_amount), value=0.0)
        y_padded = F.pad(y_padded, (0, 0, 0, pad_amount), value=0.0)
    
    # Create the boolean mask using the actual lengths, but sized to next_pow_2
    # Shape: (Batch, next_pow_2)
    mask = torch.arange(next_pow_2).expand(len(lengths), next_pow_2) < lengths.unsqueeze(1)
    
    return X_padded, y_padded, mask

class InsectFlightSeq2SeqDataset(Dataset):
    def __init__(self, X_data, y_data, feature_scaler=None, target_scaler=None, is_train=True):
        self.X = X_data
        self.y = y_data
        assert len(self.X) == len(self.y), "Mismatch between number of feature and target trajectories."

        if feature_scaler is not None:
            if is_train:
                self.X = feature_scaler.fit_transform(self.X)
            else:
                self.X = feature_scaler.transform(self.X)

        if target_scaler is not None:
            if is_train:
                self.y = target_scaler.fit_transform(self.y)
            else:
                self.y = target_scaler.transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class S4Seq2SeqModel(nn.Module):
    def __init__(self, d_input=12, d_output=6, d_model=128, n_layers=4, dropout=0.1, prenorm=False):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=0.001))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = self.encoder(x)  
        x = x.transpose(-1, -2)
        
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2) 
        x = self.decoder(x)  
        return x


# ==========================================
# 4. UTILITIES & TRAINING LOOPS
# ==========================================
def setup_optimizer(model, lr, weight_decay, epochs):
    all_parameters = list(model.parameters())
    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
    
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler

def train_epoch(epoch, model, dataloader, optimizer, device, disable_tqdm=False):
    model.train()
    train_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=disable_tqdm)
    
    for batch_idx, (inputs, targets, mask) in pbar:
        inputs = inputs.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss_unreduced = F.mse_loss(outputs, targets, reduction='none').mean(dim=-1) 
        masked_loss = loss_unreduced * mask
        loss = masked_loss.sum() / mask.sum()
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_description(f'Train Epoch: {epoch} | Masked MSE: {train_loss/(batch_idx+1):.4f}')

def evaluate(epoch, model, dataloader, device, optimizer, checkpoint_dir, is_val=True, best_val_loss=float('inf'), disable_tqdm=False):
    model.eval()
    eval_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=disable_tqdm)
        for batch_idx, (inputs, targets, mask) in pbar:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.float32)
            
            outputs = model(inputs) 
            
            loss_unreduced = F.mse_loss(outputs, targets, reduction='none').mean(dim=-1)
            masked_loss = loss_unreduced * mask
            loss = masked_loss.sum() / mask.sum()

            eval_loss += loss.item()
            avg_loss = eval_loss / (batch_idx + 1)

            mode = "Val" if is_val else "Test"
            pbar.set_description(f'{mode} Epoch: {epoch} | Masked MSE: {avg_loss:.4f}')

    if is_val and avg_loss < best_val_loss:
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'val_loss': avg_loss
        }
        save_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, save_path)
        best_val_loss = avg_loss

    return avg_loss, best_val_loss


# ==========================================
# 5. PIPELINE EXECUTION
# ==========================================
def run_training_pipeline(config, X_train, y_train, X_val, y_val, run_dir, checkpoint_dir, device, conf_idx=None, disable_tqdm=False):
    """Encapsulates a single training run with a specific config dictionary."""
    run_label = f"Config {conf_idx}" if conf_idx else "Single Config"
    print(f"\n[{run_label}] Output Directory: {run_dir}")
    
    # Extract Hyperparameters for this specific run
    try:
        EPOCHS = config["epochs"]
        BATCH_SIZE = config["batch_size"]
        LR = config["lr"]
        WEIGHT_DECAY = config["weight_decay"]
        NUM_WORKERS = config["num_workers"]
        D_MODEL = config["d_model"]
        N_LAYERS = config["n_layers"]
        FEATURE_NORMALIZER = config["feature_normalizer"]
        TARGET_NORMALIZER = config["target_normalizer"]
    except KeyError as e:
        raise KeyError(f"Missing required hyperparameter in JSON config file: {e}")

    # Save a copy of this run's configuration
    config_path_out = os.path.join(run_dir, 'config.json')
    with open(config_path_out, 'w') as f:
        json.dump(config, f, indent=4)

    # Initialize Scalers (Reset for each run to avoid state contamination)
    feature_scaler = NormalizerFactory.create(FEATURE_NORMALIZER, global_normalizer=True)
    target_scaler = NormalizerFactory.create(TARGET_NORMALIZER, global_normalizer=True)

    # ---> ADD THESE LINES TO SAVE THE SCALERS <---
    with open(os.path.join(run_dir, 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(os.path.join(run_dir, 'target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_scaler, f)

    # Instantiate Datasets
    trainset = InsectFlightSeq2SeqDataset(X_train, y_train, feature_scaler=feature_scaler, target_scaler=target_scaler, is_train=True)
    valset = InsectFlightSeq2SeqDataset(X_val, y_val, feature_scaler=feature_scaler, target_scaler=target_scaler, is_train=False)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=pad_collate)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=pad_collate)

    # Initialize Model & Optimizer
    model = S4Seq2SeqModel(d_input=12, d_output=6, d_model=D_MODEL, n_layers=N_LAYERS).to(device)
    optimizer, scheduler = setup_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY, epochs=EPOCHS)

    # Training Loop
    best_val_loss = float('inf')
    print(f"[{run_label}] Starting Training Loop for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_epoch(epoch, model, trainloader, optimizer, device, disable_tqdm=disable_tqdm)
        val_loss, best_val_loss = evaluate(
            epoch, model, valloader, device, 
            optimizer=optimizer, checkpoint_dir=checkpoint_dir, 
            is_val=True, best_val_loss=best_val_loss, disable_tqdm=disable_tqdm
        )
        scheduler.step()
        print(f"[{run_label}] Epoch {epoch} | Val MSE: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    print(f"[{run_label}] Completed. Best model saved to: {checkpoint_dir}")


def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Train S4 Model with Grid Search")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument('--name', type=str, default="default_run", help="Name prefix for the output directory")
    parser.add_argument('--disable_tqdm', action='store_true', help="Manually turn off progress bars")
    args = parser.parse_args()

    DISABLE_PBARS = args.disable_tqdm or ('SLURM_JOB_ID' in os.environ)
    if DISABLE_PBARS:
        print("==> Slurm environment detected (or flag passed). Disabling tqdm progress bars to keep logs clean.")

    with open(args.config, 'r') as f:
        raw_config = json.load(f)

    # Extract non-grid parameters (Data paths and Working Directory)
    WORKING_DIR = raw_config.pop("working_directory", "flight_dynamics")
    FEATURES_FILE = raw_config.pop("features_file", "data/features.pt")
    TARGETS_FILE = raw_config.pop("targets_file", "data/targets.pt")
    
    TRAIN_RATIO = raw_config.pop("train_split_ratio", 0.85)

    # Build Configuration Grid
    listified_config = {k: (v if isinstance(v, list) else [v]) for k, v in raw_config.items()}
    keys, values = zip(*listified_config.items())
    config_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    num_configs = len(config_combinations)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cuda':
        print(f"==> CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True
        torch.backends.cuda.cufft_plan_cache.clear()
        torch.backends.cuda.cufft_plan_cache.max_size = 0

    print(f"==> Initiating on {DEVICE.upper()}")
    print(f"==> Found {num_configs} configuration(s) to execute.")

    # Load and split data globally ONCE
    print(f"==> Loading master dataset files...")
    X_full = torch.load(FEATURES_FILE, map_location='cpu')
    y_full = torch.load(TARGETS_FILE, map_location='cpu')

    total_samples = len(X_full)
    assert total_samples == len(y_full), "Mismatch in total features and targets."

    torch.manual_seed(42)
    indices = torch.randperm(total_samples).tolist()
    
    # --- YOUR FIX: Direct calculation based on the single float ---
    train_size = int(TRAIN_RATIO * total_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train = [X_full[i] for i in train_indices]
    y_train = [y_full[i] for i in train_indices]
    X_val = [X_full[i] for i in val_indices]
    y_val = [y_full[i] for i in val_indices]
    print(f"Dataset split: {len(X_train)} Train | {len(X_val)} Val")

    # Execute all configurations
    for i, config_instance in enumerate(config_combinations, start=1):
        if num_configs > 1:
            print(f"\n==========================================")
            print(f"   STARTING RUN {i} of {num_configs}")
            print(f"==========================================")
            
        # Create directory for this specific run
        run_idx = i if num_configs > 1 else None
        run_dir, checkpoint_dir = create_directory_structure(WORKING_DIR, args.name, run_idx)
        
        run_training_pipeline(
            config_instance, 
            X_train, y_train, X_val, y_val, 
            run_dir, checkpoint_dir, 
            DEVICE, conf_idx=run_idx,
            disable_tqdm=DISABLE_PBARS
        )

if __name__ == '__main__':
    main()