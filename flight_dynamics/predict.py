import os
import sys
import json
import torch
import pickle
import argparse
from tqdm.auto import tqdm

# Import your model class (adjust the import path if your filename is different!)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from flight_dynamics.train import S4Seq2SeqModel 

def main():
    parser = argparse.ArgumentParser(description="Generate Kinematic Predictions for a Grid Search Experiment")
    parser.add_argument('--experiment_dir', type=str, required=True, help="Path to the parent experiment folder (e.g., runs/reluctant_emu)")
    parser.add_argument('--features_path', type=str, required=True, help="Path to the new features .pt file")
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"==> Loading prediction environment on {DEVICE.upper()}...")

    # Load the raw features ONCE globally to save memory and time
    print(f"==> Loading features from {args.features_path}...")
    X_raw_list = torch.load(args.features_path, map_location='cpu')
    print(f"==> Found {len(X_raw_list)} trajectories to predict per model.")

    # Find all subdirectories in the experiment folder
    subdirs = [os.path.join(args.experiment_dir, d) for d in os.listdir(args.experiment_dir) 
               if os.path.isdir(os.path.join(args.experiment_dir, d))]
    
    print(f"==> Found {len(subdirs)} potential run directories in {args.experiment_dir}")

    # Iterate through each grid search run
    for run_dir in subdirs:
        run_name = os.path.basename(run_dir)
        config_path = os.path.join(run_dir, 'config.json')
        checkpoint_path = os.path.join(run_dir, 'checkpoints', 'best_model.pth')
        
        # Safety Check: Ensure this directory actually finished training and has the required files
        if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
            print(f"\n[Skipping] {run_name}: Missing config.json or best_model.pth")
            continue

        print(f"\n==========================================")
        print(f" Processing: {run_name}")
        print(f"==========================================")

        # 1. Load the Configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 2. Load the Normalizers
        with open(os.path.join(run_dir, 'feature_scaler.pkl'), 'rb') as f:
            feature_scaler = pickle.load(f)
        with open(os.path.join(run_dir, 'target_scaler.pkl'), 'rb') as f:
            target_scaler = pickle.load(f)

        # 3. Rebuild the Empty Model
        model = S4Seq2SeqModel(
            d_input=12, 
            d_output=6, 
            d_model=config['d_model'], 
            n_layers=config['n_layers']
        ).to(DEVICE)

        # 4. Inject the Saved Weights
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        predictions_list = []

        # 5. Predict Loop
        with torch.no_grad():
            for x_seq in tqdm(X_raw_list, desc=f"Predicting {run_name}", leave=False):
                # A. Scale the input and move to device
                x_scaled = feature_scaler.transform(x_seq.unsqueeze(0)).to(DEVICE, dtype=torch.float32)

                # B. Forward pass
                y_pred_scaled = model(x_scaled) # Shape: (1, L, 6)

                # C. Un-scale back to real physical angles and remove batch dim
                y_pred_real = target_scaler.inverse_transform(y_pred_scaled.cpu())
                predictions_list.append(y_pred_real.squeeze(0))

        # 6. Save the final predictions directly into the individual model's directory
        save_path = os.path.join(run_dir, 'prediction.pt')
        torch.save(predictions_list, save_path)
        print(f"Saved: {save_path}")

    print("\n==> Experiment Prediction Pipeline Complete!")

if __name__ == '__main__':
    main()