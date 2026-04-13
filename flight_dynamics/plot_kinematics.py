import os
import torch
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_trajectory(preds, targets, output_dir, run_name):
    """Plots the 6 wing angles for trajectory 0 using Plotly."""
    
    # 1. Hardcode to index 0
    traj_idx = 0
    
    # Extract the specific sequence
    pred_seq = preds[traj_idx].numpy()     # Shape: (L, 6)
    target_seq = targets[traj_idx].numpy() # Shape: (L, 6)
    
    seq_length = pred_seq.shape[0]
    time_steps = list(range(seq_length))

    # The order Plotly populates titles (Row 1 Left, Row 1 Right, Row 2 Left...)
    grid_titles = [
        "Left Stroke Angle", "Right Stroke Angle",
        "Left Deviation Angle", "Right Deviation Angle",
        "Left Rotation Angle", "Right Rotation Angle"
    ]

    # Create a 3x2 grid of interactive plots
    fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=grid_titles,
        shared_xaxes=True,
        vertical_spacing=0.08
    )

    for i in range(6):
        # NEW LAYOUT MATH: Column-Major mapping
        # Indices 0,1,2 (Left wing) -> Column 1. Indices 3,4,5 (Right wing) -> Column 2.
        row = (i % 3) + 1
        col = (i // 3) + 1
        
        # Only show the legend on the very first trace to avoid duplicates
        show_legend = True if i == 0 else False

        # Plot Ground Truth
        fig.add_trace(
            go.Scatter(
                x=time_steps, 
                y=target_seq[:, i], 
                mode='lines', 
                name='Ground Truth',
                line=dict(color='rgba(0, 0, 0, 0.7)', width=2),
                showlegend=show_legend,
                legendgroup="ground_truth" # Links all GT traces together
            ),
            row=row, col=col
        )

        # Plot Prediction
        fig.add_trace(
            go.Scatter(
                x=time_steps, 
                y=pred_seq[:, i], 
                mode='lines', 
                name='S4 Prediction',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=show_legend,
                legendgroup="prediction" # Links all Pred traces together
            ),
            row=row, col=col
        )
        
        # Format axes
        fig.update_yaxes(title_text="Angle", row=row, col=col, title_font=dict(size=10))
        if row == 3:
            fig.update_xaxes(title_text="Time Steps", row=row, col=col)

    # Polish the layout
    fig.update_layout(
        title_text=f"<b>{run_name}</b> | Flight Kinematics - Trajectory #{traj_idx}",
        title_x=0.5,
        height=900,
        width=1200,
        template="plotly_white",
        hovermode="x unified", # Shows values for both lines when hovering
        legend=dict(
            yanchor="top", y=0.99, 
            xanchor="right", x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )

    # 2. Save the plot as an interactive HTML file
    save_path = os.path.join(output_dir, f"trajectory_plot_{traj_idx}.html")
    fig.write_html(save_path)
    
    print(f"  -> Saved interactive plot to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot Predicted vs Ground Truth Wing Angles (HTML)")
    parser.add_argument('--experiment_dir', type=str, required=True, help="Path to the parent experiment folder")
    parser.add_argument('--targets', type=str, required=True, help="Path to the actual ground truth targets.pt")
    args = parser.parse_args()

    print(f"==> Loading ground truth targets from {args.targets}...")
    targets = torch.load(args.targets, map_location='cpu')

    # Find all subdirectories
    subdirs = [os.path.join(args.experiment_dir, d) for d in os.listdir(args.experiment_dir) 
               if os.path.isdir(os.path.join(args.experiment_dir, d))]
    
    print(f"==> Found {len(subdirs)} potential run directories in {args.experiment_dir}")

    # Iterate through each grid search run
    for run_dir in subdirs:
        run_name = os.path.basename(run_dir)
        pred_path = os.path.join(run_dir, 'prediction.pt')

        # Safety Check
        if not os.path.exists(pred_path):
            print(f"\n[Skipping] {run_name}: No prediction.pt found.")
            continue

        print(f"\nProcessing: {run_name}")
        preds = torch.load(pred_path, map_location='cpu')

        # Dimension checks
        if len(preds) != len(targets):
            print(f"  -> [Warning] Mismatch: preds ({len(preds)}) != targets ({len(targets)}). Skipping.")
            continue

        # Generate and save the HTML plot
        plot_trajectory(preds, targets, run_dir, run_name)

    print("\n==> Plotting Pipeline Complete!")

if __name__ == '__main__':
    main()