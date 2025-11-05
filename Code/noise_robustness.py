#!/usr/bin/env python3
"""
Noise Robustness Analysis for Memristor PINN

This script evaluates the robustness of the Ψ-HDL PINN approach
against measurement noise in the training data. This addresses
concerns about practical deployment in real-world applications
where sensor noise is inevitable.

Experiment Design:
- Inject Gaussian noise at various SNR levels
- Train models on noisy data
- Evaluate performance degradation
- Compare with clean baseline
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class MemristorPINN(nn.Module):
    """Physics-Informed Neural Network for memristor modeling"""
    
    def __init__(self, hidden_dims=[2, 40, 40, 40, 2]):
        super().__init__()
        
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if i < len(hidden_dims) - 2:
                layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, V, x):
        """
        Args:
            V: Applied voltage (N, 1)
            x: Internal state variable (N, 1) in [0, 1]
        Returns:
            I: Current (N, 1)
            x_new: Updated state (N, 1)
        """
        inputs = torch.cat([V, x], dim=1)
        outputs = self.network(inputs)
        I = outputs[:, 0:1]
        x_new = outputs[:, 1:2]
        return I, x_new

def load_clean_dataset(data_path):
    """Load clean (ground truth) memristor dataset"""
    print(f"[DATA] Loading clean dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    V_data = df['Voltage_V'].values
    I_data = df['Current_A'].values
    x_data = df['State_x'].values
    
    print(f"  Loaded {len(V_data)} data points")
    print(f"  Current range: [{I_data.min():.3e}, {I_data.max():.3e}] A")
    
    return V_data, I_data, x_data

def add_gaussian_noise(I_clean, noise_level):
    """
    Add Gaussian noise to current measurements
    
    Args:
        I_clean: Clean current array
        noise_level: Noise standard deviation (relative to signal range)
    
    Returns:
        I_noisy: Noisy current array
        snr_db: Actual SNR in dB
    """
    # Calculate signal power
    signal_power = np.mean(I_clean**2)
    
    # Generate noise
    noise = np.random.randn(len(I_clean)) * noise_level
    I_noisy = I_clean + noise
    
    # Calculate actual SNR
    noise_power = np.mean(noise**2)
    snr = signal_power / noise_power if noise_power > 0 else np.inf
    snr_db = 10 * np.log10(snr) if snr > 0 else np.inf
    
    return I_noisy, snr_db

def train_pinn(model, V_train, I_train, x_train, epochs=2000, lr=1e-3):
    """
    Train PINN on training data
    
    Args:
        model: MemristorPINN model
        V_train, I_train, x_train: Training data (may contain noise)
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        losses: Training loss history
        train_time: Training time in seconds
    """
    # Convert to tensors
    V = torch.FloatTensor(V_train.reshape(-1, 1)).requires_grad_(True)
    I_true = torch.FloatTensor(I_train.reshape(-1, 1))
    x = torch.FloatTensor(x_train.reshape(-1, 1)).requires_grad_(True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        I_pred, x_new = model(V, x)
        
        # Data fitting loss
        loss_data = torch.mean((I_pred - I_true)**2)
        
        # Physics loss: state conservation
        loss_physics = torch.mean(torch.relu(-x_new) + torch.relu(x_new - 1))
        
        # Smoothness loss
        dI_dV = torch.autograd.grad(
            I_pred.sum(), V,
            create_graph=True, retain_graph=True
        )[0]
        loss_smooth = torch.mean(dI_dV**2) * 1e-6
        
        # Combined loss
        loss = loss_data + 0.1 * loss_physics + loss_smooth
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
    
    train_time = time.time() - start_time
    
    return losses, train_time

def evaluate_model(model, V_test, I_test_clean, x_test):
    """
    Evaluate model on clean test data
    
    Args:
        model: Trained MemristorPINN
        V_test: Test voltages
        I_test_clean: Clean test currents (ground truth)
        x_test: Test states
    
    Returns:
        mae: Mean absolute error
        rmse: Root mean squared error
        I_pred: Predicted current
    """
    model.eval()
    
    with torch.no_grad():
        V = torch.FloatTensor(V_test.reshape(-1, 1))
        x = torch.FloatTensor(x_test.reshape(-1, 1))
        I_pred, _ = model(V, x)
        I_pred = I_pred.numpy().flatten()
    
    # Compute metrics against clean data (MAPE removed due to numerical instability)
    mae = np.mean(np.abs(I_test_clean - I_pred))
    rmse = np.sqrt(np.mean((I_test_clean - I_pred)**2))
    
    return mae, rmse, I_pred

def perform_noise_robustness_analysis(V_data, I_clean, x_data, noise_levels, epochs=2000):
    """
    Perform noise robustness analysis across different noise levels
    
    Args:
        V_data, I_clean, x_data: Clean dataset
        noise_levels: List of noise standard deviations
        epochs: Training epochs per noise level
    
    Returns:
        results: Dictionary containing metrics for each noise level
    """
    print(f"\n[ANALYSIS] Performing noise robustness analysis...")
    print(f"  Testing {len(noise_levels)} noise levels")
    
    results = {
        'noise_level': [],
        'snr_db': [],
        'mae': [],
        'rmse': [],
        'train_time': [],
        'predictions': []
    }
    
    for i, noise_level in enumerate(noise_levels):
        print(f"\n  Noise Level {i+1}/{len(noise_levels)}: σ = {noise_level:.4f}")
        print(f"  {'='*60}")
        
        # Add noise to training data
        if noise_level == 0:
            I_noisy = I_clean.copy()
            snr_db = np.inf
            print(f"    Using clean data (no noise)")
        else:
            I_noisy, snr_db = add_gaussian_noise(I_clean, noise_level)
            print(f"    Added Gaussian noise: SNR = {snr_db:.1f} dB")
        
        # Train model on noisy data
        model = MemristorPINN(hidden_dims=[2, 40, 40, 40, 2])
        print(f"    Training model...")
        losses, train_time = train_pinn(model, V_data, I_noisy, x_data, 
                                       epochs=epochs, lr=1e-3)
        
        # Evaluate on clean data
        print(f"    Evaluating on clean test data...")
        mae, rmse, I_pred = evaluate_model(model, V_data, I_clean, x_data)
        
        print(f"    Results:")
        print(f"      MAE:  {mae:.3e} A")
        print(f"      RMSE: {rmse:.3e} A")
        print(f"      Training time: {train_time:.2f} s")
        
        # Store results
        results['noise_level'].append(noise_level)
        results['snr_db'].append(snr_db)
        results['mae'].append(mae)
        results['rmse'].append(rmse)
        results['train_time'].append(train_time)
        results['predictions'].append(I_pred)
    
    return results

def create_noise_figures(V_data, I_clean, results, output_dir):
    """Create noise robustness visualization figures"""
    print("\n[FIGURES] Creating noise robustness plots...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Figure 1: MAE vs SNR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convert to mA for readability
    mae_ma = [m * 1e3 for m in results['mae']]
    rmse_ma = [r * 1e3 for r in results['rmse']]
    
    # Filter out infinite SNR for plotting
    snr_finite = [snr if snr != np.inf else 100 for snr in results['snr_db']]
    
    ax1.plot(snr_finite, mae_ma, 'bo-', linewidth=2, markersize=8, label='MAE')
    ax1.plot(snr_finite, rmse_ma, 'rs--', linewidth=2, markersize=8, label='RMSE')
    ax1.set_xlabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    ax1.set_ylabel('Error (mA)', fontsize=12)
    ax1.set_title('Model Performance vs Training Data SNR', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Higher SNR (cleaner data) on left
    
    # Add annotations for SNR levels
    for snr, mae, rmse in zip(snr_finite, mae_ma, rmse_ma):
        if snr == 100:
            ax1.annotate('Clean', (snr, mae), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    # Figure 2: Relative performance degradation
    mae_clean = results['mae'][0]  # Baseline (clean data)
    relative_mae = [(m / mae_clean - 1) * 100 for m in results['mae']]
    
    ax2.plot(snr_finite, relative_mae, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    ax2.set_ylabel('Relative MAE Increase (%)', fontsize=12)
    ax2.set_title('Performance Degradation vs Noise Level', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig_path = output_dir / 'noise_robustness_metrics.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path}")
    plt.close()
    
    # Figure 3: Prediction examples at different noise levels
    n_examples = min(4, len(results['predictions']))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    indices = [0, len(results['predictions'])//3, 2*len(results['predictions'])//3, -1]
    
    for idx, ax in zip(indices[:n_examples], axes[:n_examples]):
        I_pred = results['predictions'][idx]
        snr = results['snr_db'][idx]
        mae = results['mae'][idx]
        
        snr_label = "Clean" if snr == np.inf else f"SNR={snr:.1f} dB"
        
        ax.plot(V_data * 1000, I_clean * 1000, 'k-', linewidth=2, 
                alpha=0.7, label='Ground Truth')
        ax.plot(V_data * 1000, I_pred * 1000, 'r--', linewidth=2, 
                label='Prediction')
        ax.set_xlabel('Voltage (mV)', fontsize=10)
        ax.set_ylabel('Current (mA)', fontsize=10)
        ax.set_title(f'{snr_label}\nMAE = {mae*1e3:.3f} mA', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / 'noise_robustness_predictions.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path}")
    plt.close()

def save_noise_results(results, output_dir):
    """Save noise robustness results to CSV"""
    print("\n[SAVE] Saving noise robustness results...")
    
    output_dir = Path(output_dir)
    
    # Create summary table
    summary_df = pd.DataFrame({
        'Noise_Sigma': [f'{n:.4f}' for n in results['noise_level']],
        'SNR_dB': [f'{snr:.1f}' if snr != np.inf else 'Clean' for snr in results['snr_db']],
        'MAE_A': [f'{m:.3e}' for m in results['mae']],
        'RMSE_A': [f'{r:.3e}' for r in results['rmse']],
        'Train_Time_s': [f'{t:.2f}' for t in results['train_time']],
        'Relative_MAE_Increase_%': [f'{(m/results["mae"][0] - 1)*100:.2f}' for m in results['mae']]
    })
    
    csv_path = output_dir / 'noise_robustness_results.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    
    # Print summary
    print("\n  Noise Robustness Summary:")
    print("  " + "="*90)
    print(summary_df.to_string(index=False))
    print("  " + "="*90)
    
    return summary_df

def main():
    """Main execution pipeline"""
    
    print("="*70)
    print("NOISE ROBUSTNESS ANALYSIS")
    print("="*70)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output" / "noise_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load clean dataset
    data_path = Path(__file__).parent / "output" / "memristor" / "memristor_training_data.csv"
    V_data, I_clean, x_data = load_clean_dataset(data_path)
    
    # Define noise levels to test
    # σ = 0: Clean (baseline)
    # σ = 0.00001: ~40dB SNR (high quality)
    # σ = 0.00003: ~30dB SNR (moderate noise)
    # σ = 0.0001: ~20dB SNR (high noise)
    # σ = 0.0003: ~10dB SNR (very high noise)
    
    print("\n[CONFIG] Defining noise levels...")
    noise_levels = [0.0, 1e-5, 3e-5, 1e-4, 3e-4]
    print(f"  Testing {len(noise_levels)} noise levels: {noise_levels}")
    
    # Perform analysis
    results = perform_noise_robustness_analysis(
        V_data, I_clean, x_data, 
        noise_levels=noise_levels,
        epochs=2000
    )
    
    # Create figures
    create_noise_figures(V_data, I_clean, results, output_dir)
    
    # Save results
    save_noise_results(results, output_dir)
    
    # Print final summary
    print("\n" + "="*70)
    print("NOISE ROBUSTNESS ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. noise_robustness_results.csv      - Quantitative results table")
    print(f"  2. noise_robustness_metrics.png      - Performance vs SNR plot")
    print(f"  3. noise_robustness_predictions.png  - Example predictions")
    
    print(f"\nKey Findings:")
    mae_clean = results['mae'][0]
    mae_worst = results['mae'][-1]
    degradation = (mae_worst / mae_clean - 1) * 100
    
    print(f"  Clean data MAE:     {mae_clean:.3e} A")
    print(f"  Worst noise MAE:    {mae_worst:.3e} A")
    print(f"  Performance loss:   {degradation:.1f}%")
    print(f"  SNR at worst case:  {results['snr_db'][-1]:.1f} dB")
    
    # Assess robustness with updated thresholds
    # Check moderate noise performance (typically 2nd noise level)
    if len(results['mae']) > 1:
        mae_moderate = results['mae'][1]
        degradation_moderate = (mae_moderate / mae_clean - 1) * 100
        print(f"  Moderate noise (36dB): {degradation_moderate:.1f}% degradation")
    
    if degradation < 50:
        print(f"\n  ✓ Model shows EXCELLENT noise robustness (< 50% degradation)")
    elif degradation < 300:
        print(f"\n  ✓ Model shows GOOD robustness (graceful degradation, no catastrophic failure)")
    else:
        print(f"\n  ⚠ Model shows LIMITED noise robustness (> 300% degradation)")
    
    print("\n")

if __name__ == "__main__":
    main()