#!/usr/bin/env python3
"""
VTEAM Baseline Model for Memristor Comparison

This implements the Voltage Threshold Adaptive Memristor (VTEAM) model
as a baseline for quantitative comparison with the Ψ-HDL PINN approach.

VTEAM is an industry-standard memristor compact model widely used in
circuit simulation and neuromorphic computing applications.

Reference: Kvatinsky et al., "TEAM: ThrEshold Adaptive Memristor Model",
IEEE Trans. Circuits Syst. I, 2013
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
import time

class VTEAMModel:
    """
    VTEAM (Voltage Threshold Adaptive Memristor) Model
    
    State equation:
        dw/dt = k_on * (V/V_on - 1)^α_on    if V > V_on
        dw/dt = k_off * (V/V_off - 1)^α_off if V < V_off
        dw/dt = 0                            otherwise
    
    Resistance:
        R(w) = R_on * w + R_off * (1 - w)
    
    Current:
        I = V / R(w)
    """
    
    def __init__(self, R_on=1e3, R_off=1e5, V_on=0.8, V_off=-0.6, 
                 k_on=1e3, k_off=1e3, alpha_on=2.0, alpha_off=2.0):
        """
        Initialize VTEAM model parameters
        
        Args:
            R_on: ON-state resistance (Ω)
            R_off: OFF-state resistance (Ω)
            V_on: Positive threshold voltage (V)
            V_off: Negative threshold voltage (V)
            k_on: Rate constant for SET transition
            k_off: Rate constant for RESET transition
            alpha_on: Nonlinearity exponent for SET
            alpha_off: Nonlinearity exponent for RESET
        """
        self.R_on = R_on
        self.R_off = R_off
        self.V_on = V_on
        self.V_off = V_off
        self.k_on = k_on
        self.k_off = k_off
        self.alpha_on = alpha_on
        self.alpha_off = alpha_off
        
        # For tracking parameter count
        self.n_params = 8
    
    def state_derivative(self, V, w):
        """
        Calculate state variable derivative dw/dt
        
        Args:
            V: Applied voltage (scalar or array)
            w: State variable [0, 1] (scalar or array)
        
        Returns:
            dw/dt: State derivative
        """
        dwdt = np.zeros_like(V)
        
        # SET transition (V > V_on)
        mask_on = V > self.V_on
        if np.any(mask_on):
            dwdt[mask_on] = self.k_on * ((V[mask_on] / self.V_on - 1) ** self.alpha_on)
        
        # RESET transition (V < V_off)
        mask_off = V < self.V_off
        if np.any(mask_off):
            dwdt[mask_off] = -self.k_off * ((V[mask_off] / self.V_off - 1) ** self.alpha_off)
        
        return dwdt
    
    def resistance(self, w):
        """
        Calculate state-dependent resistance
        
        Args:
            w: State variable [0, 1]
        
        Returns:
            R: Resistance (Ω)
        """
        return self.R_on * w + self.R_off * (1 - w)
    
    def current(self, V, w):
        """
        Calculate current through memristor
        
        Args:
            V: Applied voltage
            w: State variable [0, 1]
        
        Returns:
            I: Current (A)
        """
        R = self.resistance(w)
        return V / R
    
    def simulate(self, V_data, dt=0.01, w0=0.1):
        """
        Simulate memristor response to voltage input
        
        Args:
            V_data: Voltage time series (N,)
            dt: Time step (s)
            w0: Initial state
        
        Returns:
            I_pred: Predicted current (N,)
            w_states: State evolution (N,)
        """
        N = len(V_data)
        I_pred = np.zeros(N)
        w_states = np.zeros(N)
        
        w = w0
        
        for i in range(N):
            V = V_data[i]
            
            # Calculate current
            I_pred[i] = self.current(V, w)
            w_states[i] = w
            
            # Update state
            dwdt = self.state_derivative(V, w)
            w = np.clip(w + dwdt * dt, 0, 1)
        
        return I_pred, w_states

def load_training_data(data_path):
    """Load memristor training data from CSV"""
    print(f"[DATA] Loading training data from {data_path}")
    df = pd.read_csv(data_path)
    
    V_data = df['Voltage_V'].values
    I_data = df['Current_A'].values
    x_data = df['State_x'].values
    
    print(f"  Loaded {len(V_data)} data points")
    print(f"  Voltage range: [{V_data.min():.2f}, {V_data.max():.2f}] V")
    print(f"  Current range: [{I_data.min():.2e}, {I_data.max():.2e}] A")
    
    return V_data, I_data, x_data

def fit_vteam_model(V_data, I_data, initial_params=None):
    """
    Fit VTEAM model parameters to training data using optimization
    
    Args:
        V_data: Voltage array (N,)
        I_data: Current array (N,)
        initial_params: Initial parameter guess [R_on, R_off, V_on, V_off, k_on, k_off, alpha_on, alpha_off]
    
    Returns:
        model: Fitted VTEAMModel
        result: Optimization result
    """
    print("\n[TRAIN] Fitting VTEAM model to data...")
    
    if initial_params is None:
        # Use physically reasonable initial guess
        initial_params = [1e3, 1e5, 0.8, -0.6, 1e3, 1e3, 2.0, 2.0]
    
    start_time = time.time()
    
    def objective(params):
        """Loss function: Mean squared error of current prediction"""
        R_on, R_off, V_on, V_off, k_on, k_off, alpha_on, alpha_off = params
        
        # Create model with current parameters
        model = VTEAMModel(R_on, R_off, V_on, V_off, k_on, k_off, alpha_on, alpha_off)
        
        # Simulate
        try:
            I_pred, _ = model.simulate(V_data, dt=0.01, w0=0.1)
            
            # Mean squared error
            mse = np.mean((I_pred - I_data)**2)
            return mse
        except:
            # Return large penalty if simulation fails
            return 1e10
    
    # Parameter bounds (must be positive, reasonable ranges)
    bounds = [
        (1e2, 1e4),      # R_on
        (1e4, 1e6),      # R_off
        (0.5, 1.5),      # V_on
        (-1.0, -0.3),    # V_off
        (1e2, 1e4),      # k_on
        (1e2, 1e4),      # k_off
        (1.0, 5.0),      # alpha_on
        (1.0, 5.0)       # alpha_off
    ]
    
    # Optimize
    print("  Running optimization (this may take a few minutes)...")
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'disp': True}
    )
    
    train_time = time.time() - start_time
    
    # Create final model
    optimal_params = result.x
    model = VTEAMModel(*optimal_params)
    
    print(f"\n  Optimization complete! Training time: {train_time:.2f} s")
    print(f"  Final loss (MSE): {result.fun:.6e}")
    print(f"\n  Fitted parameters:")
    print(f"    R_on     = {model.R_on:.3e} Ω")
    print(f"    R_off    = {model.R_off:.3e} Ω")
    print(f"    V_on     = {model.V_on:.3f} V")
    print(f"    V_off    = {model.V_off:.3f} V")
    print(f"    k_on     = {model.k_on:.3e}")
    print(f"    k_off    = {model.k_off:.3e}")
    print(f"    alpha_on = {model.alpha_on:.3f}")
    print(f"    alpha_off = {model.alpha_off:.3f}")
    
    return model, result, train_time

def compute_metrics(I_true, I_pred):
    """Compute error metrics"""
    mae = np.mean(np.abs(I_true - I_pred))
    rmse = np.sqrt(np.mean((I_true - I_pred)**2))
    # Note: MAPE removed due to numerical instability near zero-crossing
    
    return mae, rmse

def create_comparison_figure(V_data, I_data, I_vteam, I_pinn, output_dir):
    """Create side-by-side comparison figure"""
    print("\n[FIGURES] Creating comparison plots...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Figure 1: Overlay comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(V_data * 1000, I_data * 1000, 'k-', linewidth=2, alpha=0.7, label='Ground Truth')
    ax.plot(V_data * 1000, I_vteam * 1000, 'b--', linewidth=2, label='VTEAM Model')
    ax.plot(V_data * 1000, I_pinn * 1000, 'r:', linewidth=2, label='Ψ-HDL PINN')
    ax.set_xlabel('Voltage (mV)', fontsize=12)
    ax.set_ylabel('Current (mA)', fontsize=12)
    ax.set_title('Memristor I-V Characteristics: VTEAM vs Ψ-HDL Comparison', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = output_dir / 'vteam_pinn_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path}")
    plt.close()
    
    # Figure 2: Error comparison
    error_vteam = np.abs(I_data - I_vteam)
    error_pinn = np.abs(I_data - I_pinn)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(V_data, error_vteam * 1000, 'b-', linewidth=1.5, alpha=0.7, label='VTEAM Error')
    ax1.plot(V_data, error_pinn * 1000, 'r-', linewidth=1.5, alpha=0.7, label='Ψ-HDL Error')
    ax1.set_xlabel('Voltage (V)', fontsize=11)
    ax1.set_ylabel('Absolute Error (mA)', fontsize=11)
    ax1.set_title('Current Prediction Error vs Voltage', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.hist([error_vteam * 1000, error_pinn * 1000], bins=50, 
             label=['VTEAM', 'Ψ-HDL'], alpha=0.6, color=['blue', 'red'])
    ax2.set_xlabel('Absolute Error (mA)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    fig_path = output_dir / 'error_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path}")
    plt.close()

def save_comparison_table(mae_vteam, rmse_vteam, mae_pinn, rmse_pinn, 
                          n_params_vteam, n_params_pinn, train_time_vteam, 
                          train_time_pinn, output_dir):
    """Save quantitative comparison table"""
    print("\n[TABLE] Creating comparison table...")
    
    output_dir = Path(output_dir)
    
    comparison_data = {
        'Model': ['VTEAM Baseline', 'Ψ-HDL PINN'],
        'MAE (A)': [f'{mae_vteam:.3e}', f'{mae_pinn:.3e}'],
        'RMSE (A)': [f'{rmse_vteam:.3e}', f'{rmse_pinn:.3e}'],
        'Parameters': [n_params_vteam, n_params_pinn],
        'Training Time (s)': [f'{train_time_vteam:.1f}', f'{train_time_pinn:.1f}'],
        'Improvement': ['-', f'{(1 - mae_pinn/mae_vteam)*100:+.1f}%']
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = output_dir / 'vteam_pinn_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    
    # Print to console
    print("\n  Quantitative Comparison:")
    print("  " + "="*80)
    print(df.to_string(index=False))
    print("  " + "="*80)
    
    return df

def main():
    """Main execution pipeline"""
    
    print("="*70)
    print("VTEAM BASELINE COMPARISON")
    print("="*70)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output" / "vteam_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    data_path = Path(__file__).parent / "output" / "memristor" / "memristor_training_data.csv"
    V_data, I_data, _ = load_training_data(data_path)
    
    # Fit VTEAM model
    vteam_model, result, train_time_vteam = fit_vteam_model(V_data, I_data)
    
    # Generate VTEAM predictions
    print("\n[PREDICT] Generating VTEAM predictions...")
    I_vteam, w_vteam = vteam_model.simulate(V_data, dt=0.01, w0=0.1)
    
    # Compute VTEAM metrics
    mae_vteam, rmse_vteam = compute_metrics(I_data, I_vteam)
    print(f"  VTEAM MAE:  {mae_vteam:.3e} A")
    print(f"  VTEAM RMSE: {rmse_vteam:.3e} A")
    
    # Load PINN predictions (from demo_memristor.py output)
    print("\n[COMPARE] Loading Ψ-HDL PINN results...")
    pinn_model_exists = Path(__file__).parent / "output" / "memristor" / "memristor_pinn.pth"
    
    if pinn_model_exists.exists():
        # Load PINN model and generate predictions
        import torch
        import torch.nn as nn
        
        class MemristorPINN(nn.Module):
            def __init__(self, hidden_dims=[2, 40, 40, 40, 2]):
                super().__init__()
                layers = []
                for i in range(len(hidden_dims) - 1):
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                    if i < len(hidden_dims) - 2:
                        layers.append(nn.Tanh())
                self.network = nn.Sequential(*layers)
            
            def forward(self, V, x):
                inputs = torch.cat([V, x], dim=1)
                outputs = self.network(inputs)
                I = outputs[:, 0:1]
                x_new = outputs[:, 1:2]
                return I, x_new
        
        pinn_model = MemristorPINN()
        pinn_model.load_state_dict(torch.load(pinn_model_exists))
        pinn_model.eval()
        
        # Generate PINN predictions using actual state data from CSV
        df_pinn = pd.read_csv(data_path)
        x_data_pinn = df_pinn['State_x'].values
        
        with torch.no_grad():
            V_tensor = torch.FloatTensor(V_data.reshape(-1, 1))
            x_tensor = torch.FloatTensor(x_data_pinn.reshape(-1, 1))
            I_pinn, _ = pinn_model(V_tensor, x_tensor)
            I_pinn = I_pinn.numpy().flatten()
        
        mae_pinn, rmse_pinn = compute_metrics(I_data, I_pinn)
        print(f"  PINN MAE:  {mae_pinn:.3e} A")
        print(f"  PINN RMSE: {rmse_pinn:.3e} A")
        
        # Count PINN parameters
        n_params_pinn = sum(p.numel() for p in pinn_model.parameters())
        train_time_pinn = 180.0  # Approximate from demo (3000 epochs)
        
        # Create comparison figures
        create_comparison_figure(V_data, I_data, I_vteam, I_pinn, output_dir)
        
        # Save comparison table
        save_comparison_table(
            mae_vteam, rmse_vteam, mae_pinn, rmse_pinn,
            vteam_model.n_params, n_params_pinn,
            train_time_vteam, train_time_pinn,
            output_dir
        )
    else:
        print("  Warning: PINN model not found. Run demo_memristor.py first.")
        print("  Saving VTEAM results only...")
    
    # Save VTEAM predictions
    vteam_results = pd.DataFrame({
        'Voltage_V': V_data,
        'Current_True_A': I_data,
        'Current_VTEAM_A': I_vteam,
        'State_w': w_vteam
    })
    vteam_csv = output_dir / 'vteam_predictions.csv'
    vteam_results.to_csv(vteam_csv, index=False)
    print(f"\n[SAVE] VTEAM results saved to: {vteam_csv}")
    
    print("\n" + "="*70)
    print("VTEAM BASELINE COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. vteam_predictions.csv        - VTEAM model predictions")
    print(f"  2. vteam_pinn_comparison.csv    - Quantitative comparison table")
    print(f"  3. vteam_pinn_comparison.png    - I-V curve comparison figure")
    print(f"  4. error_comparison.png         - Error analysis figure")
    print("\n")

if __name__ == "__main__":
    main()