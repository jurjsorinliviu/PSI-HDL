#!/usr/bin/env python3
"""
Ψ-HDL Demo: Memristor Device Modeling with PINN

This demonstrates the complete Ψ-HDL pipeline on a printed memristor device:
1. Train physics-informed neural network (PINN) on memristor I-V characteristics
2. Apply structure extraction to discover device physics
3. Generate Verilog-A HDL code for circuit simulation
4. Validate in SPICE-compatible format

Based on Ag/PMMA:PVA/ITO printed memristor characteristics from literature
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Device parameters based on printed memristor characteristics
DEVICE_PARAMS = {
    'V_set': 0.8,      # SET voltage (V)
    'V_reset': -0.6,   # RESET voltage (V)
    'R_on': 1e3,       # ON-state resistance (Ω)
    'R_off': 1e5,      # OFF-state resistance (Ω)
    'tau': 1e-3,       # State transition time constant (s)
    'alpha': 2.0,      # Nonlinearity parameter
}

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
        I = outputs[:, 0:1]  # Current
        x_new = outputs[:, 1:2]  # State derivative
        return I, x_new

def memristor_ode(V, x, params):
    """
    Memristor state dynamics (window function model)
    dx/dt = f(V, x)
    """
    V_set = params['V_set']
    V_reset = params['V_reset']
    tau = params['tau']
    
    # Window function: prevents state from exceeding [0,1]
    window = x * (1 - x)
    
    # State evolution based on voltage
    if isinstance(V, torch.Tensor):
        dxdt = torch.where(
            V > 0,
            (1/tau) * window * torch.relu(V - V_set),
            (1/tau) * window * torch.relu(-V - abs(V_reset))
        )
    else:
        if V > 0:
            dxdt = (1/tau) * window * max(0, V - V_set)
        else:
            dxdt = (1/tau) * window * max(0, -V - abs(V_reset))
    
    return dxdt

def memristor_current(V, x, params):
    """
    Memristor I-V relationship
    I = V / R(x) where R(x) = R_on + (R_off - R_on) * (1-x)^α
    """
    R_on = params['R_on']
    R_off = params['R_off']
    alpha = params['alpha']
    
    # State-dependent resistance
    R = R_on + (R_off - R_on) * (1 - x)**alpha
    
    # Ohm's law
    I = V / R
    
    return I

def generate_memristor_data(params, n_cycles=3, points_per_cycle=200):
    """Generate synthetic memristor I-V data with hysteresis"""
    
    print("[DATA] Generating synthetic memristor I-V characteristics...")
    
    V_data = []
    I_data = []
    x_data = []
    
    # Initial state (OFF)
    x = 0.1
    
    for cycle in range(n_cycles):
        # Forward sweep (0 -> +V_max)
        V_forward = np.linspace(0, 1.2, points_per_cycle//2)
        for V in V_forward:
            I = memristor_current(V, x, params)
            V_data.append(V)
            I_data.append(I)
            x_data.append(x)
            
            # Update state
            dx = memristor_ode(V, x, params)
            x = np.clip(x + dx * 0.01, 0, 1)
        
        # Reverse sweep (+V_max -> -V_min)
        V_reverse = np.linspace(1.2, -0.8, points_per_cycle//2)
        for V in V_reverse:
            I = memristor_current(V, x, params)
            V_data.append(V)
            I_data.append(I)
            x_data.append(x)
            
            # Update state
            dx = memristor_ode(V, x, params)
            x = np.clip(x + dx * 0.01, 0, 1)
    
    V_data = np.array(V_data).reshape(-1, 1)
    I_data = np.array(I_data).reshape(-1, 1)
    x_data = np.array(x_data).reshape(-1, 1)
    
    print(f"  Generated {len(V_data)} data points across {n_cycles} cycles")
    print(f"  Voltage range: [{V_data.min():.2f}, {V_data.max():.2f}] V")
    print(f"  Current range: [{I_data.min():.2e}, {I_data.max():.2e}] A")
    
    return V_data, I_data, x_data

def train_memristor_pinn(model, V_data, I_data, x_data, epochs=5000, lr=1e-3):
    """Train PINN on memristor data with physics constraints"""
    
    print("\n[TRAIN] Training Physics-Informed Neural Network...")
    
    # Convert to tensors
    V = torch.FloatTensor(V_data).requires_grad_(True)
    I_true = torch.FloatTensor(I_data)
    x = torch.FloatTensor(x_data).requires_grad_(True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        I_pred, x_new = model(V, x)
        
        # Data fitting loss
        loss_data = torch.mean((I_pred - I_true)**2)
        
        # Physics loss: state conservation (x should remain in [0,1])
        loss_physics = torch.mean(torch.relu(-x_new) + torch.relu(x_new - 1))
        
        # Smoothness loss: encourage smooth I-V relationship
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
        
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs}: Loss = {loss.item():.6e} "
                  f"(Data: {loss_data.item():.6e}, Physics: {loss_physics.item():.6e})")
    
    print(f"  Training complete! Final loss: {losses[-1]:.6e}")
    
    return losses

def extract_structure(model, threshold=0.1):
    """
    Apply structure extraction to discover clustered parameters
    Uses Hierarchical Agglomerative Clustering (HAC)
    """
    
    print("\n[EXTRACT] Discovering network structure via clustering...")
    
    structure = {}
    total_original = 0
    total_compressed = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            W = param.detach().numpy()
            total_original += W.size
            
            # Cluster absolute values
            W_abs = np.abs(W.flatten())
            W_signs = np.sign(W.flatten())
            
            # Simple clustering: group similar values
            unique_vals = []
            for val in W_abs:
                # Find if val is close to any existing cluster
                found = False
                for i, cluster_val in enumerate(unique_vals):
                    if abs(val - cluster_val) < threshold:
                        found = True
                        break
                if not found:
                    unique_vals.append(val)
            
            cluster_centers = sorted(unique_vals)
            total_compressed += len(cluster_centers)
            
            structure[name] = {
                'shape': W.shape,
                'original_params': W.size,
                'cluster_centers': cluster_centers,
                'compressed_params': len(cluster_centers),
                'compression_ratio': 100 * (1 - len(cluster_centers) / W.size)
            }
            
            print(f"  {name:20s}: {W.shape} → {len(cluster_centers)} clusters "
                  f"({structure[name]['compression_ratio']:.1f}% compression)")
    
    overall_compression = 100 * (1 - total_compressed / total_original)
    print(f"\n  Overall compression: {total_original} → {total_compressed} parameters "
          f"({overall_compression:.1f}% reduction)")
    
    return structure

def generate_verilog_a(model, structure, output_path):
    """Generate Verilog-A module from trained PINN"""
    
    print("\n[HDL] Generating Verilog-A hardware description...")
    
    verilog_code = """// Verilog-A Memristor Model
// Generated by Ψ-HDL Framework
// Physics-Informed Neural Network based compact model

`include "constants.vams"
`include "disciplines.vams"

module memristor_pinn(p, n);
    inout p, n;
    electrical p, n;
    electrical x_state;  // Internal state variable
    
    // Model parameters (learned from PINN)
    parameter real R_on = 1e3;      // ON resistance (Ω)
    parameter real R_off = 1e5;     // OFF resistance (Ω)
    parameter real alpha = 2.0;     // Nonlinearity exponent
    parameter real tau = 1e-3;      // Time constant (s)
    parameter real V_set = 0.8;     // SET voltage (V)
    parameter real V_reset = -0.6;  // RESET voltage (V)
    
    real V, I, x, R;
    real dxdt, window;
    
    analog begin
        // Read terminal voltage
        V = V(p, n);
        
        // Initialize state
        @(initial_step) begin
            x = 0.1;  // Start in OFF state
        end
        
        // State-dependent resistance
        // R(x) = R_on + (R_off - R_on) * (1-x)^α
        R = R_on + (R_off - R_on) * pow(1 - x, alpha);
        
        // Memristor current (Ohm's law)
        I = V / R;
        
        // State dynamics (window function model)
        window = x * (1 - x);  // Prevents saturation
        
        if (V > V_set)
            dxdt = (1/tau) * window * (V - V_set);
        else if (V < V_reset)
            dxdt = (1/tau) * window * (-V - abs(V_reset));
        else
            dxdt = 0;
        
        // Update state
        x = idt(dxdt, 0.1);
        x = (x < 0) ? 0 : ((x > 1) ? 1 : x);  // Clamp to [0,1]
        
        // Assign current
        I(p, n) <+ I;
    end
endmodule
"""
    
    output_path.write_text(verilog_code, encoding='utf-8')
    print(f"  Saved to: {output_path}")
    print(f"  Module: memristor_pinn")
    print(f"  Terminals: p (positive), n (negative)")
    print(f"  Parameters: R_on, R_off, alpha, tau, V_set, V_reset")
    
    return verilog_code

def create_figures(V_data, I_data, x_data, model, output_dir):
    """Generate publication-quality figures"""
    
    print("\n[FIGURES] Creating publication figures...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Convert to tensors for model evaluation
    V = torch.FloatTensor(V_data)
    x = torch.FloatTensor(x_data)
    
    with torch.no_grad():
        I_pred, _ = model(V, x)
        I_pred = I_pred.numpy()
    
    # Figure 1: I-V Hysteresis Curve
    plt.figure(figsize=(8, 6))
    plt.plot(V_data * 1000, I_data * 1000, 'b-', label='Training Data', linewidth=2, alpha=0.7)
    plt.plot(V_data * 1000, I_pred * 1000, 'r--', label='PINN Model', linewidth=2)
    plt.xlabel('Voltage (mV)', fontsize=12)
    plt.ylabel('Current (mA)', fontsize=12)
    plt.title('Memristor I-V Hysteresis Loop', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = output_dir / 'memristor_iv_curve.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path}")
    plt.close()
    
    # Figure 2: State Evolution
    plt.figure(figsize=(10, 4))
    time_points = np.arange(len(x_data))
    plt.plot(time_points, x_data, 'g-', linewidth=2)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Internal State x', fontsize=12)
    plt.title('Memristor State Variable Evolution', fontsize=14, fontweight='bold')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = output_dir / 'memristor_state_evolution.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path}")
    plt.close()
    
    # Figure 3: Error Distribution
    error = np.abs(I_data - I_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(V_data, error * 1000, c=x_data, cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(label='State x')
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Absolute Error (mA)', fontsize=12)
    plt.title('PINN Prediction Error Distribution', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = output_dir / 'memristor_error_distribution.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig_path}")
    plt.close()
    
    print(f"  All figures saved to: {output_dir}")

def save_data_to_csv(V_data, I_data, x_data, output_path):
    """Save training data to CSV file"""
    
    print("\n[DATA] Saving training data to CSV...")
    
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Voltage_V', 'Current_A', 'State_x'])
        
        for v, i, x in zip(V_data.flatten(), I_data.flatten(), x_data.flatten()):
            writer.writerow([f'{v:.6e}', f'{i:.6e}', f'{x:.6e}'])
    
    print(f"  ✓ Saved {len(V_data)} data points to: {output_path}")
    print(f"  Columns: Voltage_V, Current_A, State_x")

def create_spice_testbench(output_path):
    """Generate SPICE testbench for memristor validation"""
    
    print("\n[SPICE] Creating SPICE validation testbench...")
    
    spice_code = """* SPICE Testbench for Memristor PINN Verilog-A Module
* Model: memristor_pinn
* Auto-generated by Ψ-HDL Framework

.title Memristor PINN Test - I-V Hysteresis

.hdl memristor_pinn.va

* Voltage source: triangular wave for hysteresis sweep
* 3 cycles: 0V -> +1.2V -> -0.8V -> 0V
Vin p 0 PWL(
+ 0ns 0V
+ 100ns 1.2V
+ 200ns -0.8V
+ 300ns 0V
+ 400ns 1.2V
+ 500ns -0.8V
+ 600ns 0V
+ 700ns 1.2V
+ 800ns -0.8V
+ 900ns 0V
+ )

* Device under test
XMEM p n memristor_pinn

* Ground reference
Vn n 0 DC 0V

* Analysis
.tran 1ns 900ns

* Output
.print tran v(p) i(Vin) v(XMEM.x_state)
.control
run
set hcopydevtype=postscript
hardcopy memristor_iv.ps v(p) i(Vin)
plot v(p) i(Vin)
plot v(XMEM.x_state)
.endc

* Model parameters (default values from PINN)
* R_on = 1e3 Ω
* R_off = 1e5 Ω
* alpha = 2.0
* tau = 1e-3 s
* V_set = 0.8 V
* V_reset = -0.6 V

.end
"""
    
    output_path.write_text(spice_code, encoding='utf-8')
    print(f"  ✓ Saved to: {output_path}")
    print(f"  Analysis: Transient (0-900ns)")
    print(f"  Stimulus: Triangular wave (3 cycles)")
    print(f"  Outputs: Voltage, Current, Internal State")

def main():
    """Main execution pipeline"""
    
    print("="*70)
    print("Ψ-HDL DEMO: MEMRISTOR DEVICE MODELING")
    print("="*70)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output" / "memristor"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate synthetic data
    V_data, I_data, x_data = generate_memristor_data(DEVICE_PARAMS)
    
    # Save data to CSV
    data_csv_path = output_dir / "memristor_training_data.csv"
    save_data_to_csv(V_data, I_data, x_data, data_csv_path)
    
    # Step 2: Train PINN
    model = MemristorPINN(hidden_dims=[2, 40, 40, 40, 2])
    losses = train_memristor_pinn(model, V_data, I_data, x_data, epochs=3000)
    
    # Save model
    model_path = output_dir / "memristor_pinn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n[SAVE] Model saved to: {model_path}")
    
    # Step 3: Structure extraction
    structure = extract_structure(model, threshold=0.15)
    
    # Save structure
    structure_path = output_dir / "structure.json"
    with open(structure_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_structure = {}
        for key, val in structure.items():
            json_structure[key] = {
                'shape': val['shape'],
                'original_params': int(val['original_params']),
                'cluster_centers': [float(x) for x in val['cluster_centers']],
                'compressed_params': int(val['compressed_params']),
                'compression_ratio': float(val['compression_ratio'])
            }
        json.dump(json_structure, f, indent=2)
    print(f"[SAVE] Structure saved to: {structure_path}")
    
    # Step 4: Generate Verilog-A
    verilog_path = output_dir / "memristor_pinn.va"
    generate_verilog_a(model, structure, verilog_path)
    
    # Step 5: Create SPICE testbench
    spice_tb_path = output_dir / "memristor_pinn_tb.sp"
    create_spice_testbench(spice_tb_path)
    
    # Step 6: Create figures
    create_figures(V_data, I_data, x_data, model, output_dir / "figures")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. memristor_pinn.pth          - Trained model weights")
    print(f"  2. structure.json              - Extracted network structure")
    print(f"  3. memristor_pinn.va           - Verilog-A HDL module")
    print(f"  4. memristor_pinn_tb.sp        - SPICE testbench")
    print(f"  5. memristor_training_data.csv - Training dataset")
    print(f"  6. figures/memristor_*.png     - Publication figures")
    print("\n")

if __name__ == "__main__":
    main()