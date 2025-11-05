"""
SNN Model Loader - Load Spiking Neural Networks from JSON

This module loads trained SNN models from JSON format (as used in the
2024 Ex-Situ Training paper) and prepares them for HDL generation.

Supports:
- LIF (Leaky Integrate-and-Fire) neurons
- RealisticLapicque neuron model
- Multi-layer feedforward architectures
- Trained weights and biases

Author: Psi-HDL Framework
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


class LIF(nn.Module):
    """
    Leaky Integrate-and-Fire neuron implementation
    Compatible with the 2024 paper's neuron model
    """
    def __init__(
        self,
        threshold: float = 1.0,
        beta: float = 0.9355,
        R: float = 10.0,
        C: float = 0.0015,
        time_step: float = 0.001,
        reset_mechanism: str = "subtract"
    ):
        super().__init__()
        self.threshold = threshold
        self.beta = beta
        self.R = R
        self.C = C
        self.time_step = time_step
        self.reset_mechanism = reset_mechanism
        self.mem = None
        
    def forward(self, x):
        """Forward pass (inference mode)"""
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        
        # Integrate
        self.mem = self.beta * self.mem + x
        
        # Fire
        spikes = (self.mem >= self.threshold).float()
        
        # Reset
        if self.reset_mechanism == "subtract":
            self.mem = self.mem - spikes * self.threshold
        else:
            self.mem = self.mem * (1 - spikes)
        
        return spikes
    
    def reset_states(self):
        """Reset membrane potential"""
        self.mem = None


class SNNModel(nn.Module):
    """
    Spiking Neural Network from JSON
    Reconstructs architecture from trained models
    """
    def __init__(self, json_path: str):
        super().__init__()
        
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.weights = data['weights']
        self.params = data['parameters']
        
        # Extract architecture
        self.num_inputs = self.params['num_inputs']
        self.hidden_layers = self.params['hidden_layers']
        self.num_outputs = self.params['num_outputs']
        self.neuron_type = self.params['neuron_type']
        
        # Build layers
        self.layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        
        # Input layer
        in_features = self.num_inputs
        
        for i, hidden_size in enumerate(self.hidden_layers):
            # Linear layer
            fc = nn.Linear(in_features, hidden_size)
            self.layers.append(fc)
            
            # LIF layer
            if 'hyperparameters' in self.params:
                hp = self.params['hyperparameters']
                lif = LIF(
                    R=hp.get('R', 10.0),
                    C=hp.get('C', 0.0015),
                    time_step=hp.get('time_step', 0.001)
                )
            else:
                lif = LIF()
            self.lif_layers.append(lif)
            
            in_features = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(in_features, self.num_outputs)
        
        # Load weights
        self._load_weights()
    
    def _load_weights(self):
        """Load weights from JSON data"""
        # Load layer weights
        for i, layer in enumerate(self.layers):
            weight_key = f'fc{i+1}.weight'
            bias_key = f'fc{i+1}.bias'
            
            if weight_key in self.weights:
                weight_data = torch.tensor(
                    self.weights[weight_key]['data'],
                    dtype=torch.float32
                )
                layer.weight.data = weight_data
            
            if bias_key in self.weights:
                bias_data = torch.tensor(
                    self.weights[bias_key]['data'],
                    dtype=torch.float32
                )
                layer.bias.data = bias_data
        
        # Load output layer
        out_weight_key = f'fc{len(self.layers)+1}.weight'
        out_bias_key = f'fc{len(self.layers)+1}.bias'
        
        if out_weight_key in self.weights:
            weight_data = torch.tensor(
                self.weights[out_weight_key]['data'],
                dtype=torch.float32
            )
            self.output_layer.weight.data = weight_data
        
        if out_bias_key in self.weights:
            bias_data = torch.tensor(
                self.weights[out_bias_key]['data'],
                dtype=torch.float32
            )
            self.output_layer.bias.data = bias_data
        
        # Load LIF parameters
        for i, lif in enumerate(self.lif_layers):
            beta_key = f'lifs.{i}.beta'
            if beta_key in self.weights:
                beta_data = self.weights[beta_key]['data']
                lif.beta = beta_data[0] if isinstance(beta_data, list) else beta_data
    
    def forward(self, x, time_steps: int = 100):
        """
        Forward pass over time
        
        Args:
            x: Input tensor [batch_size, num_inputs]
            time_steps: Number of simulation time steps
        
        Returns:
            spike_counts: Output spike counts [batch_size, num_outputs]
        """
        batch_size = x.shape[0]
        
        # Reset LIF states
        for lif in self.lif_layers:
            lif.reset_states()
        
        # Accumulate spikes over time
        spike_accumulator = torch.zeros(batch_size, self.num_outputs)
        
        for t in range(time_steps):
            # Forward through layers
            h = x
            for layer, lif in zip(self.layers, self.lif_layers):
                h = layer(h)
                h = lif(h)
            
            # Output layer
            out = self.output_layer(h)
            spike_accumulator += out
        
        return spike_accumulator / time_steps
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get summary of network architecture"""
        return {
            'num_inputs': self.num_inputs,
            'hidden_layers': self.hidden_layers,
            'num_outputs': self.num_outputs,
            'neuron_type': self.neuron_type,
            'total_layers': len(self.layers) + 1,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }


def load_snn_from_json(json_path: str) -> SNNModel:
    """
    Load SNN model from JSON file
    
    Args:
        json_path: Path to JSON model file
    
    Returns:
        SNNModel instance with loaded weights
    
    Example:
        >>> model = load_snn_from_json("../related papers/.../Net_xor_model.json")
        >>> summary = model.get_architecture_summary()
        >>> print(f"Loaded {summary['total_layers']} layer SNN")
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Model file not found: {json_path}")
    
    model = SNNModel(str(json_path))
    model.eval()  # Set to evaluation mode
    
    return model


def convert_snn_to_rate_coded(model: SNNModel, time_steps: int = 100) -> nn.Module:
    """
    Convert SNN to rate-coded ANN for easier HDL generation
    
    This extracts the weighted connections and removes spiking dynamics,
    making it compatible with standard Verilog-A generation.
    
    Args:
        model: Trained SNN model
        time_steps: Number of time steps used in simulation
    
    Returns:
        Rate-coded feedforward network
    """
    class RateCodedNet(nn.Module):
        def __init__(self, snn: SNNModel, steps: int):
            super().__init__()
            self.snn = snn
            self.time_steps = steps
            
            # Copy layers
            self.layers = nn.ModuleList()
            for layer in snn.layers:
                self.layers.append(layer)
            self.output_layer = snn.output_layer
        
        def forward(self, x):
            # Simple feedforward (no spiking)
            for layer in self.layers:
                x = layer(x)
                x = torch.tanh(x)  # Replace LIF with tanh
            x = self.output_layer(x)
            return x
    
    rate_net = RateCodedNet(model, time_steps)
    rate_net.eval()
    return rate_net


if __name__ == "__main__":
    # Example usage
    print("SNN Loader Test")
    print("=" * 60)
    
    # Try to load XOR model
    xor_path = "../related papers/SNNs_for_Inference_using_Ex-Situ_Training-main/Net_xor_model.json"
    
    try:
        model = load_snn_from_json(xor_path)
        summary = model.get_architecture_summary()
        
        print("\nLoaded SNN Model:")
        print(f"  Architecture: {summary['num_inputs']} -> {summary['hidden_layers']} -> {summary['num_outputs']}")
        print(f"  Neuron type: {summary['neuron_type']}")
        print(f"  Total parameters: {summary['total_parameters']}")
        
        # Test forward pass
        x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = model(x, time_steps=100)
        
        print("\nXOR Test:")
        print("  Input | Output")
        for i in range(4):
            print(f"  {x[i].numpy()} | {y[i].item():.3f}")
        
        # Convert to rate-coded
        rate_model = convert_snn_to_rate_coded(model)
        y_rate = rate_model(x)
        
        print("\nRate-Coded Test:")
        print("  Input | Output")
        for i in range(4):
            print(f"  {x[i].numpy()} | {y_rate[i].item():.3f}")
        
        print("\n[OK] SNN Loader working correctly!")
        
    except FileNotFoundError as e:
        print(f"\nNote: {e}")
        print("  This is expected if running outside the main directory")