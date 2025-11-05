"""
Structure Extractor for Psi-NN Models
Extracts discovered physical structures from trained Psi-NN and PINN models
for translation to Verilog-A HDL.

Author: Psi-HDL Pipeline
"""

import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Any
import json


class NetworkStructure:
    """Container for extracted network structure information"""
    
    def __init__(self):
        self.layers: List[Dict[str, Any]] = []
        self.connections: List[Dict[str, Any]] = []
        self.input_dim: int = 0
        self.output_dim: int = 0
        self.activation: str = "tanh"
        self.special_structures: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict:
        """Convert structure to dictionary for JSON serialization"""
        return {
            "layers": self.layers,
            "connections": self.connections,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "special_structures": self.special_structures,
            "metadata": self.metadata
        }
    
    def save_json(self, filepath: str):
        """Save structure to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Structure saved to {filepath}")


class StructureExtractor:
    """Extract network structure from trained PyTorch models"""
    
    def __init__(self, model: torch.nn.Module, model_type: str = "auto"):
        """
        Initialize structure extractor
        
        Args:
            model: Trained PyTorch model (PINN or PsiNN)
            model_type: Type of model ("PINN", "PsiNN_burgers", "PsiNN_laplace", or "auto")
        """
        self.model = model
        self.model_type = model_type if model_type != "auto" else self._detect_model_type()
        self.structure = NetworkStructure()
        
    def _detect_model_type(self) -> str:
        """Automatically detect model type from module name"""
        module_name = self.model.__module__.split('.')[-1]
        return module_name
    
    def extract(self) -> NetworkStructure:
        """
        Extract complete network structure
        
        Returns:
            NetworkStructure object containing all extracted information
        """
        print(f"Extracting structure from {self.model_type} model...")
        
        # Extract based on model type
        if "PINN" in self.model_type and "PsiNN" not in self.model_type:
            self._extract_pinn_structure()
        elif "PsiNN_burgers" in self.model_type:
            self._extract_psinn_burgers_structure()
        elif "PsiNN_laplace" in self.model_type:
            self._extract_psinn_laplace_structure()
        else:
            # Fall back to generic extraction for unknown types (SNN, custom models, etc.)
            print(f"Using generic extraction for model type: {self.model_type}")
            self._extract_generic_structure()
        
        # Add metadata
        self._extract_metadata()
        
        print("Structure extraction complete.")
        return self.structure
    
    def _extract_pinn_structure(self):
        """Extract structure from standard PINN model"""
        print("Extracting PINN structure...")
        
        # PINN uses Sequential layers
        if hasattr(self.model, 'layers'):
            layers_dict = dict(self.model.layers.named_children())
            
            layer_idx = 0
            for name, layer in layers_dict.items():
                if isinstance(layer, torch.nn.Linear):
                    weight = layer.weight.detach().cpu().numpy()
                    bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None
                    
                    layer_info = {
                        "index": layer_idx,
                        "type": "linear",
                        "name": name,
                        "input_dim": weight.shape[1],
                        "output_dim": weight.shape[0],
                        "weight_shape": weight.shape,
                        "has_bias": bias is not None,
                        "weight_stats": {
                            "mean": float(np.mean(weight)),
                            "std": float(np.std(weight)),
                            "min": float(np.min(weight)),
                            "max": float(np.max(weight))
                        }
                    }
                    
                    if layer_idx == 0:
                        self.structure.input_dim = weight.shape[1]
                    
                    self.structure.layers.append(layer_info)
                    layer_idx += 1
                    
                elif isinstance(layer, torch.nn.Tanh):
                    # Record activation layer
                    activation_info = {
                        "index": layer_idx,
                        "type": "activation",
                        "name": name,
                        "function": "tanh"
                    }
                    self.structure.layers.append(activation_info)
                    layer_idx += 1
            
            # Set output dimension from last linear layer
            if self.structure.layers:
                for layer in reversed(self.structure.layers):
                    if layer["type"] == "linear":
                        self.structure.output_dim = layer["output_dim"]
                        break
        
        # Extract sequential connections
        self._extract_sequential_connections()
    
    def _extract_generic_structure(self):
        """Extract structure from generic PyTorch model (e.g., rate-coded SNN, custom networks)"""
        print("Extracting generic model structure...")
        
        layer_idx = 0
        
        # Try to find layers in ModuleList
        if hasattr(self.model, 'layers') and isinstance(self.model.layers, torch.nn.ModuleList):
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, torch.nn.Linear):
                    weight = layer.weight.detach().cpu().numpy()
                    bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None
                    
                    layer_info = {
                        "index": layer_idx,
                        "type": "linear",
                        "name": f"layer_{i}",
                        "input_dim": weight.shape[1],
                        "output_dim": weight.shape[0],
                        "weight_shape": weight.shape,
                        "has_bias": bias is not None,
                        "weight_stats": self._compute_stats(weight)
                    }
                    
                    if layer_idx == 0:
                        self.structure.input_dim = weight.shape[1]
                    
                    self.structure.layers.append(layer_info)
                    layer_idx += 1
        
        # Try to find output layer
        if hasattr(self.model, 'output_layer'):
            layer = self.model.output_layer
            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight.detach().cpu().numpy()
                bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None
                
                layer_info = {
                    "index": layer_idx,
                    "type": "linear",
                    "name": "output",
                    "input_dim": weight.shape[1],
                    "output_dim": weight.shape[0],
                    "weight_shape": weight.shape,
                    "has_bias": bias is not None,
                    "weight_stats": self._compute_stats(weight)
                }
                
                self.structure.layers.append(layer_info)
                self.structure.output_dim = weight.shape[0]
                layer_idx += 1
        
        # If no layers found, try to iterate through all named modules
        if len(self.structure.layers) == 0:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    weight = module.weight.detach().cpu().numpy()
                    bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
                    
                    layer_info = {
                        "index": layer_idx,
                        "type": "linear",
                        "name": name,
                        "input_dim": weight.shape[1],
                        "output_dim": weight.shape[0],
                        "weight_shape": weight.shape,
                        "has_bias": bias is not None,
                        "weight_stats": self._compute_stats(weight)
                    }
                    
                    if layer_idx == 0:
                        self.structure.input_dim = weight.shape[1]
                    
                    self.structure.layers.append(layer_info)
                    layer_idx += 1
        
        # Set output dimension from last layer
        if self.structure.layers:
            self.structure.output_dim = self.structure.layers[-1]["output_dim"]
        
        # Extract sequential connections
        self._extract_sequential_connections()
        
        print(f"Extracted {len(self.structure.layers)} layers from generic model")
    
    def _extract_psinn_burgers_structure(self):
        """Extract structure from PsiNN Burgers equation model"""
        print("Extracting PsiNN Burgers structure...")
        
        # PsiNN Burgers has special +/-W structure
        layer_idx = 0
        
        # Layer 1: fc1_1 and fc1_3 (x +/- y pattern)
        if hasattr(self.model, 'fc1_1') and hasattr(self.model, 'fc1_3'):
            fc1_1_weight = self.model.fc1_1.weight.detach().cpu().numpy()
            fc1_1_bias = self.model.fc1_1.bias.detach().cpu().numpy()
            fc1_3_weight = self.model.fc1_3.weight.detach().cpu().numpy()
            
            self.structure.input_dim = 2  # (x, y)
            
            # Record the +/-W structure
            layer1_info = {
                "index": layer_idx,
                "type": "psi_plus_minus",
                "name": "layer1",
                "fc1_1": {
                    "weight_shape": fc1_1_weight.shape,
                    "has_bias": True,
                    "weight_stats": self._compute_stats(fc1_1_weight)
                },
                "fc1_3": {
                    "weight_shape": fc1_3_weight.shape,
                    "has_bias": False,
                    "weight_stats": self._compute_stats(fc1_3_weight)
                },
                "structure": "tanh(fc1_1(x) +/- fc1_3(y))",
                "output_dim": fc1_1_weight.shape[0] * 2  # Two branches
            }
            self.structure.layers.append(layer1_info)
            layer_idx += 1
            
            # Record special structure
            self.structure.special_structures["plus_minus_input"] = {
                "description": "Input layer uses +/-W structure for symmetry",
                "equation": "u1_1 = tanh(fc1_1(x) + fc1_3(y)), u1_2 = tanh(fc1_1(x) - fc1_3(y))"
            }
        
        # Layer 2: fc2_1, fc2_2, fc2_3 (complex symmetric structure)
        if hasattr(self.model, 'fc2_1') and hasattr(self.model, 'fc2_2') and hasattr(self.model, 'fc2_3'):
            fc2_1_weight = self.model.fc2_1.weight.detach().cpu().numpy()
            fc2_2_weight = self.model.fc2_2.weight.detach().cpu().numpy()
            fc2_3_weight = self.model.fc2_3.weight.detach().cpu().numpy()
            
            layer2_info = {
                "index": layer_idx,
                "type": "psi_symmetric",
                "name": "layer2",
                "fc2_1": {
                    "weight_shape": fc2_1_weight.shape,
                    "weight_stats": self._compute_stats(fc2_1_weight)
                },
                "fc2_2": {
                    "weight_shape": fc2_2_weight.shape,
                    "weight_stats": self._compute_stats(fc2_2_weight)
                },
                "fc2_3": {
                    "weight_shape": fc2_3_weight.shape,
                    "weight_stats": self._compute_stats(fc2_3_weight)
                },
                "structure": "tanh(fc2_1(u1_1) + fc2_3(u1_2)), tanh(fc2_3(u1_1) + fc2_1(u1_2)), tanh(fc2_2(u1_1) - fc2_2(u1_2))",
                "output_dim": fc2_1_weight.shape[0] * 3
            }
            self.structure.layers.append(layer2_info)
            layer_idx += 1
            
            self.structure.special_structures["symmetric_interchange"] = {
                "description": "Layer 2 uses weight sharing with symmetric interchange",
                "equation": "u2_1 = tanh(fc2_1(u1_1) + fc2_3(u1_2)), u2_2 = tanh(fc2_3(u1_1) + fc2_1(u1_2))"
            }
        
        # Layer 3: fc3_1, fc3_2
        if hasattr(self.model, 'fc3_1') and hasattr(self.model, 'fc3_2'):
            fc3_1_weight = self.model.fc3_1.weight.detach().cpu().numpy()
            fc3_2_weight = self.model.fc3_2.weight.detach().cpu().numpy()
            
            layer3_info = {
                "index": layer_idx,
                "type": "psi_combination",
                "name": "layer3",
                "fc3_1": {
                    "weight_shape": fc3_1_weight.shape,
                    "weight_stats": self._compute_stats(fc3_1_weight)
                },
                "fc3_2": {
                    "weight_shape": fc3_2_weight.shape,
                    "weight_stats": self._compute_stats(fc3_2_weight)
                },
                "structure": "tanh(fc3_1(u2_1) - fc3_1(u2_2) + fc3_2(u2_3))",
                "output_dim": fc3_1_weight.shape[0]
            }
            self.structure.layers.append(layer3_info)
            layer_idx += 1
        
        # Layer 4: fc4_1 (output layer)
        if hasattr(self.model, 'fc4_1'):
            fc4_1_weight = self.model.fc4_1.weight.detach().cpu().numpy()
            
            layer4_info = {
                "index": layer_idx,
                "type": "linear",
                "name": "output",
                "weight_shape": fc4_1_weight.shape,
                "has_bias": False,  # Odd function, no bias
                "weight_stats": self._compute_stats(fc4_1_weight),
                "output_dim": fc4_1_weight.shape[0]
            }
            self.structure.layers.append(layer4_info)
            self.structure.output_dim = fc4_1_weight.shape[0]
            layer_idx += 1
            
            self.structure.special_structures["odd_function"] = {
                "description": "Output layer has no bias to preserve odd function property",
                "equation": "u = fc4_1(u3_1)"
            }
    
    def _extract_psinn_laplace_structure(self):
        """Extract structure from PsiNN Laplace equation model"""
        print("Extracting PsiNN Laplace structure...")
        
        layer_idx = 0
        self.structure.input_dim = 2  # (x, y)
        
        # Layer 1: fc1_2, fc1_4
        if hasattr(self.model, 'fc1_2') and hasattr(self.model, 'fc1_4'):
            fc1_2_weight = self.model.fc1_2.weight.detach().cpu().numpy()
            fc1_4_weight = self.model.fc1_4.weight.detach().cpu().numpy()
            
            layer1_info = {
                "index": layer_idx,
                "type": "psi_plus_minus",
                "name": "layer1",
                "fc1_2": {
                    "weight_shape": fc1_2_weight.shape,
                    "weight_stats": self._compute_stats(fc1_2_weight)
                },
                "fc1_4": {
                    "weight_shape": fc1_4_weight.shape,
                    "weight_stats": self._compute_stats(fc1_4_weight)
                },
                "structure": "tanh(fc1_2(x) +/- fc1_4(y))",
                "output_dim": fc1_2_weight.shape[0] * 2
            }
            self.structure.layers.append(layer1_info)
            layer_idx += 1
        
        # Layer 2: fc2_2, fc2_4
        if hasattr(self.model, 'fc2_2') and hasattr(self.model, 'fc2_4'):
            fc2_2_weight = self.model.fc2_2.weight.detach().cpu().numpy()
            fc2_4_weight = self.model.fc2_4.weight.detach().cpu().numpy()
            
            layer2_info = {
                "index": layer_idx,
                "type": "psi_symmetric",
                "name": "layer2",
                "fc2_2": {
                    "weight_shape": fc2_2_weight.shape,
                    "weight_stats": self._compute_stats(fc2_2_weight)
                },
                "fc2_4": {
                    "weight_shape": fc2_4_weight.shape,
                    "weight_stats": self._compute_stats(fc2_4_weight)
                },
                "structure": "tanh(fc2_2(u1_3) + fc2_4(u1_4)), tanh(fc2_4(u1_3) + fc2_2(u1_4))",
                "output_dim": fc2_2_weight.shape[0] * 2
            }
            self.structure.layers.append(layer2_info)
            layer_idx += 1
        
        # Layer 3: fc3_2
        if hasattr(self.model, 'fc3_2'):
            fc3_2_weight = self.model.fc3_2.weight.detach().cpu().numpy()
            
            layer3_info = {
                "index": layer_idx,
                "type": "psi_combination",
                "name": "layer3",
                "fc3_2": {
                    "weight_shape": fc3_2_weight.shape,
                    "weight_stats": self._compute_stats(fc3_2_weight)
                },
                "structure": "tanh(fc3_2(u2_3) + fc3_2(u2_4))",
                "output_dim": fc3_2_weight.shape[0]
            }
            self.structure.layers.append(layer3_info)
            layer_idx += 1
        
        # Layer 4: fc4_1 (output)
        if hasattr(self.model, 'fc4_1'):
            fc4_1_weight = self.model.fc4_1.weight.detach().cpu().numpy()
            fc4_1_bias = self.model.fc4_1.bias.detach().cpu().numpy()
            
            layer4_info = {
                "index": layer_idx,
                "type": "linear",
                "name": "output",
                "weight_shape": fc4_1_weight.shape,
                "has_bias": True,  # Even function, has bias
                "weight_stats": self._compute_stats(fc4_1_weight),
                "output_dim": fc4_1_weight.shape[0]
            }
            self.structure.layers.append(layer4_info)
            self.structure.output_dim = fc4_1_weight.shape[0]
            layer_idx += 1
            
            self.structure.special_structures["even_function"] = {
                "description": "Output layer has bias to allow even function property",
                "equation": "u = fc4_1(u3_2)"
            }
    
    def _extract_sequential_connections(self):
        """Extract sequential layer connections for PINN"""
        for i in range(len(self.structure.layers) - 1):
            connection = {
                "from_layer": i,
                "to_layer": i + 1,
                "type": "sequential"
            }
            self.structure.connections.append(connection)
    
    def _compute_stats(self, weight: np.ndarray) -> Dict[str, float]:
        """Compute statistics for weight matrix"""
        return {
            "mean": float(np.mean(weight)),
            "std": float(np.std(weight)),
            "min": float(np.min(weight)),
            "max": float(np.max(weight)),
            "l2_norm": float(np.linalg.norm(weight))
        }
    
    def _extract_metadata(self):
        """Extract model metadata"""
        self.structure.metadata = {
            "model_type": self.model_type,
            "total_layers": len(self.structure.layers),
            "input_dim": self.structure.input_dim,
            "output_dim": self.structure.output_dim,
            "activation": self.structure.activation,
            "has_special_structures": len(self.structure.special_structures) > 0,
            "special_structure_types": list(self.structure.special_structures.keys())
        }
    
    def export_weights(self, filepath: str):
        """
        Export all weight matrices to NumPy format
        
        Args:
            filepath: Path to save weights (.npz file)
        """
        weights_dict = {}
        
        for name, param in self.model.named_parameters():
            weights_dict[name] = param.detach().cpu().numpy()
        
        np.savez(filepath, **weights_dict)
        print(f"Weights exported to {filepath}")
    
    def print_summary(self):
        """Print a summary of the extracted structure"""
        print("\n" + "="*70)
        print("EXTRACTED NETWORK STRUCTURE SUMMARY")
        print("="*70)
        print(f"Model Type: {self.structure.metadata['model_type']}")
        print(f"Input Dimension: {self.structure.input_dim}")
        print(f"Output Dimension: {self.structure.output_dim}")
        print(f"Total Layers: {self.structure.metadata['total_layers']}")
        print(f"Activation Function: {self.structure.activation}")
        print("\nLayer Details:")
        print("-" * 70)
        
        for layer in self.structure.layers:
            print(f"  [{layer['index']}] {layer['name']}: {layer['type']}")
            if 'structure' in layer:
                print(f"      Structure: {layer['structure']}")
            if 'output_dim' in layer:
                print(f"      Output Dim: {layer['output_dim']}")
        
        if self.structure.special_structures:
            print("\nSpecial Structures Discovered:")
            print("-" * 70)
            for key, value in self.structure.special_structures.items():
                print(f"  - {key}:")
                print(f"    {value['description']}")
                if 'equation' in value:
                    print(f"    Equation: {value['equation']}")
        
        print("="*70 + "\n")


# Convenience functions
def extract_from_checkpoint(checkpoint_path: str, model_class, model_type: str = "auto") -> NetworkStructure:
    """
    Load a model from checkpoint and extract its structure
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        model_class: Model class to instantiate
        model_type: Type of model
    
    Returns:
        NetworkStructure object
    """
    # Create model instance (you'll need to provide proper initialization)
    # This is a placeholder - actual implementation depends on model requirements
    raise NotImplementedError("Please implement model loading logic for the specific use case")


def compare_structures(struct1: NetworkStructure, struct2: NetworkStructure):
    """Compare two network structures"""
    print("\nStructure Comparison:")
    print("-" * 70)
    print(f"Model 1: {struct1.metadata['model_type']}")
    print(f"Model 2: {struct2.metadata['model_type']}")
    print(f"Layer Count: {struct1.metadata['total_layers']} vs {struct2.metadata['total_layers']}")
    print(f"Input Dim: {struct1.input_dim} vs {struct2.input_dim}")
    print(f"Output Dim: {struct1.output_dim} vs {struct2.output_dim}")
    
    # Compare special structures
    keys1 = set(struct1.special_structures.keys())
    keys2 = set(struct2.special_structures.keys())
    
    common = keys1.intersection(keys2)
    only1 = keys1 - keys2
    only2 = keys2 - keys1
    
    if common:
        print(f"\nCommon special structures: {common}")
    if only1:
        print(f"Only in Model 1: {only1}")
    if only2:
        print(f"Only in Model 2: {only2}")
    print("-" * 70)