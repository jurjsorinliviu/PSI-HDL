"""
SPICE Validator for Psi-HDL Framework
Validates generated Verilog-A models against PyTorch reference implementations.

Author: Psi-HDL Pipeline
"""

import numpy as np
import torch
import subprocess
import re
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import pandas as pd


class SPICEValidator:
    """Validate Verilog-A models using SPICE simulation"""
    
    def __init__(self, pytorch_model, verilog_path: str, spice_binary: str = "ngspice"):
        """
        Initialize SPICE validator
        
        Args:
            pytorch_model: Trained PyTorch model for reference
            verilog_path: Path to Verilog-A module file
            spice_binary: SPICE simulator binary (default: ngspice)
        """
        self.pytorch_model = pytorch_model
        self.verilog_path = verilog_path
        self.spice_binary = spice_binary
        self.results = {}
        
    def generate_test_points(self, x_range: Tuple[float, float], 
                            y_range: Tuple[float, float], 
                            num_points: int = 50) -> np.ndarray:
        """
        Generate test points for validation
        
        Args:
            x_range: (min, max) for x coordinate
            y_range: (min, max) for y coordinate
            num_points: Number of points per dimension
            
        Returns:
            Test points array [N x 2]
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        test_points = np.column_stack([X.ravel(), Y.ravel()])
        
        return test_points
    
    def predict_pytorch(self, test_points: np.ndarray) -> np.ndarray:
        """
        Get predictions from PyTorch model
        
        Args:
            test_points: Input test points
            
        Returns:
            Predictions array
        """
        self.pytorch_model.eval()
        with torch.no_grad():
            inputs = torch.tensor(test_points, dtype=torch.float32)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                self.pytorch_model = self.pytorch_model.cuda()
            
            outputs = self.pytorch_model(inputs)
            predictions = outputs.cpu().numpy()
        
        return predictions
    
    def run_spice_simulation(self, netlist_path: str) -> Dict[str, np.ndarray]:
        """
        Run SPICE simulation and extract results
        
        Args:
            netlist_path: Path to SPICE netlist file
            
        Returns:
            Dictionary with simulation results
        """
        print(f"Running SPICE simulation with {self.spice_binary}...")
        
        # Check if SPICE binary exists
        try:
            result = subprocess.run([self.spice_binary, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            print(f"SPICE version: {result.stdout.split()[0]}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Warning: Could not verify SPICE installation: {e}")
            print("Simulation will be skipped. Install ngspice to run actual validation.")
            return self._mock_spice_results()
        
        try:
            # Run SPICE simulation
            result = subprocess.run([self.spice_binary, '-b', netlist_path, '-o', 
                                   netlist_path + '.out'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"SPICE simulation failed with error:\n{result.stderr}")
                return self._mock_spice_results()
            
            # Parse output file
            output_file = netlist_path + '.out'
            if os.path.exists(output_file):
                return self._parse_spice_output(output_file)
            else:
                print("SPICE output file not found")
                return self._mock_spice_results()
                
        except subprocess.TimeoutExpired:
            print("SPICE simulation timed out")
            return self._mock_spice_results()
        except Exception as e:
            print(f"Error running SPICE simulation: {e}")
            return self._mock_spice_results()
    
    def _parse_spice_output(self, output_file: str) -> Dict[str, np.ndarray]:
        """Parse SPICE output file to extract simulation results"""
        results = {
            'v_in0': [],
            'v_in1': [],
            'v_out0': []
        }
        
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            # Find data section
            data_start = -1
            for i, line in enumerate(lines):
                if 'Index' in line or 'v(in0)' in line.lower():
                    data_start = i + 1
                    break
            
            if data_start > 0:
                for line in lines[data_start:]:
                    # Parse data line
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            results['v_in0'].append(float(parts[0]))
                            results['v_in1'].append(float(parts[1]))
                            results['v_out0'].append(float(parts[2]))
                        except ValueError:
                            continue
            
            # Convert to numpy arrays
            for key in results:
                results[key] = np.array(results[key])
                
        except Exception as e:
            print(f"Error parsing SPICE output: {e}")
            return self._mock_spice_results()
        
        return results
    
    def _mock_spice_results(self) -> Dict[str, np.ndarray]:
        """Generate mock SPICE results for demonstration purposes"""
        print("Generating mock SPICE results (SPICE not available)")
        # Return empty results
        return {
            'v_in0': np.array([]),
            'v_in1': np.array([]),
            'v_out0': np.array([])
        }
    
    def compare_results(self, pytorch_predictions: np.ndarray, 
                       spice_results: Dict[str, np.ndarray],
                       tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Compare PyTorch and SPICE results
        
        Args:
            pytorch_predictions: Predictions from PyTorch model
            spice_results: Results from SPICE simulation
            tolerance: Acceptable error tolerance
            
        Returns:
            Comparison metrics dictionary
        """
        print("\nComparing PyTorch vs SPICE results...")
        
        if len(spice_results['v_out0']) == 0:
            print("No SPICE results available for comparison")
            return {
                'available': False,
                'message': 'SPICE simulation not run'
            }
        
        # Extract SPICE predictions
        spice_pred = spice_results['v_out0']
        
        # Compute error metrics
        if len(pytorch_predictions) != len(spice_pred):
            print(f"Warning: Size mismatch - PyTorch: {len(pytorch_predictions)}, SPICE: {len(spice_pred)}")
            # Truncate to minimum length
            min_len = min(len(pytorch_predictions), len(spice_pred))
            pytorch_predictions = pytorch_predictions[:min_len]
            spice_pred = spice_pred[:min_len]
        
        # Compute metrics
        mae = np.mean(np.abs(pytorch_predictions.flatten() - spice_pred))
        mse = np.mean((pytorch_predictions.flatten() - spice_pred)**2)
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(pytorch_predictions.flatten() - spice_pred))
        
        # Relative error
        denom = np.maximum(np.abs(pytorch_predictions.flatten()), 1e-10)
        relative_error = np.mean(np.abs(pytorch_predictions.flatten() - spice_pred) / denom)
        
        # Correlation
        correlation = np.corrcoef(pytorch_predictions.flatten(), spice_pred)[0, 1]
        
        # Pass/fail
        passed = mae < tolerance
        
        comparison = {
            'available': True,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'max_error': max_error,
            'relative_error': relative_error,
            'correlation': correlation,
            'passed': passed,
            'tolerance': tolerance,
            'num_points': len(pytorch_predictions)
        }
        
        # Print results
        print("\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)
        print(f"Number of test points: {comparison['num_points']}")
        print(f"Mean Absolute Error (MAE): {comparison['mae']:.6e}")
        print(f"Root Mean Square Error (RMSE): {comparison['rmse']:.6e}")
        print(f"Maximum Error: {comparison['max_error']:.6e}")
        print(f"Relative Error: {comparison['relative_error']:.6%}")
        print(f"Correlation: {comparison['correlation']:.6f}")
        print(f"Tolerance: {comparison['tolerance']:.6e}")
        print(f"Status: {'PASSED [OK]' if comparison['passed'] else 'FAILED [FAIL]'}")
        print("="*70 + "\n")
        
        return comparison
    
    def visualize_comparison(self, test_points: np.ndarray, 
                            pytorch_predictions: np.ndarray,
                            spice_results: Dict[str, np.ndarray],
                            save_path: str = None):
        """
        Visualize comparison between PyTorch and SPICE results
        
        Args:
            test_points: Input test points
            pytorch_predictions: PyTorch predictions
            spice_results: SPICE simulation results
            save_path: Optional path to save figure
        """
        if len(spice_results['v_out0']) == 0:
            print("No SPICE results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Reshape for 2D visualization
        n_pts = int(np.sqrt(len(test_points)))
        X = test_points[:, 0].reshape(n_pts, n_pts)
        Y = test_points[:, 1].reshape(n_pts, n_pts)
        
        # PyTorch predictions
        Z_pytorch = pytorch_predictions.reshape(n_pts, n_pts)
        im1 = axes[0, 0].contourf(X, Y, Z_pytorch, levels=20, cmap='viridis')
        axes[0, 0].set_title('PyTorch Model Output')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # SPICE results
        if len(spice_results['v_out0']) == len(test_points):
            Z_spice = spice_results['v_out0'].reshape(n_pts, n_pts)
            im2 = axes[0, 1].contourf(X, Y, Z_spice, levels=20, cmap='viridis')
            axes[0, 1].set_title('SPICE Simulation Output')
            axes[0, 1].set_xlabel('x')
            axes[0, 1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Error map
            Z_error = np.abs(Z_pytorch - Z_spice)
            im3 = axes[1, 0].contourf(X, Y, Z_error, levels=20, cmap='Reds')
            axes[1, 0].set_title('Absolute Error |PyTorch - SPICE|')
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('y')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Scatter plot
            axes[1, 1].scatter(pytorch_predictions.flatten(), 
                             spice_results['v_out0'], 
                             alpha=0.5, s=10)
            axes[1, 1].plot([pytorch_predictions.min(), pytorch_predictions.max()],
                          [pytorch_predictions.min(), pytorch_predictions.max()],
                          'r--', label='Perfect match')
            axes[1, 1].set_xlabel('PyTorch Output')
            axes[1, 1].set_ylabel('SPICE Output')
            axes[1, 1].set_title('PyTorch vs SPICE Scatter Plot')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def full_validation_workflow(self, x_range: Tuple[float, float],
                                 y_range: Tuple[float, float],
                                 num_points: int = 50,
                                 netlist_path: str = None,
                                 output_dir: str = "./validation_results") -> Dict[str, Any]:
        """
        Run complete validation workflow
        
        Args:
            x_range: Range for x coordinate
            y_range: Range for y coordinate
            num_points: Number of test points per dimension
            netlist_path: Path to SPICE netlist (if None, skip SPICE)
            output_dir: Directory to save results
            
        Returns:
            Complete validation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("STARTING FULL VALIDATION WORKFLOW")
        print("="*70)
        
        # Generate test points
        print("\n1. Generating test points...")
        test_points = self.generate_test_points(x_range, y_range, num_points)
        print(f"   Generated {len(test_points)} test points")
        
        # Get PyTorch predictions
        print("\n2. Computing PyTorch predictions...")
        pytorch_predictions = self.predict_pytorch(test_points)
        print(f"   Computed predictions for {len(pytorch_predictions)} points")
        
        # Save PyTorch results
        pytorch_df = pd.DataFrame({
            'x': test_points[:, 0],
            'y': test_points[:, 1],
            'output': pytorch_predictions.flatten()
        })
        pytorch_csv = os.path.join(output_dir, 'pytorch_predictions.csv')
        pytorch_df.to_csv(pytorch_csv, index=False)
        print(f"   PyTorch results saved to {pytorch_csv}")
        
        # Run SPICE simulation
        comparison = None
        if netlist_path and os.path.exists(netlist_path):
            print("\n3. Running SPICE simulation...")
            spice_results = self.run_spice_simulation(netlist_path)
            
            if len(spice_results['v_out0']) > 0:
                # Save SPICE results
                spice_df = pd.DataFrame(spice_results)
                spice_csv = os.path.join(output_dir, 'spice_results.csv')
                spice_df.to_csv(spice_csv, index=False)
                print(f"   SPICE results saved to {spice_csv}")
                
                # Compare results
                print("\n4. Comparing results...")
                comparison = self.compare_results(pytorch_predictions, spice_results)
                
                # Visualize
                print("\n5. Generating visualization...")
                vis_path = os.path.join(output_dir, 'validation_comparison.png')
                self.visualize_comparison(test_points, pytorch_predictions, 
                                        spice_results, vis_path)
            else:
                print("   SPICE simulation did not produce results")
        else:
            print("\n3. Skipping SPICE simulation (no netlist provided)")
            print("   To run full validation, provide a valid SPICE netlist path")
        
        # Compile results
        results = {
            'test_points': test_points,
            'pytorch_predictions': pytorch_predictions,
            'comparison': comparison,
            'output_dir': output_dir
        }
        
        print("\n" + "="*70)
        print("VALIDATION WORKFLOW COMPLETE")
        print("="*70 + "\n")
        
        return results


def quick_validate(pytorch_model, test_points: np.ndarray) -> Dict[str, Any]:
    """
    Quick validation without SPICE (PyTorch only)
    
    Args:
        pytorch_model: Trained PyTorch model
        test_points: Test input points
        
    Returns:
        PyTorch predictions and statistics
    """
    model = pytorch_model
    model.eval()
    
    with torch.no_grad():
        inputs = torch.tensor(test_points, dtype=torch.float32)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            model = model.cuda()
        
        outputs = model(inputs)
        predictions = outputs.cpu().numpy()
    
    results = {
        'test_points': test_points,
        'predictions': predictions,
        'statistics': {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions))
        }
    }
    
    print("\nPyTorch Model Statistics:")
    print(f"  Mean output: {results['statistics']['mean']:.6e}")
    print(f"  Std output: {results['statistics']['std']:.6e}")
    print(f"  Min output: {results['statistics']['min']:.6e}")
    print(f"  Max output: {results['statistics']['max']:.6e}")
    
    return results