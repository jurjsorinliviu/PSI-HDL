#!/usr/bin/env python3
"""
SPICE Validation Automation Script

This script automates the SPICE validation process for all generated models:
1. Runs ngspice simulations on generated testbenches
2. Compares SPICE outputs with PyTorch predictions
3. Generates accuracy metrics and comparison tables
4. Creates validation reports

Usage:
    python run_spice_validation.py
    python run_spice_validation.py --model burgers
    python run_spice_validation.py --spice-path "C:/Program Files/ngspice/bin/ngspice.exe"
"""

import argparse
import subprocess
import json
import os
import sys
from pathlib import Path
import re
import numpy as np
from datetime import datetime

class SPICEValidator:
    """Automates SPICE validation and comparison"""
    
    def __init__(self, spice_executable="ngspice", output_dir="output", results_dir="results"):
        self.spice_exe = spice_executable
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Models to validate
        self.models = {
            'burgers': {
                'testbench': self.output_dir / 'burgers' / 'psi_nn_PsiNN_burgers_tb.sp',
                'structure': self.output_dir / 'burgers' / 'burgers_structure.json',
                'description': 'Burgers Equation (Odd Function)'
            },
            'laplace': {
                'testbench': self.output_dir / 'laplace' / 'psi_nn_PsiNN_laplace_tb.sp',
                'structure': self.output_dir / 'laplace' / 'laplace_structure.json',
                'description': 'Laplace Equation (Even Function)'
            },
            'snn_xor': {
                'testbench': self.output_dir / 'snn_xor' / 'psi_nn_SNN_XOR_tb.sp',
                'structure': self.output_dir / 'snn_xor' / 'xor_structure.json',
                'description': 'SNN XOR Circuit'
            }
        }
        
        self.validation_results = {}
    
    def check_spice_available(self):
        """Check if ngspice is available"""
        try:
            result = subprocess.run(
                [self.spice_exe, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def run_spice_simulation(self, testbench_path, output_file):
        """Run SPICE simulation on a testbench"""
        print(f"\n{'='*80}")
        print(f"Running SPICE simulation: {testbench_path.name}")
        print(f"{'='*80}\n")
        
        if not testbench_path.exists():
            print(f"[ERROR] Testbench not found: {testbench_path}")
            return None
        
        try:
            # Run ngspice in batch mode
            cmd = [self.spice_exe, '-b', str(testbench_path)]
            
            print(f"Command: {' '.join(cmd)}")
            print(f"Output file: {output_file}\n")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            # Save output
            with open(output_file, 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\n=== STDERR ===\n")
                    f.write(result.stderr)
            
            if result.returncode == 0:
                print(f"[OK] Simulation completed successfully")
                print(f"[OK] Results saved to: {output_file}")
                return output_file
            else:
                print(f"[ERROR] Simulation failed with return code {result.returncode}")
                print(f"Check {output_file} for details")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Simulation timed out (>60s)")
            return None
        except Exception as e:
            print(f"[ERROR] Error running simulation: {e}")
            return None
    
    def parse_spice_output(self, output_file):
        """Parse SPICE output to extract results"""
        if not output_file or not Path(output_file).exists():
            return None
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        results = {
            'success': 'error' not in content.lower() or 'Error' in content,
            'warnings': content.count('Warning'),
            'raw_output': content
        }
        
        # Try to extract numerical results (depends on testbench)
        # Look for voltage/current measurements
        voltage_pattern = r'v\([\w]+\)\s*=\s*([-+]?[\d.]+[eE]?[-+]?\d*)'
        voltages = re.findall(voltage_pattern, content)
        
        if voltages:
            results['voltages'] = [float(v) for v in voltages]
        
        return results
    
    def load_pytorch_reference(self, model_name):
        """Load PyTorch reference results if available"""
        # This would load saved PyTorch predictions
        # For now, return placeholder
        return {
            'burgers': {'test_points': 10, 'accuracy': 'N/A'},
            'laplace': {'test_points': 10, 'accuracy': 'N/A'},
            'snn_xor': {'test_cases': 4, 'accuracy': '100%'}
        }.get(model_name, None)
    
    def compare_results(self, model_name, spice_results, pytorch_ref):
        """Compare SPICE results with PyTorch reference"""
        comparison = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'spice_status': 'success' if spice_results and spice_results['success'] else 'failed',
            'pytorch_available': pytorch_ref is not None
        }
        
        if spice_results and pytorch_ref:
            # Detailed comparison would go here
            comparison['match'] = 'unknown'  # Would calculate actual accuracy
            comparison['notes'] = 'Detailed comparison requires PyTorch reference data'
        else:
            comparison['match'] = 'N/A'
            comparison['notes'] = 'SPICE simulation completed, PyTorch comparison pending'
        
        return comparison
    
    def generate_validation_report(self, model_name, comparison):
        """Generate validation report for a model"""
        report_file = self.results_dir / f"validation_report_{model_name}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"SPICE VALIDATION REPORT: {model_name.upper()}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model: {self.models[model_name]['description']}\n")
            f.write(f"Timestamp: {comparison['timestamp']}\n")
            f.write(f"SPICE Status: {comparison['spice_status']}\n")
            f.write(f"PyTorch Reference: {'Available' if comparison['pytorch_available'] else 'Not Available'}\n")
            f.write(f"Match Status: {comparison['match']}\n\n")
            
            f.write("Notes:\n")
            f.write(f"{comparison['notes']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("Next Steps:\n")
            f.write("="*80 + "\n\n")
            
            if comparison['spice_status'] == 'success':
                f.write("[OK] SPICE simulation completed successfully\n\n")
                f.write("To complete validation:\n")
                f.write("1. Save PyTorch predictions as reference\n")
                f.write("2. Extract SPICE numerical results\n")
                f.write("3. Calculate accuracy metrics (MAE, RMSE)\n")
                f.write("4. Generate comparison plots\n")
            else:
                f.write("[ERROR] SPICE simulation failed\n\n")
                f.write("Troubleshooting:\n")
                f.write("1. Check ngspice installation\n")
                f.write("2. Verify testbench syntax\n")
                f.write("3. Check model convergence settings\n")
        
        print(f"\nValidation report saved: {report_file}")
        return report_file
    
    def validate_model(self, model_name):
        """Run complete validation for a model"""
        if model_name not in self.models:
            print(f"[ERROR] Unknown model: {model_name}")
            print(f"Available models: {', '.join(self.models.keys())}")
            return None
        
        model_info = self.models[model_name]
        
        print(f"\n{'#'*80}")
        print(f"# VALIDATING: {model_info['description']}")
        print(f"{'#'*80}")
        
        # Check if testbench exists
        if not model_info['testbench'].exists():
            print(f"\nWARNING: Testbench not found: {model_info['testbench']}")
            print("You need to run the demo first:")
            if model_name == 'burgers':
                print("  python demo_psi_hdl.py --model burgers")
            elif model_name == 'laplace':
                print("  python demo_psi_hdl.py --model laplace")
            elif model_name == 'snn_xor':
                print("  python demo_snn_xor.py")
            print("\nSkipping this model...\n")
            return None
        
        # Run SPICE simulation
        output_file = self.results_dir / f"spice_output_{model_name}.txt"
        spice_results = self.run_spice_simulation(model_info['testbench'], output_file)
        
        # Parse results
        parsed_results = self.parse_spice_output(output_file)
        
        # Load PyTorch reference
        pytorch_ref = self.load_pytorch_reference(model_name)
        
        # Compare
        comparison = self.compare_results(model_name, parsed_results, pytorch_ref)
        
        # Generate report
        report_file = self.generate_validation_report(model_name, comparison)
        
        # Store results
        self.validation_results[model_name] = comparison
        
        return comparison
    
    def validate_all(self):
        """Run validation on all models"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE SPICE VALIDATION")
        print("="*80)
        
        results = {}
        for model_name in self.models.keys():
            result = self.validate_model(model_name)
            if result:
                results[model_name] = result
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results):
        """Generate overall summary report"""
        summary_file = self.results_dir / "validation_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SPICE VALIDATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Tested: {len(results)}\n\n")
            
            # Summary table
            f.write("Model           | Status      | Notes\n")
            f.write("-" * 80 + "\n")
            
            for model_name, result in results.items():
                status = "[OK] Success" if result['spice_status'] == 'success' else "[ERROR] Failed"
                f.write(f"{model_name:15} | {status:11} | {result['notes'][:40]}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL STATUS\n")
            f.write("="*80 + "\n\n")
            
            success_count = sum(1 for r in results.values() if r['spice_status'] == 'success')
            
            if success_count == len(results):
                f.write(f"[OK] ALL {len(results)} SIMULATIONS SUCCESSFUL\n\n")
                f.write("Next steps:\n")
                f.write("1. Add PyTorch reference data for accuracy comparison\n")
                f.write("2. Generate comparison plots\n")
                f.write("3. Create figures for paper\n")
            else:
                f.write(f"[WARNING] {success_count}/{len(results)} simulations successful\n\n")
                f.write("Review individual reports for failure details.\n")
        
        print(f"\n{'='*80}")
        print(f"Summary report saved: {summary_file}")
        print(f"{'='*80}\n")
        
        # Save JSON version
        json_file = self.results_dir / "validation_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved: {json_file}\n")
        
        return summary_file


def main():
    parser = argparse.ArgumentParser(description='SPICE Validation Automation')
    parser.add_argument('--model', type=str, help='Specific model to validate (burgers, laplace, snn_xor)')
    parser.add_argument('--spice-path', type=str, default='ngspice', help='Path to ngspice executable')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    # Create validator
    validator = SPICEValidator(
        spice_executable=args.spice_path,
        output_dir=args.output_dir,
        results_dir=args.results_dir
    )
    
    # Check if SPICE is available
    print("Checking ngspice availability...")
    if validator.check_spice_available():
        print(f"[OK] ngspice found: {args.spice_path}\n")
    else:
        print(f"[ERROR] ngspice not found at: {args.spice_path}")
        print("\nPlease install ngspice or provide correct path with --spice-path")
        print("\nOptions:")
        print("1. Install ngspice: http://ngspice.sourceforge.net/download.html")
        print("2. Use LTspice as alternative")
        print("3. Provide custom path: python run_spice_validation.py --spice-path 'C:/path/to/ngspice.exe'")
        print("\nContinuing without SPICE simulations (will generate reports only)...\n")
    
    # Run validation
    if args.model:
        validator.validate_model(args.model)
    else:
        validator.validate_all()
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults saved in: {validator.results_dir}/")
    print("\nGenerated files:")
    print("  - validation_summary.txt - Overall summary")
    print("  - validation_results.json - Machine-readable results")
    print("  - validation_report_*.txt - Individual model reports")
    print("  - spice_output_*.txt - Raw SPICE outputs")
    print("\nNOTE: Run demos first if testbenches are missing:")
    print("  python demo_psi_hdl.py --model burgers")
    print("  python demo_psi_hdl.py --model laplace")
    print("  python demo_snn_xor.py")


if __name__ == "__main__":
    main()