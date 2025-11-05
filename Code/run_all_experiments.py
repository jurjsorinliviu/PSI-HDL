#!/usr/bin/env python3
"""
Master Experiment Runner for IEEE Access Submission

This script runs all three critical experiments required for the manuscript:
1. VTEAM Baseline Comparison
2. Cross-Validation Analysis
3. Noise Robustness Study

Run this script to generate all experimental results needed.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_experiment(script_name, description):
    """
    Run an experiment script and report results
    
    Args:
        script_name: Name of the Python script to run
        description: Description of the experiment
    
    Returns:
        success: True if experiment completed successfully
        duration: Time taken in seconds
    """
    print_header(description)
    print(f"Running: {script_name}")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True,
            check=True
        )
        
        duration = time.time() - start_time
        
        print(f"\n✓ {description} completed successfully!")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n✗ {description} failed!")
        print(f"Error: {e}")
        print(f"Duration before failure: {duration:.1f} seconds")
        
        return False, duration

def check_prerequisites():
    """Check if all prerequisites are met"""
    print_header("Checking Prerequisites")
    
    # Check if memristor data exists
    data_path = Path(__file__).parent / "output" / "memristor" / "memristor_training_data.csv"
    
    if not data_path.exists():
        print("✗ Memristor training data not found!")
        print(f"  Expected location: {data_path}")
        print("\n  Please run demo_memristor.py first to generate training data:")
        print("  $ python Code/demo_memristor.py\n")
        return False
    
    print("✓ Memristor training data found")
    
    # Check if PINN model exists
    model_path = Path(__file__).parent / "output" / "memristor" / "memristor_pinn.pth"
    
    if not model_path.exists():
        print("⚠ Memristor PINN model not found")
        print(f"  Expected location: {model_path}")
        print("\n  VTEAM comparison will run without PINN baseline.")
        print("  For full comparison, run demo_memristor.py first.\n")
    else:
        print("✓ Memristor PINN model found")
    
    # Check required packages
    print("\nChecking required packages...")
    required_packages = ['numpy', 'pandas', 'torch', 'matplotlib', 'scipy']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n✗ Missing required packages: {', '.join(missing_packages)}")
        print("\n  Install missing packages with:")
        print(f"  $ pip install {' '.join(missing_packages)}\n")
        return False
    
    print("\n✓ All prerequisites met!")
    return True

def generate_summary_report(results, total_time):
    """Generate final summary report"""
    print_header("EXPERIMENTAL RESULTS SUMMARY")
    
    print("Experiment Status:")
    print("-" * 80)
    
    for exp_name, (success, duration) in results.items():
        status = "✓ COMPLETED" if success else "✗ FAILED"
        print(f"  {exp_name:40s} {status:15s} {duration:8.1f}s ({duration/60:5.1f}m)")
    
    print("-" * 80)
    print(f"  {'TOTAL TIME':40s} {'':15s} {total_time:8.1f}s ({total_time/60:5.1f}m)")
    print()
    
    # Count successes
    n_completed = sum(1 for success, _ in results.values() if success)
    n_total = len(results)
    
    if n_completed == n_total:
        print("✓ All experiments completed successfully!")
    else:
        print(f"⚠ {n_completed}/{n_total} experiments completed")
    
    print("\nGenerated Output Directories:")
    print("-" * 80)
    output_dirs = [
        ("VTEAM Comparison", "Code/output/vteam_comparison/"),
        ("Cross-Validation", "Code/output/cross_validation/"),
        ("Noise Robustness", "Code/output/noise_robustness/")
    ]
    
    for name, dir_path in output_dirs:
        full_path = Path(__file__).parent / dir_path.replace("Code/", "")
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name:20s} → {dir_path}")
    
    print("\nKey Files for Manuscript:")
    print("-" * 80)
    key_files = [
        "output/vteam_comparison/vteam_pinn_comparison.csv",
        "output/vteam_comparison/vteam_pinn_comparison.png",
        "output/cross_validation/cross_validation_results.csv",
        "output/cross_validation/cv_metrics_summary.png",
        "output/noise_robustness/noise_robustness_results.csv",
        "output/noise_robustness/noise_robustness_metrics.png"
    ]
    
    for file_rel in key_files:
        file_path = Path(__file__).parent / file_rel
        exists = file_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_rel}")
    
    print("\nNext Steps:")
    print("-" * 80)
    print("  1. Review experimental results in output directories")
    print("  2. Integrate figures and tables into manuscript")
    print("  3. Update manuscript text with quantitative findings")
    print("  4. Address Poisson inconsistency (see POISSON_INCONSISTENCY_REPORT.md)")
    print("  5. Reduce abstract to 200 words")
    print("  6. Prepare GitHub repository with all code and data")
    print()

def main():
    """Main execution"""
    print("="*80)
    print("  Ψ-HDL EXPERIMENTAL SUITE")
    print("  IEEE Access Submission - Comprehensive Evaluation")
    print("="*80)
    print(f"\nStarted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n✗ Prerequisites not met. Please resolve issues and try again.\n")
        return 1
    
    # Define experiments
    experiments = [
        ("vteam_baseline.py", "VTEAM Baseline Comparison"),
        ("cross_validation.py", "Cross-Validation Analysis"),
        ("noise_robustness.py", "Noise Robustness Study")
    ]
    
    # Ask for confirmation
    print(f"\nReady to run {len(experiments)} experiments.")
    print("Estimated total time: ~15-30 minutes (depending on hardware)")
    
    response = input("\nProceed with all experiments? [y/N]: ")
    if response.lower() != 'y':
        print("\nExperiments cancelled by user.\n")
        return 0
    
    # Run experiments
    overall_start = time.time()
    results = {}
    
    for script_name, description in experiments:
        success, duration = run_experiment(script_name, description)
        results[description] = (success, duration)
        
        if not success:
            print(f"\n⚠ Experiment failed. Continuing with remaining experiments...")
        
        # Small pause between experiments
        time.sleep(2)
    
    total_time = time.time() - overall_start
    
    # Generate summary
    generate_summary_report(results, total_time)
    
    print(f"\nCompleted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Return 0 if all successful, 1 if any failed
    return 0 if all(s for s, _ in results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())