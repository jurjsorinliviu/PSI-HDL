"""
Psi-HDL Pipeline Demo
Demonstrates the complete workflow: Train -> Extract -> Translate -> Validate

This demo shows how to:
1. Load a trained Psi-NN model (Burgers equation)
2. Extract the discovered network structure
3. Translate to Verilog-A HDL
4. Generate SPICE testbench
5. Validate against PyTorch reference

Author: Psi-HDL Framework
"""

import os
import torch
import numpy as np

# Import local implementations with proper attribution
# PsiNN_burgers, PsiNN_laplace, and PINN are from the original Psi-NN paper:
# Source: https://github.com/ZitiLiu/Psi-NN
# Citation: Liu, Z., Liu, Y., Yan, X. et al. Automatic network structure
#           discovery of physics informed neural networks via knowledge
#           distillation. Nat Commun 16, 9558 (2025).
#           https://doi.org/10.1038/s41467-025-64624-3
from structure_extractor import StructureExtractor, NetworkStructure
from verilog_generator import VerilogAGenerator, generate_complete_hdl_package
from spice_validator import SPICEValidator, quick_validate
import PsiNN_burgers
import PsiNN_laplace


def demo_psinn_burgers():
    """
    Demo: Complete Psi-HDL pipeline for PsiNN Burgers equation model
    """
    print("\n" + "="*80)
    print(" Psi-HDL PIPELINE DEMONSTRATION")
    print(" Case Study: Burgers Equation with PsiNN")
    print("="*80 + "\n")
    
    # ============================================================================
    # STEP 1: Create/Load Model
    # ============================================================================
    print("STEP 1: Creating PsiNN Burgers Model")
    print("-" * 80)
    
    # Create model instance
    node_num = 20  # Number of nodes per layer
    model = PsiNN_burgers.Net(node_num=node_num, output_num=1)
    print(f"[OK] Model created with {node_num} nodes per layer")
    print(f"  Model type: {model.__module__.split('.')[-1]}")
    
    # Initialize with random weights (in real case, load trained weights)
    print("\nNote: Using random weights for demonstration.")
    print("      In practice, load trained model weights with:")
    print("      model.load_state_dict(torch.load('path/to/model.pth'))")
    
    # ============================================================================
    # STEP 2: Extract Network Structure
    # ============================================================================
    print("\n" + "-"*80)
    print("STEP 2: Extracting Network Structure")
    print("-" * 80)
    
    extractor = StructureExtractor(model, model_type="PsiNN_burgers")
    structure = extractor.extract()
    
    # Print summary
    extractor.print_summary()
    
    # Save structure to JSON
    output_dir = "./output/burgers"
    os.makedirs(output_dir, exist_ok=True)
    
    structure_json = os.path.join(output_dir, "burgers_structure.json")
    structure.save_json(structure_json)
    print(f"[OK] Structure saved to {structure_json}")
    
    # Export weights
    weights_npz = os.path.join(output_dir, "burgers_weights.npz")
    extractor.export_weights(weights_npz)
    print(f"[OK] Weights exported to {weights_npz}")
    
    # ============================================================================
    # STEP 3: Generate Verilog-A HDL
    # ============================================================================
    print("\n" + "-"*80)
    print("STEP 3: Generating Verilog-A HDL")
    print("-" * 80)
    
    generator = VerilogAGenerator(structure)
    
    # Generate main Verilog-A module
    verilog_path = os.path.join(output_dir, f"{generator.module_name}.va")
    verilog_code = generator.generate(verilog_path)
    print(f"[OK] Verilog-A module generated: {verilog_path}")
    
    # Generate weight parameters file
    params_path = os.path.join(output_dir, f"{generator.module_name}_params.txt")
    generator.generate_weight_file(model, params_path)
    print(f"[OK] Weight parameters saved: {params_path}")
    
    # Generate test points for testbench
    test_points = np.random.uniform(-1, 1, (100, 2))
    tb_path = os.path.join(output_dir, f"{generator.module_name}_tb.sp")
    generator.generate_testbench(test_points, tb_path)
    print(f"[OK] SPICE testbench generated: {tb_path}")
    
    # ============================================================================
    # STEP 4: Validate Results
    # ============================================================================
    print("\n" + "-"*80)
    print("STEP 4: Validation")
    print("-" * 80)
    
    # Create validator
    validator = SPICEValidator(model, verilog_path)
    
    # Generate structured test points
    print("\nGenerating test points for validation...")
    test_points = validator.generate_test_points(
        x_range=(0.0, 1.0),
        y_range=(-1.0, 1.0),
        num_points=20
    )
    print(f"[OK] Generated {len(test_points)} test points")
    
    # Get PyTorch predictions
    print("\nComputing PyTorch reference predictions...")
    pytorch_pred = validator.predict_pytorch(test_points)
    print(f"[OK] Computed predictions for {len(pytorch_pred)} points")
    print(f"  Output range: [{pytorch_pred.min():.4f}, {pytorch_pred.max():.4f}]")
    
    # Quick validation (PyTorch only)
    print("\nRunning quick validation (PyTorch statistics)...")
    quick_results = quick_validate(model, test_points)
    
    # Note about SPICE simulation
    print("\n" + "-"*80)
    print("NOTE: Full SPICE Validation")
    print("-"*80)
    print("To run complete validation with SPICE simulation:")
    print("  1. Install ngspice: https://ngspice.sourceforge.io/")
    print("  2. Ensure Verilog-A is supported (or use Xyce)")
    print("  3. Run: validator.full_validation_workflow(...)")
    print("\nFor now, PyTorch reference results have been generated.")
    
    # ============================================================================
    # STEP 5: Summary
    # ============================================================================
    print("\n" + "="*80)
    print(" PIPELINE SUMMARY")
    print("="*80)
    print("\n[OK] Generated Files:")
    print(f"  - Structure JSON:      {structure_json}")
    print(f"  - Weight NPZ:          {weights_npz}")
    print(f"  - Verilog-A module:    {verilog_path}")
    print(f"  - Weight parameters:   {params_path}")
    print(f"  - SPICE testbench:     {tb_path}")
    
    print("\n[OK] Key Discoveries in Burgers Model:")
    for key, value in structure.special_structures.items():
        print(f"  - {key}: {value['description']}")
    
    print("\n[OK] Next Steps:")
    print("  1. Load trained model weights (currently using random)")
    print("  2. Run SPICE simulation for hardware validation")
    print("  3. Compare PyTorch vs SPICE results")
    print("  4. Analyze accuracy and performance metrics")
    
    print("\n" + "="*80)
    print(" DEMO COMPLETE")
    print("="*80 + "\n")
    
    return {
        'model': model,
        'structure': structure,
        'verilog_path': verilog_path,
        'test_points': test_points,
        'pytorch_pred': pytorch_pred,
        'output_dir': output_dir
    }


def demo_psinn_laplace():
    """
    Demo: Psi-HDL pipeline for PsiNN Laplace equation model
    """
    print("\n" + "="*80)
    print(" Psi-HDL PIPELINE - LAPLACE EQUATION")
    print("="*80 + "\n")
    
    # Create model
    print("Creating PsiNN Laplace Model...")
    model = PsiNN_laplace.Net(node_num=20, output_num=1)
    print(f"[OK] Model created: {model.__module__.split('.')[-1]}")
    
    # Extract structure
    print("\nExtracting structure...")
    extractor = StructureExtractor(model, model_type="PsiNN_laplace")
    structure = extractor.extract()
    extractor.print_summary()
    
    # Save structure and weights
    output_dir = "./output/laplace"
    os.makedirs(output_dir, exist_ok=True)
    
    structure_json = os.path.join(output_dir, "laplace_structure.json")
    structure.save_json(structure_json)
    print(f"[OK] Structure saved to {structure_json}")
    
    weights_npz = os.path.join(output_dir, "laplace_weights.npz")
    extractor.export_weights(weights_npz)
    print(f"[OK] Weights exported to {weights_npz}")
    
    # Generate HDL
    print("\nGenerating Verilog-A...")
    generator = VerilogAGenerator(structure)
    verilog_path = os.path.join(output_dir, f"{generator.module_name}.va")
    generator.generate(verilog_path)
    print(f"[OK] Verilog-A generated: {verilog_path}")
    
    # Generate testbench for consistency
    params_path = os.path.join(output_dir, f"{generator.module_name}_params.txt")
    generator.generate_weight_file(model, params_path)
    print(f"[OK] Weight parameters saved: {params_path}")
    
    test_points = np.random.uniform(-1, 1, (100, 2))
    tb_path = os.path.join(output_dir, f"{generator.module_name}_tb.sp")
    generator.generate_testbench(test_points, tb_path)
    print(f"[OK] SPICE testbench generated: {tb_path}")
    
    print("\n[OK] Laplace demo complete!")
    print(f"  Output directory: {output_dir}")
    
    return {
        'model': model,
        'structure': structure,
        'verilog_path': verilog_path,
        'output_dir': output_dir
    }


def compare_structures():
    """
    Demo: Compare Burgers and Laplace discovered structures
    """
    print("\n" + "="*80)
    print(" COMPARING DISCOVERED STRUCTURES")
    print("="*80 + "\n")
    
    # Create both models
    burgers_model = PsiNN_burgers.Net(node_num=20, output_num=1)
    laplace_model = PsiNN_laplace.Net(node_num=20, output_num=1)
    
    # Extract structures
    burgers_extractor = StructureExtractor(burgers_model, "PsiNN_burgers")
    laplace_extractor = StructureExtractor(laplace_model, "PsiNN_laplace")
    
    burgers_structure = burgers_extractor.extract()
    laplace_structure = laplace_extractor.extract()
    
    # Compare
    from structure_extractor import compare_structures
    compare_structures(burgers_structure, laplace_structure)
    
    print("\nKey Differences:")
    print("-" * 80)
    print("\nBurgers Equation (Odd Function):")
    print("  - Output has NO bias (preserves odd function property)")
    print("  - Special +/-W structure for advection-diffusion physics")
    
    print("\nLaplace Equation (Even Function):")
    print("  - Output HAS bias (allows even function property)")
    print("  - Symmetric structure for harmonic solutions")
    
    print("\n[OK] Comparison complete!")


def main():
    """
    Main demo entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Psi-HDL Pipeline Demo')
    parser.add_argument('--model', type=str, default='burgers',
                       choices=['burgers', 'laplace', 'compare', 'all'],
                       help='Which demo to run')
    
    args = parser.parse_args()
    
    if args.model == 'burgers':
        results = demo_psinn_burgers()
    elif args.model == 'laplace':
        results = demo_psinn_laplace()
    elif args.model == 'compare':
        compare_structures()
    elif args.model == 'all':
        print("\n" + "="*80)
        print(" RUNNING ALL DEMOS")
        print("="*80)
        
        results_burgers = demo_psinn_burgers()
        input("\nPress Enter to continue to Laplace demo...")
        
        results_laplace = demo_psinn_laplace()
        input("\nPress Enter to continue to structure comparison...")
        
        compare_structures()
        
        print("\n" + "="*80)
        print(" ALL DEMOS COMPLETE")
        print("="*80 + "\n")
    
    print("\nThank you for trying the Psi-HDL Framework!")
    print("For more information, see the documentation in README.md\n")


if __name__ == "__main__":
    main()