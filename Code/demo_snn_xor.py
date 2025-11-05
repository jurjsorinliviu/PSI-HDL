"""
Demo: SNN XOR Circuit to Verilog-A HDL

This demo shows how to translate a trained SNN (from 2024 Ex-Situ paper)
into Verilog-A hardware description for circuit simulation.

Case Study: XOR logic with memristor-based SNN
- 2 inputs -> 2 hidden (LIF) -> 1 output
- Trained weights from the 2024 paper
- Demonstrates neuromorphic hardware design workflow

Author: Psi-HDL Framework
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Import Psi-HDL framework
from snn_loader import load_snn_from_json, convert_snn_to_rate_coded
from structure_extractor import StructureExtractor
from verilog_generator import VerilogAGenerator
from spice_validator import SPICEValidator


def demo_snn_xor():
    """
    Complete workflow: SNN XOR model -> Verilog-A HDL
    """
    print("\n" + "="*80)
    print(" SNN XOR CIRCUIT -> HARDWARE DEMONSTRATION")
    print(" From the 2024 Ex-Situ Training Paper")
    print("="*80 + "\n")
    
    # ============================================================================
    # STEP 1: Load Trained SNN Model
    # ============================================================================
    print("STEP 1: Loading Trained SNN Model")
    print("-" * 80)
    
    # Path to trained model
    model_path = "../related papers/SNNs_for_Inference_using_Ex-Situ_Training-main/Net_xor_model.json"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("\nNote: This demo requires the trained XOR model.")
        print("      Place it in the 'related papers' folder or adjust the path.")
        return None
    
    # Load SNN
    snn_model = load_snn_from_json(model_path)
    summary = snn_model.get_architecture_summary()
    
    print(f"[OK] Loaded SNN Model:")
    print(f"  Architecture: {summary['num_inputs']} -> {summary['hidden_layers']} -> {summary['num_outputs']}")
    print(f"  Neuron Type: {summary['neuron_type']}")
    print(f"  Total Parameters: {summary['total_parameters']}")
    print(f"  Memristor params: R={snn_model.lif_layers[0].R}Ohm, C={snn_model.lif_layers[0].C}F")
    
    # Test XOR functionality
    print("\nTesting XOR Logic:")
    print("  Input (A, B) | Output | Expected")
    print("  " + "-"*40)
    
    xor_test = torch.tensor([
        [0.0, 0.0],  # 0 XOR 0 = 0
        [0.0, 1.0],  # 0 XOR 1 = 1
        [1.0, 0.0],  # 1 XOR 0 = 1
        [1.0, 1.0],  # 1 XOR 1 = 0
    ])
    
    with torch.no_grad():
        outputs = snn_model(xor_test, time_steps=100)
    
    for i, (inp, out) in enumerate(zip(xor_test, outputs)):
        expected = int(inp[0] != inp[1])  # XOR truth table
        actual = 1 if out.item() > 0.5 else 0
        match = "[OK]" if actual == expected else "[FAIL]"
        print(f"  ({inp[0]:.0f}, {inp[1]:.0f})      | {out.item():.3f}  | {expected}        {match}")
    
    # ============================================================================
    # STEP 2: Convert to Rate-Coded Network
    # ============================================================================
    print("\n" + "-"*80)
    print("STEP 2: Converting to Rate-Coded Network")
    print("-" * 80)
    
    print("\nNote: SNNs use spiking dynamics.")
    print("      For hardware synthesis, we extract the weighted connections")
    print("      and convert to rate-coded representation (compatible with Verilog-A)")
    
    rate_model = convert_snn_to_rate_coded(snn_model)
    
    print("\n[OK] Converted to rate-coded network")
    print("  - Removed spiking dynamics")
    print("  - Preserved synaptic weights")
    print("  - Compatible with standard HDL generation")
    
    # Verify rate-coded version
    print("\nRate-Coded XOR Test:")
    with torch.no_grad():
        rate_outputs = rate_model(xor_test)
    
    for i, (inp, out) in enumerate(zip(xor_test, rate_outputs)):
        print(f"  ({inp[0]:.0f}, {inp[1]:.0f}) -> {out.item():.3f}")
    
    # ============================================================================
    # STEP 3: Extract Network Structure
    # ============================================================================
    print("\n" + "-"*80)
    print("STEP 3: Extracting Network Structure")
    print("-" * 80)
    
    # Use structure extractor (works with standard PyTorch models)
    extractor = StructureExtractor(rate_model, model_type="SNN_XOR")
    structure = extractor.extract()
    
    # Print summary
    extractor.print_summary()
    
    # Save structure
    output_dir = "./output/snn_xor"
    os.makedirs(output_dir, exist_ok=True)
    
    structure_json = os.path.join(output_dir, "xor_structure.json")
    structure.save_json(structure_json)
    print(f"\n[OK] Structure saved to {structure_json}")
    
    # ============================================================================
    # STEP 4: Generate Verilog-A HDL
    # ============================================================================
    print("\n" + "-"*80)
    print("STEP 4: Generating Verilog-A HDL")
    print("-" * 80)
    
    generator = VerilogAGenerator(structure)
    
    # Generate main module
    verilog_path = os.path.join(output_dir, f"{generator.module_name}.va")
    verilog_code = generator.generate(verilog_path)
    print(f"[OK] Verilog-A module: {verilog_path}")
    
    # Generate weight parameters
    params_path = os.path.join(output_dir, f"{generator.module_name}_params.txt")
    generator.generate_weight_file(rate_model, params_path)
    print(f"[OK] Weight parameters: {params_path}")
    
    # Generate SPICE testbench
    test_points = xor_test.numpy()
    tb_path = os.path.join(output_dir, f"{generator.module_name}_tb.sp")
    generator.generate_testbench(test_points, tb_path)
    print(f"[OK] SPICE testbench: {tb_path}")
    
    # ============================================================================
    # STEP 5: Hardware Design Information
    # ============================================================================
    print("\n" + "-"*80)
    print("STEP 5: Hardware Implementation Details")
    print("-" * 80)
    
    print("\n[SPEC] Memristor Crossbar Mapping:")
    print("  Input Layer:  2 inputs × 2 neurons  = 4 memristors")
    print("  Hidden Layer: 2 neurons × 1 output  = 2 memristors")
    print("  Total:        6 memristors")
    
    print("\n[PARAM] Neuron Parameters (from training):")
    print(f"  Resistance: {snn_model.lif_layers[0].R} Ohm")
    print(f"  Capacitance: {snn_model.lif_layers[0].C} F")
    print(f"  Time step: {snn_model.lif_layers[0].time_step} s")
    print(f"  Decay (beta): {snn_model.lif_layers[0].beta:.4f}")
    
    print("\n[TOOL] Circuit Components:")
    print("  - 6 memristors (programmable synapses)")
    print("  - 3 LIF neuron circuits (RC + comparator)")
    print("  - Input encoding circuitry")
    print("  - Output decoding circuitry")
    
    # ============================================================================
    # STEP 6: Summary
    # ============================================================================
    print("\n" + "="*80)
    print(" DEMO SUMMARY")
    print("="*80)
    
    print("\n[OK] Successfully translated SNN XOR to Verilog-A HDL!")
    
    print("\n[FILES] Generated Files:")
    print(f"  - {structure_json}")
    print(f"  - {verilog_path}")
    print(f"  - {params_path}")
    print(f"  - {tb_path}")
    
    print("\n[NEXT] Next Steps:")
    print("  1. Run SPICE simulation with ngspice:")
    print(f"     ngspice -b {tb_path}")
    print("  2. Compare SPICE vs PyTorch outputs")
    print("  3. Integrate into larger circuit design")
    print("  4. Use for neuromorphic chip tape-out")
    
    print("\n[INFO] Research Impact:")
    print("  - Demonstrates automated SNN -> hardware workflow")
    print("  - Enables rapid prototyping of neuromorphic circuits")
    print("  - Bridges machine learning and VLSI design")
    print("  - Uses trained model from the 2024 paper")

    print("\n" + "="*80)
    print(" DEMO COMPLETE")
    print("="*80 + "\n")
    
    return {
        'snn_model': snn_model,
        'rate_model': rate_model,
        'structure': structure,
        'verilog_path': verilog_path,
        'output_dir': output_dir
    }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SNN XOR -> Verilog-A Demo')
    parser.add_argument('--model', type=str, 
                       default='../related papers/SNNs_for_Inference_using_Ex-Situ_Training-main/Net_xor_model.json',
                       help='Path to XOR model JSON')
    
    args = parser.parse_args()
    
    results = demo_snn_xor()
    
    if results:
        print("Thank you for using the Psi-HDL Framework!")
        print("This demo showcases research from the 2024 Ex-Situ Training paper.\n")


if __name__ == "__main__":
    main()