#!/usr/bin/env python3
"""
Figure Generation Script for Paper

This script generates all figures and tables needed for the paper:
1. Network architecture diagrams
2. Hardware mapping visualizations
3. Comparison plots
4. Summary tables

Usage:
    python generate_figures.py
    python generate_figures.py --model burgers
    python generate_figures.py --output-dir figures
"""

import argparse
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime

# Try to import additional visualization libraries
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Note: networkx not installed. Network diagrams will be simplified.")
    print("Install with: pip install networkx")

class FigureGenerator:
    """Generates publication-quality figures for the paper"""
    
    def __init__(self, output_dir="output", figures_dir="figures"):
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
        
        # Models to process
        self.models = {
            'burgers': {
                'structure': self.output_dir / 'burgers' / 'burgers_structure.json',
                'description': 'Burgers Equation (Odd Function)',
                'type': 'PDE'
            },
            'laplace': {
                'structure': self.output_dir / 'laplace' / 'laplace_structure.json',
                'description': 'Laplace Equation (Even Function)',
                'type': 'PDE'
            },
            'snn_xor': {
                'structure': self.output_dir / 'snn_xor' / 'xor_structure.json',
                'description': 'SNN XOR Circuit',
                'type': 'SNN'
            }
        }
        
        # Figure style settings
        plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
        self.fig_dpi = 300
        self.fig_size = (8, 6)
    
    def load_structure(self, model_name):
        """Load network structure from JSON"""
        if model_name not in self.models:
            return None
        
        structure_file = self.models[model_name]['structure']
        if not structure_file.exists():
            print(f"[ERROR] Structure file not found: {structure_file}")
            return None
        
        with open(structure_file, 'r') as f:
            return json.load(f)
    
    def generate_architecture_diagram(self, model_name):
        """Generate network architecture diagram"""
        print(f"\nGenerating architecture diagram for {model_name}...")
        
        structure = self.load_structure(model_name)
        if not structure:
            return None
        
        # Calculate required dimensions based on number of layers
        layers = structure.get('layers', [])
        num_layers = len(layers)
        
        # Calculate actual content width
        # Input box: 1.5 wide at x=1, so spans 1 to 2.5
        # Each layer: 1.5 wide, spaced 2 units apart
        # Output box: 1.5 wide at x=(1 + num_layers*2), so ends at (2.5 + num_layers*2)
        content_start = 0.5  # Left margin
        content_end = 2.5 + num_layers * 2  # End of output box
        required_width = content_end + 0.5  # Right margin
        x_center = (content_start + content_end) / 2
        
        # Calculate required height based on content
        # Title(1) + network_diagram(1.5) + spacing(0.5) + layer_details_title(0.5) + layers(num*0.4) + activation(0.5) + margin(1)
        required_height = 1 + 1.5 + 0.5 + 0.5 + (num_layers * 0.4) + 0.5 + 1
        
        # Adjust figure size to be more compact
        fig_width = min(12, max(6, required_width * 0.7))
        fig_height = min(8, max(4, required_height * 0.8))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(content_start, content_end + 0.5)
        ax.set_ylim(0, required_height)
        ax.axis('off')
        
        # Title
        y_start = required_height - 0.5
        ax.text(x_center, y_start, f"{self.models[model_name]['description']}",
                ha='center', va='top', fontsize=14, fontweight='bold')
        
        # Network info
        input_dim = structure.get('input_dim', 'N/A')
        output_dim = structure.get('output_dim', 'N/A')
        
        # Simple visualization
        y_pos = y_start - 1.5
        
        # Input layer
        ax.add_patch(patches.Rectangle((1, y_pos-0.3), 1.5, 0.6, 
                                       facecolor='lightblue', edgecolor='black', linewidth=2))
        ax.text(1.75, y_pos, f'Input\n({input_dim})', ha='center', va='center', fontsize=10)
        
        # Hidden layers
        x_pos = 3
        for i, layer in enumerate(layers[:-1]):  # Exclude output layer
            ax.add_patch(patches.Rectangle((x_pos, y_pos-0.3), 1.5, 0.6,
                                          facecolor='lightgreen', edgecolor='black', linewidth=2))
            output_dim_layer = layer.get('output_dim', '?')
            ax.text(x_pos+0.75, y_pos, f'Layer {i+1}\n({output_dim_layer})', 
                   ha='center', va='center', fontsize=10)
            
            # Arrow
            ax.arrow(x_pos-0.3, y_pos, -0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
            x_pos += 2
        
        # Output layer
        ax.add_patch(patches.Rectangle((x_pos, y_pos-0.3), 1.5, 0.6,
                                       facecolor='lightcoral', edgecolor='black', linewidth=2))
        ax.text(x_pos+0.75, y_pos, f'Output\n({output_dim})', ha='center', va='center', fontsize=10)
        ax.arrow(x_pos-0.3, y_pos, -0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
        
        # Layer details below (more compact spacing)
        y_pos = y_start - 3.5
        ax.text(x_center, y_pos, 'Layer Details:', ha='center', fontsize=12, fontweight='bold')
        y_pos -= 0.5
        
        for i, layer in enumerate(layers):
            layer_type = layer.get('type', 'unknown')
            output_dim_layer = layer.get('output_dim', '?')
            ax.text(x_center, y_pos, f"[{i}] {layer_type} -> {output_dim_layer}",
                   ha='center', fontsize=10, family='monospace')
            y_pos -= 0.35
        
        # Activation function
        activation = structure.get('activation', 'unknown')
        y_pos -= 0.2
        ax.text(x_center, y_pos, f'Activation: {activation}', ha='center', fontsize=10, style='italic')
        
        # Save figure
        output_file = self.figures_dir / f"architecture_{model_name}.png"
        plt.tight_layout(pad=2.0)
        plt.savefig(output_file, dpi=self.fig_dpi, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        print(f"[OK] Saved: {output_file}")
        return output_file
    
    def generate_hardware_mapping(self, model_name):
        """Generate hardware mapping visualization"""
        print(f"\nGenerating hardware mapping for {model_name}...")
        
        structure = self.load_structure(model_name)
        if not structure:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Crossbar layout
        ax1.set_title('Memristor Crossbar Layout', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        
        layers = structure.get('layers', [])
        input_dim = structure.get('input_dim', 0)
        
        # Calculate memristor count
        total_memristors = 0
        y_pos = 8
        
        ax1.text(5, 9, 'Layer-by-Layer Mapping', ha='center', fontsize=11, fontweight='bold')
        
        for i, layer in enumerate(layers):
            output_dim = layer.get('output_dim', 0)
            if i == 0:
                prev_dim = input_dim
            else:
                prev_dim = layers[i-1].get('output_dim', 0)
            
            memristors = prev_dim * output_dim
            total_memristors += memristors
            
            # Draw crossbar representation
            ax1.add_patch(patches.Rectangle((2, y_pos-0.4), 6, 0.8,
                                           facecolor='lightyellow', edgecolor='black', linewidth=1.5))
            ax1.text(5, y_pos, f'Layer {i}: {prev_dim}×{output_dim} = {memristors} memristors',
                    ha='center', va='center', fontsize=10)
            y_pos -= 1.2
        
        y_pos -= 0.3
        ax1.add_patch(patches.Rectangle((2, y_pos-0.4), 6, 0.8,
                                       facecolor='lightcoral', edgecolor='black', linewidth=2))
        ax1.text(5, y_pos, f'TOTAL: {total_memristors} memristors',
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Right: Circuit components
        ax2.set_title('Circuit Components', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        
        y_pos = 8
        components = [
            ('Memristors', total_memristors, 'lightblue'),
            ('Neurons/Activations', sum(l.get('output_dim', 0) for l in layers), 'lightgreen'),
            ('Input Encoders', input_dim, 'lightyellow'),
            ('Output Decoders', structure.get('output_dim', 0), 'lightcoral')
        ]
        
        for comp_name, count, color in components:
            ax2.add_patch(patches.Rectangle((1, y_pos-0.4), 8, 0.8,
                                           facecolor=color, edgecolor='black', linewidth=1.5))
            ax2.text(5, y_pos, f'{comp_name}: {count}',
                    ha='center', va='center', fontsize=11)
            y_pos -= 1.2
        
        # Implementation notes
        y_pos -= 0.5
        ax2.text(5, y_pos, 'Implementation:', ha='center', fontsize=10, fontweight='bold')
        y_pos -= 0.5
        notes = [
            f'- Crossbar: {total_memristors} programmable synapses',
            f'- Neurons: RC circuits + comparators',
            f'- Technology: Can use ReRAM, PCM, or ECRAM'
        ]
        for note in notes:
            ax2.text(5, y_pos, note, ha='center', fontsize=9)
            y_pos -= 0.4
        
        # Save figure
        output_file = self.figures_dir / f"hardware_{model_name}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.fig_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {output_file}")
        return output_file
    
    def generate_comparison_table(self):
        """Generate model comparison table"""
        print("\nGenerating comparison table...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Collect data
        table_data = []
        headers = ['Model', 'Type', 'Layers', 'Parameters', 'Memristors', 'Status']
        
        for model_name, model_info in self.models.items():
            structure = self.load_structure(model_name)
            if structure:
                layers = structure.get('layers', [])
                input_dim = structure.get('input_dim', 0)
                
                # Calculate total parameters
                total_params = 0
                total_memristors = 0
                prev_dim = input_dim
                for layer in layers:
                    output_dim = layer.get('output_dim', 0)
                    total_params += prev_dim * output_dim
                    total_memristors += prev_dim * output_dim
                    prev_dim = output_dim
                
                table_data.append([
                    model_info['description'],
                    model_info['type'],
                    len(layers),
                    total_params,
                    total_memristors,
                    '[OK] Complete'
                ])
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.25, 0.1, 0.1, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('lightgray')
            table[(0, i)].set_text_props(weight='bold')
        
        # Style rows
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('lightblue')
        
        plt.title('Network Comparison Summary', fontsize=14, fontweight='bold', pad=20)
        
        # Save
        output_file = self.figures_dir / "comparison_table.png"
        plt.savefig(output_file, dpi=self.fig_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {output_file}")
        return output_file
    
    def generate_method_comparison(self):
        """Generate Verilog-A vs LUT comparison chart"""
        print("\nGenerating method comparison chart...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['Verilog-A\n(Psi-HDL)', 'LUT\n(2025 Paper)']
        aspects = ['Design\nFlexibility', 'Simulation\nSpeed', 'Physics\nAccuracy', 
                  'Implementation\nComplexity', 'Portability']
        
        # Ratings (0-5 scale)
        verilog_ratings = [5, 3, 5, 4, 5]
        lut_ratings = [3, 5, 4, 5, 4]
        
        x = np.arange(len(aspects))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, verilog_ratings, width, label='Verilog-A', color='skyblue')
        bars2 = ax.bar(x + width/2, lut_ratings, width, label='LUT', color='lightcoral')
        
        ax.set_ylabel('Rating (0-5)', fontsize=11)
        ax.set_title('Method Comparison: Verilog-A vs LUT', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(aspects, fontsize=10)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 5.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_file = self.figures_dir / "method_comparison.png"
        plt.savefig(output_file, dpi=self.fig_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved: {output_file}")
        return output_file
    
    def generate_all_figures(self):
        """Generate all figures"""
        print("\n" + "="*80)
        print("GENERATING ALL FIGURES FOR PAPER")
        print("="*80)
        
        figures_generated = []
        
        # Architecture diagrams for each model
        for model_name in self.models.keys():
            fig_file = self.generate_architecture_diagram(model_name)
            if fig_file:
                figures_generated.append(fig_file)
        
        # Hardware mappings
        for model_name in self.models.keys():
            fig_file = self.generate_hardware_mapping(model_name)
            if fig_file:
                figures_generated.append(fig_file)
        
        # Comparison table
        fig_file = self.generate_comparison_table()
        if fig_file:
            figures_generated.append(fig_file)
        
        # Method comparison
        fig_file = self.generate_method_comparison()
        if fig_file:
            figures_generated.append(fig_file)
        
        # Generate index
        self.generate_figure_index(figures_generated)
        
        return figures_generated
    
    def generate_figure_index(self, figures):
        """Generate index of all figures"""
        index_file = self.figures_dir / "FIGURES_INDEX.txt"
        
        with open(index_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GENERATED FIGURES INDEX\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Figures: {len(figures)}\n\n")
            
            f.write("Figure List:\n")
            f.write("-" * 80 + "\n\n")
            
            for i, fig_path in enumerate(figures, 1):
                f.write(f"{i}. {fig_path.name}\n")
                f.write(f"   Path: {fig_path}\n")
                f.write(f"   Size: {fig_path.stat().st_size / 1024:.1f} KB\n\n")
            
            f.write("="*80 + "\n")
            f.write("USAGE IN PAPER\n")
            f.write("="*80 + "\n\n")
            
            f.write("Suggested figure placement:\n\n")
            f.write("Section IV.A (Methodology Validation):\n")
            f.write("  - architecture_*.png - Network topologies\n")
            f.write("  - comparison_table.png - Model statistics\n\n")
            
            f.write("Section IV.B (Hardware Implementation):\n")
            f.write("  - hardware_*.png - Crossbar mappings\n\n")
            
            f.write("Section IV.C (Method Comparison):\n")
            f.write("  - method_comparison.png - Verilog-A vs LUT\n\n")
        
        print(f"\n[OK] Figure index saved: {index_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate figures for paper')
    parser.add_argument('--model', type=str, help='Generate figures for specific model only')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory with results')
    parser.add_argument('--figures-dir', type=str, default='figures', help='Directory to save figures')
    
    args = parser.parse_args()
    
    # Create generator
    generator = FigureGenerator(
        output_dir=args.output_dir,
        figures_dir=args.figures_dir
    )
    
    # Generate figures
    if args.model:
        generator.generate_architecture_diagram(args.model)
        generator.generate_hardware_mapping(args.model)
    else:
        figures = generator.generate_all_figures()
        
        print("\n" + "="*80)
        print("FIGURE GENERATION COMPLETE")
        print("="*80)
        print(f"\nGenerated {len(figures)} figures in: {generator.figures_dir}/")
        print("\nFigures ready for inclusion in paper!")
        print("See FIGURES_INDEX.txt for usage guide.")


if __name__ == "__main__":
    main()