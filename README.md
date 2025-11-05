# Ψ-HDL: Physics Structured-Informed Neural Networks for Hardware Description Language Generation

> 🔬 **Submitted to IEEE Access** | 🚀 **Extends Ψ-NN to HDL Generation** | ⚡ **99.6% Parameter Reduction**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Ψ-HDL is a novel framework that extends [Ψ-NN](https://github.com/ZitiLiu/Psi-NN) (Published in Nature Communications) to automatically generate hardware description language (Verilog-A) code from Physics-Informed Neural Networks (PINNs). The framework achieves **99.6% parameter reduction** while maintaining high accuracy across diverse applications: PDEs, neuromorphic circuits, and analog devices.

---

## 🎯 Key Features

- **Automatic HDL Generation**: Transform trained PINNs into synthesizable Verilog-A code
- **Extreme Compression**: 99.6% parameter reduction (3482 → 12 parameters for memristor model)
- **Multi-Domain Support**: Continuous PDEs, discrete circuits, analog device characterization
- **Validated Performance**: Outperforms industry-standard baselines (13.2% better than VTEAM)
- **Robust Generalization**: Consistent steady-state prediction across 3-fold cross-validation
- **Noise Tolerance**: Graceful degradation (16% at SNR = 6.5 dB)

---

## 📊 Results Summary

| **Application** | **Original Parameters** | **Compressed Parameters** | **Compression** | **Error (MAE)** |
|-----------------|-------------------------|---------------------------|-----------------|-----------------|
| Burgers Equation | 3482 | 12 | 99.66% | 3.24×10⁻³ |
| Laplace Equation | 3482 | 11 | 99.68% | 5.12×10⁻⁴ |
| SNN XOR Circuit | 3482 | 14 | 99.60% | 2.35×10⁻² |
| Memristor Device | 3482 | 12 | 99.66% | 1.33×10⁻⁴ A |

**Benchmark Comparison** (Memristor):
- **VTEAM Baseline**: MAE = 1.531×10⁻⁴ A, RMSE = 3.110×10⁻⁴ A
- **Ψ-HDL (Ours)**: MAE = 1.328×10⁻⁴ A, RMSE = 1.931×10⁻⁴ A
- **Improvement**: +13.2% MAE, +37.9% RMSE

---

## 🛠️ Installation

### Requirements

- Python 3.11+
- CUDA 11.7+ (for GPU acceleration)
- PyTorch 2.0+

### Quick Install

```bash
# Clone the repository
git clone https://github.com/jurjsorinliviu/PSI-HDL.git
cd PSI-HDL

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Tested Environment

- **OS**: Windows 11 Pro
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **CPU**: Intel Core i9-13900K
- **RAM**: 128GB DDR5

---

## 🚀 Quick Start

### 1. Run a Demo

```bash
# Burgers Equation Demo (Ψ-NN method)
python Code/demo_psi_hdl.py --model burgers

# Laplace Equation Demo (Ψ-NN method)
python Code/demo_psi_hdl.py --model laplace

# SNN XOR Circuit Demo
python Code/demo_snn_xor.py

# Memristor Device Demo
python Code/demo_memristor.py

# Run all Ψ-NN demos
python Code/demo_psi_hdl.py --model all
```

### 2. Run Complete Pipeline

```bash
# Complete pipeline: Train → Extract → Generate Verilog-A
python Code/demo_psi_hdl.py --model burgers

# Or for comparison between Burgers and Laplace
python Code/demo_psi_hdl.py --model compare
```

### 3. Run Experimental Validation

```bash
# VTEAM Comparison + Cross-Validation + Noise Robustness
python Code/run_all_experiments.py
```

---

## 📚 Case Studies

### Case Study A: Burgers Equation

**PDE**: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂²x

```bash
python Code/demo_psi_hdl.py --model burgers
```

**Outputs**:
- Extracted structure: `Code/output/burgers/burgers_structure.json`
- Weight parameters: `Code/output/burgers/burgers_weights.npz`
- Verilog-A code: `Code/output/burgers/psi_nn_PsiNN_burgers.va`
- Parameters file: `Code/output/burgers/psi_nn_PsiNN_burgers_params.txt`
- SPICE testbench: `Code/output/burgers/psi_nn_PsiNN_burgers_tb.sp`

**Results**:
- Compression: 3482 → 12 parameters (99.66%)
- MAE: 3.24×10⁻³
- SPICE simulation validated

---

### Case Study B: Laplace Equation

**PDE**: ∂²u/∂²x + ∂²u/∂²y = 0 (Dirichlet boundary conditions)

```bash
python Code/demo_psi_hdl.py --model laplace
```

**Outputs**:
- Extracted structure: `Code/output/laplace/laplace_structure.json`
- Weight parameters: `Code/output/laplace/laplace_weights.npz`
- Verilog-A code: `Code/output/laplace/psi_nn_PsiNN_laplace.va`
- Parameters file: `Code/output/laplace/psi_nn_PsiNN_laplace_params.txt`
- SPICE testbench: `Code/output/laplace/psi_nn_PsiNN_laplace_tb.sp`

**Results**:
- Compression: 3482 → 11 parameters (99.68%)
- MAE: 5.12×10⁻⁴
- Boundary condition accuracy: 99.7%

---

### Case Study C: SNN XOR Circuit

**Description**: Spiking Neural Network implementing XOR logic gate

```bash
python Code/demo_snn_xor.py
```

**Outputs**:
- Extracted structure: `Code/output/snn_xor/xor_structure.json`
- Verilog-A code: `Code/output/snn_xor/psi_nn_SNN_XOR.va`
- Parameters file: `Code/output/snn_xor/psi_nn_SNN_XOR_params.txt`
- SPICE testbench: `Code/output/snn_xor/psi_nn_SNN_XOR_tb.sp`

**Results**:
- Compression: 3482 → 14 parameters (99.60%)
- Logic accuracy: 97.65%
- Spike timing precision: ±2.3 ns

---

### Case Study D: Memristor Device

**Model**: Voltage-controlled memristor with hysteresis

```bash
python Code/demo_memristor.py
```

**Outputs**:
- Trained model: `Code/output/memristor/memristor_pinn.pth`
- Extracted structure: `Code/output/memristor/structure.json`
- Training data: `Code/output/memristor/memristor_training_data.csv`
- Verilog-A code: `Code/output/memristor/memristor_pinn.va`
- SPICE testbench: `Code/output/memristor/memristor_pinn_tb.sp`
- I-V characteristics: `Code/output/memristor/figures/memristor_iv_curve.png`
- State evolution: `Code/output/memristor/figures/memristor_state_evolution.png`
- Error distribution: `Code/output/memristor/figures/memristor_error_distribution.png`

**Results**:
- Compression: 3482 → 12 parameters (99.66%)
- MAE: 1.33×10⁻⁴ A
- **Beats VTEAM by 13.2%** (industry standard)
- Hysteresis loop error: 2.1%

---

## 🔬 Experimental Validation

### Experiment 1: VTEAM Baseline Comparison

```bash
python Code/vteam_baseline.py
```

**Results**:

- Ψ-HDL achieves **13.2% lower MAE** than state-of-the-art VTEAM model
- Training time: 180s (Ψ-HDL) vs 0.05s (VTEAM)
- Structure discovery: Yes (Ψ-HDL) vs No (VTEAM)

---

### Experiment 2: Cross-Validation Analysis

```bash
python Code/cross_validation.py
```

**Results**:

- 3-fold cross-validation shows robust steady-state prediction
- Folds 2-3 achieve consistent performance (MAE ~2.4×10⁻⁴ A)
- **Key finding**: Forming cycle differs from steady-state physics (Fold 1 MAE = 7.63×10⁻³ A)

**Figures**:
- `Code/output/cross_validation/cv_predictions_all_folds.png` - All fold predictions
- `Code/output/cross_validation/cv_metrics_summary.png` - Metrics comparison

---

### Experiment 3: Noise Robustness

```bash
python Code/noise_robustness.py
```

**Results**:

- Tested at 5 SNR levels: 36 dB → 6.5 dB
- **Graceful degradation**: 16% MAE increase at extreme noise (SNR = 6.5 dB)
- Physics-informed regularization enhances noise tolerance

**Figure**:

- `Code/output/noise_robustness/noise_robustness_metrics.png` - MAE vs SNR curve

---

## 📁 Repository Structure

```
PSI-HDL/
├── Code/
│   ├── demo_psi_hdl.py          # Burgers & Laplace equation demos (Ψ-NN)
│   ├── demo_snn_xor.py          # SNN XOR circuit demo
│   ├── demo_memristor.py        # Memristor device demo
│   ├── vteam_baseline.py        # VTEAM comparison experiment
│   ├── cross_validation.py      # Cross-validation experiment
│   ├── noise_robustness.py      # Noise robustness experiment
│   ├── run_all_experiments.py   # Run all experiments (one-click)
│   ├── structure_extractor.py   # Hierarchical clustering module
│   ├── verilog_generator.py     # Verilog-A code generation
│   ├── spice_validator.py       # SPICE validation utilities
│   ├── PsiNN_burgers.py         # Ψ-NN Burgers equation model
│   ├── PsiNN_laplace.py         # Ψ-NN Laplace equation model
│   ├── snn_loader.py            # SNN model loader utilities
│   ├── PINN.py                  # Base PINN implementation
│   └── output/                  # Generated results
│       ├── burgers/             # Burgers equation outputs
│       ├── laplace/             # Laplace equation outputs
│       ├── snn_xor/             # SNN XOR outputs
│       ├── memristor/           # Memristor outputs
│       ├── vteam_comparison/    # VTEAM experiment results
│       ├── cross_validation/    # Cross-validation results
│       └── noise_robustness/    # Noise robustness results
│
├── Psi-NN-main/                 # Original Ψ-NN codebase (baseline)
│   ├── Panel.py                 # Ψ-NN console entry point
│   ├── Config/                  # Hyperparameter configurations
│   ├── Database/                # Training datasets
│   └── Module/                  # Core Ψ-NN modules
│
├── requirements.txt             # Python dependencies
├── LICENSE                 	 # Apache License 2.0.
```

---

## 🎓 Methodology

### Three-Stage Pipeline

```
Stage 1: PINN Training
   ↓ (Physics-informed loss)
Stage 2: Knowledge Distillation + L₂ Regularization
   ↓ (Compress 3482 → 12 parameters)
Stage 3: Structure Extraction + HDL Generation
   ↓ (Hierarchical clustering → Verilog-A)
OUTPUT: Synthesizable HDL Code
```

### Key Algorithms

1. **Physics-Informed Training** (See [`PINN.py`](Psi-NN-main/Module/PINN.py))
   
   ```python
   loss = loss_physics + loss_data + loss_boundary
   ```
   
2. **L₂ Regularization** (See [`Training.py`](Psi-NN-main/Module/Training.py))
   
   ```python
   loss += lambda_reg * torch.sum(weights ** 2)
   ```
   
3. **Hierarchical Clustering** (See [`structure_extractor.py`](Code/structure_extractor.py))
   ```python
   clusters = hierarchical_clustering(weights, n_clusters=3)
   ```

4. **Verilog-A Generation** (See [`verilog_generator.py`](Code/verilog_generator.py))
   
   ```verilog
   analog begin
       V(out) <+ tanh(w1*V(in) + b1);
   end
   ```

---

## 📊 Performance Benchmarks

### Training Time (NVIDIA RTX 4090)

| **Case Study** | **PINN Training** | **Distillation** | **Structure Extraction** | **Total** |
|----------------|-------------------|------------------|--------------------------|-----------|
| Burgers        | 120s              | 45s              | 15s                      | 180s      |
| Laplace        | 110s              | 40s              | 12s                      | 162s      |
| SNN XOR        | 95s               | 35s              | 10s                      | 140s      |
| Memristor      | 125s              | 48s              | 17s                      | 190s      |

### SPICE Simulation Overhead

| **Model Type** | **Simulation Time** | **Accuracy (vs PINN)** |
|----------------|---------------------|------------------------|
| Ψ-HDL (Verilog-A) | 0.5s             | 99.8%                  |
| LUT (1000 points) | 2.3s             | 98.5%                  |
| Original PINN  | N/A (not synthesizable) | 100% (baseline)   |

---

## 🔗 Related Publications

### Ψ-NN (Foundation)
- **Paper**: [Automatic network structure discovery of physics informed neural networks via knowledge distillation](https://doi.org/10.1038/s41467-025-64624-3)
- **Journal**: Nature Communications (2025)
- **Authors**: Liu et al.

### Ψ-HDL (This Work)
- **Paper**: *Ψ-HDL: Physics Structured-Informed Neural Networks for Hardware Description Language Generation*
- **Journal**: IEEE Access (Soon to be submitted)

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{Jurj2025PSI-HDL,
  title={Ψ-HDL: Physics Structured-Informed Neural Networks for Hardware Description Language Generation},
  author={Sorin Liviu Jurj},
  journal={IEEE Access},
  year={2025},
  note={Submitted}
}

@article{liu2025automatic,
  title={Automatic network structure discovery of physics informed neural networks via knowledge distillation},
  author={Liu, Ziti and Liu, Yang and Yan, Xunshi and Liu, Wen and Nie, Han and Guo, Shuaiqi and Zhang, Chen-an},
  journal={Nature Communications},
  volume={16},
  pages={9558},
  year={2025},
  doi={10.1038/s41467-025-64624-3}
}
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Areas for Contribution
- Additional case studies (other PDEs, circuits, devices)
- Performance optimizations
- Extended HDL backends (VHDL-AMS, SystemVerilog-AMS)
- GUI for Ψ-HDL pipeline
- Hardware synthesis benchmarks (FPGA/ASIC)

---

## 📝 License

This project is licensed under the Apache License 2.0. See the [`LICENSE`](LICENSE) file for details.

### Attribution

This work extends the [Ψ-NN framework](https://github.com/original-psi-nn) by Liu et al. (Nature Communications, 2025). The original Ψ-NN code is included in [`Psi-NN-main/`](Psi-NN-main/) directory under Apache 2.0 License.

---

## 🙏 Acknowledgments

- **Original Ψ-NN Authors**: Liu, Ziti; Liu, Yang; Yan, Xunshi; Liu, Wen; Nie, Han; Guo, Shuaiqi; Zhang, Chen-an

---

## 📅 Changelog

### Version 1.0.0 (2025-11-05)
- Initial release accompanying IEEE Access submission
- Four complete case studies: Burgers, Laplace, SNN XOR, Memristor
- Experimental validation suite: VTEAM comparison, cross-validation, noise robustness
- Automatic Verilog-A code generation
- SPICE validation testbenches
- Complete documentation and examples

---

## 🔮 Future Work

- [ ] SystemVerilog-AMS backend
- [ ] FPGA synthesis flow
- [ ] Real-time hardware deployment
- [ ] Multi-physics co-simulation
- [ ] GUI tool for non-programmers
- [ ] Cloud-based training service
- [ ] Extended device model library
