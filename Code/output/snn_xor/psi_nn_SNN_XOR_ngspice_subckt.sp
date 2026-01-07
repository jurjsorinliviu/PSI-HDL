* ngspice behavioral subcircuit (B-sources)
* Generated from model: SNN_XOR
.subckt psi_nn_snn_xor in0 in1 out0 vdd vss

* Activation (fallback-friendly): use tanh() if supported by ngspice, else replace with rational approx
.func tanh_psi(x) { tanh(x) }

B_L0_n0 L0_n0 0 V = tanh_psi((7.917837239802e-03 + -1.643253207207e+00*V(in0) + 1.396323919296e+00*V(in1)))
B_L0_n1 L0_n1 0 V = tanh_psi((-6.046491907910e-04 + 1.497709870338e+00*V(in0) + -1.741643667221e+00*V(in1)))
B_L1_n0 L1_n0 0 V = (5.327655747533e-02 + 1.949330329895e+00*V(L0_n0) + 1.837617039680e+00*V(L0_n1))
B_out_0 out0 0 V = V(L1_n0)

.ends psi_nn_snn_xor
