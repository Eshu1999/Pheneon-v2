# Pheneon NPU Core v3 — High-Performance Edge AI Accelerator

An integrated, end-to-end hardware acceleration engine and software compiler stack designed for edge AI inference. Features a parameterized $16 \times 16$ systolic array architecture, vector activation processing unit, and ONNX graph lowering pipeline optimized for high-throughput matrix multiplication and spatial convolutions.

---

## Technical Highlights

* **Architecture**: $16 \times 16$ Weight-Stationary Systolic Processing Array.
* **Precision**: INT8 activation and weight inputs with full 32-bit signed integer accumulation (`INT8 x INT8 → INT32`).
* **Supported Operations**: Matrix Multiplication (GEMM), Multi-row execution, 2D Convolution (`im2col` lowering), non-linear activation functions (BYPASS, ReLU).
* **Software Toolchain**: Automated ONNX graph parser, integer quantizer, schedule lowerer, hex stimulus generator, and Verilator test harness.
* **Verification**: Verified across 20 automated regression configurations with 100% numerical match (`max |diff| = 0`) against NumPy floating-point/INT32 reference engines.

---

## Architecture Overview

```
                          +-------------------------+
                          |   ONNX Model / Graphs   |
                          +------------+------------+
                                       |
                                       v
                          +-------------------------+
                          |   Compiler Pipeline     |
                          | (Parser, Lower, Emit)   |
                          +------------+------------+
                                       |
                                       v
                          +-------------------------+
                          | Stimulus (.hex) Files   |
                          +------------+------------+
                                       |
                                       v
 +-------------------------------------------------------------------+
 |                      Pheneon Core v3 Hardware                     |
 |                                                                   |
 |   +--------------------+     +--------------------------------+   |
 |   |  Activation Buffer | --> |    16x16 Systolic Array PE     |   |
 |   +--------------------+     +---------------+----------------+   |
 |                                              |                    |
 |   +--------------------+                     v                    |
 |   |   Weight Buffer    | --> +--------------------------------+   |
 |   +--------------------+     |        Accumulator Bank        |   |
 |                              +---------------+----------------+   |
 |                                              |                    |
 |                                              v                    |
 |                              +--------------------------------+   |
 |                              | Vector Unit (BYPASS / ReLU)    |   |
 |                              +---------------+----------------+   |
 +----------------------------------------------|--------------------+
                                                v
                                     +--------------------+
                                     |  Output Stream     |
                                     +--------------------+
```

---

## Hardware Performance & Cycle Benchmarks

Evaluated on SystemVerilog RTL via Verilator 5.028 in WSL 1/Linux environments.

### 1. Single-Row $K$-Sweep Latency ($N=32$)

$$\text{Cycles}(K\_TILES) = 206 + (K\_TILES - 1) \times 165$$

| $K$ Dimension | $K\_TILES$ ($K / 32$) | Total Clock Cycles | Verification Result |
| :---: | :---: | :---: | :---: |
| **32** | 1 | 206 | **PASS** (`max |diff| = 0`) |
| **64** | 2 | 371 | **PASS** (`max |diff| = 0`) |
| **96** | 3 | 536 | **PASS** (`max |diff| = 0`) |
| **128** | 4 | 701 | **PASS** (`max |diff| = 0`) |

### 2. Multi-Row Matrix Scaling ($N=32$)

Weight buffers remain resident across rows to eliminate re-loading overhead:

| $M$ (Rows) | $K$ Dimension | Total Clock Cycles | Average Cycles / Row |
| :---: | :---: | :---: | :---: |
| **1** | 32 | 206 | 206.0 |
| **2** | 32 | 385 | 192.5 |
| **4** | 32 | 743 | 185.8 |
| **4** | 128 | 2,723 | 680.8 |

### 3. Spatial 2D Convolution (im2col Lowering)

* **Input Image**: $5 \times 5 \times 1$ INT8
* **Kernel Configuration**: $3 \times 3$, stride 1, padding 0 ($C_{out} = 32$)
* **Flattened Dimensions**: $ROWS = 9$, $K_{padded} = 32$, $N = 32$
* **Output Tensor**: $3 \times 3 \times 32$ (288 output values)
* **Execution Latency**: 1,638 clock cycles (**PASS**)

---

## Directory Structure

```
pheneon_core_v3/
├── rtl/                        # SystemVerilog RTL Source Files
│   ├── pe.sv                   # Processing Element (MAC block)
│   ├── systolic_array.sv       # 16x16 PE grid matrix processor
│   ├── accumulator_bank.sv     # 32-bit accumulation registers
│   ├── vector_unit.sv          # Post-processing / Activation unit
│   ├── activation_buffer.sv    # Activation streaming BRAM
│   ├── weight_buffer.sv        # Weight stationary BRAM
│   ├── axi_stream_if.sv        # AXI-Stream interface wrapper
│   ├── control_fsm.sv          # Central orchestration FSM
│   └── npu_top.sv              # Top-level NPU core design
│
├── software/                   # Python Compiler Toolchain
│   ├── onnx_parser.py          # Loads ONNX models and extracts weights
│   ├── graph_lower.py          # Lowers ONNX DAG to NPU operations
│   ├── im2col.py               # Image-to-column matrix transformer
│   ├── emit.py                 # Emits .hex memory stimulus files
│   └── test_reference.py       # NumPy gold-standard reference models
│
└── tests/                      # Simulation & Verification Suite
    ├── tb_npu_top.sv           # SystemVerilog testbench harness
    ├── test_end_to_end.py      # ONNX model end-to-end regression test
    ├── test_multirow.py       # Multi-row GEMM regression test
    ├── test_relu.py           # Activation pipeline regression test
    └── test_conv2d.py         # 2D Conv im2col regression test
```

---

## Getting Started

### Prerequisites

* Python 3.10+ with `numpy`, `onnx`, `matplotlib`
* Verilator 5.028 or newer
* GCC / G++ (C++17 compliant)

### Installation

```bash
git clone https://github.com/Eshu1999/pheneon_core_v3.git
cd pheneon_core_v3
python3 -m venv .venv
source .venv/bin/activate
pip install numpy onnx matplotlib
export PYTHONPATH=".:software"
```

### Running the Complete Test Suite

Run the automated 20-test case regression suite:

```bash
# 1. K-Dimension Sweep
for K in 32 64 96 128; do
    python3 tests/test_end_to_end.py --K $K
done

# 2. Multi-Row Matrix Sweep
for ROWS in 1 2 4; do
    for K in 32 64 128; do
        python3 tests/test_multirow.py --rows $ROWS --K $K
    done
done

# 3. Activation Pipeline Sweep
for ROWS in 1 2 4; do
    for K in 32 64; do
        python3 tests/test_relu.py --rows $ROWS --K $K
    done
done

# 4. Conv2D im2col Test
python3 tests/test_conv2d.py
```

---

## Licensing & Commercial Enquiries

Pheneon NPU Core v3 is available as a customizable IP block for edge AI applications, custom silicon, and FPGA implementations. For commercial licensing terms, integration consulting, or datasheets, contact:

* **Contact**: Arjan Anant
* **Email**: [arjananant2@gmail.com](mailto:arjananant2@gmail.com)
* **GitHub**: [@Eshu1999](https://github.com/Eshu1999)