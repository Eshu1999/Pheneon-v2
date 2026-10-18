# NPU Compiler Specification — Phase 1

## Hardware Under Test

- Array: 32 × 32 weight-stationary systolic
- Precision: INT8 multiply, INT32 accumulate
- K-tile size: 32 (matches array height)
- K-loop: FSM accumulates across K-tiles in `accumulator_bank.sv`
- Max K: 128 (4 K-tiles) at current buffer depth
- Output: INT32 per column, optional vector-unit activation

## Supported Operators (Phase 1)

- **Gemm**: `Y = A @ B` with `alpha=1`, `beta=0`, `transA=0`, `transB=0`
  - A shape: `(M, K)` with `M = 1`
  - B shape: `(K, N)` with `N = 32`
  - Y shape: `(1, 32)`

## Data Types

- Inputs: INT8
- Accumulator: INT32
- Output: INT32 (before vector unit)

## Constraints

- `M` must be 1 (single-vector matmul). Batch support is Phase 2.
- `N` must be 32. Multi-tile N is Phase 2.
- `K` must be a multiple of 32. Current max 128.
- Weights must be static (initializers, not runtime inputs).

## File Formats

- `golden_A.hex`: K lines, 2 hex chars each (INT8 two's complement)
- `golden_B.hex`: K×32 lines, 2 hex chars each (row-major)
- `golden_C.hex`: 32 lines, 8 hex chars each (INT32 two's complement)

## AXI Register Map (from `npu_axi_wrapper.sv`)

- `slv_reg0[0]`: `shift_en` — pipeline advance
- `slv_reg0[1]`: `swap_buffers` — double-buffer swap pulse (self-clearing)
- `slv_reg1[2:0]`: `mode` — 0=INT8, 1=INT4, 2=FP16, 3=FP8_E4M3, 4=FP8_E5M2
- `slv_reg1[10:3]`: `act_fused` — activation type
- `slv_reg1[12:11]`: `act_mode` — 0=Bypass, 1=ReLU, 2=LeakyReLU, 3=GELU
- `slv_reg2[15:0]`: `scale_factor` — Q15 output scale
- `slv_reg2[21:16]`: `shift_amount` — output right-shift

## Known Limitations

- No Conv2D (Phase 2)
- No element-wise ops (Phase 2)
- No dynamic weight loading