# Clock definition — target 200 MHz (5 ns period)
create_clock -name clk -period 5.000 [get_ports clk]

# Input/output delay
set_input_delay  -clock clk 1.0 [all_inputs]
set_output_delay -clock clk 1.0 [all_outputs]

# ---- Zero-logic register-to-register paths (per plan §4) ----
# Ping-pong buffer swap and pipeline-stage boundaries can be hold-binding.
# Constrain them explicitly so the tool does not auto-insert delay that
# breaks setup elsewhere.

set_max_delay -datapath_only 0.500 \
    -from [get_cells -hier -filter {NAME =~ *u_pe*act_r_reg*}] \
    -to   [get_cells -hier -filter {NAME =~ *u_pe*act_out*}]

set_max_delay -datapath_only 0.500 \
    -from [get_cells -hier -filter {NAME =~ *u_acc*acc_reg*}] \
    -to   [get_cells -hier -filter {NAME =~ *u_vec*bias_add_reg*}]

# ---- False paths ----
# Reset is asynchronous; no timing requirement.
set_false_path -from [get_ports rst_n]

# ---- Clock uncertainty ----
set_clock_uncertainty 0.100 [get_clocks clk]