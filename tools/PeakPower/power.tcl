# read_vcd_activities gcd
read_liberty sky130hd_tt.lib.gz
read_verilog gcd_sky130hd.v
link_design gcd
read_sdc gcd_sky130hd.sdc
read_spef gcd_sky130hd.spef

source input_trace
set_pin_activity_and_duty

report_activity_annotation
report_power > output_power