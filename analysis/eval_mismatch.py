import numpy as np
from parser import parse_experiment

def load_step_times(path):
    exp = parse_experiment(path, "test", runs_to_evaluate=5)
    return exp.trimmed_mean_step_time, exp.trimmed_std_step_time

t_nat, s_nat = load_step_times("./profilings/container/mismatch_native.txt")
t_v3, s_v3   = load_step_times("./profilings/container/mismatch_v3.txt")
t_cnt, s_cnt = load_step_times("./profilings/container/mismatch_container.txt")

delta_isa = ((t_v3 - t_nat) / t_nat) * 100.0
delta_cnt = ((t_cnt - t_v3) / t_v3) * 100.0
total_gap = ((t_cnt - t_nat) / t_nat) * 100.0

print(f"{'Target':<25} | {'Step Time':<18} | {'Delta vs Native':<18}")
print("-" * 68)
print(f"{'1. Native AVX-512':<25} | {t_nat*1000:7.2f} +/- {s_nat*1000:5.2f} ms | {'Reference':<18}")
print(f"{'2. Native AVX2 (v3)':<25} | {t_v3*1000:7.2f} +/- {s_v3*1000:5.2f} ms | {f'+{delta_isa:.2f}% (ISA gap)':<18}")
print(f"{'3. Container AVX2':<25} | {t_cnt*1000:7.2f} +/- {s_cnt*1000:5.2f} ms | {f'+{delta_cnt:.2f}% (Cont. ovh)':<18}")
print("-" * 68)
print(f"Total Combined Overhead: +{total_gap:.2f}%")