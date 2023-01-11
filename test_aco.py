import control
import aco_pid_tuning as aco

s = control.TransferFunction.s
g = 1 / (s * (s+1) * (s+5))

params = aco.aco_pid_tune(g, n_iterations=5)

print((params["p"], params["i"], params["d"]))
print()
