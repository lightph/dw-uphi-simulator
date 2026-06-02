import numpy as np
import subprocess
import os
from matplotlib import pyplot as plt


def sol_analitica(f):
    if f <= 1.0:
        return 0.0
    else:
        return np.sqrt(f**2 - 1)


executable_path = "./solveODPendRange"
output_file = "output/forzado/vf/run_01.csv"
if not os.path.exists(executable_path):
    print(f"Error: Executable not found at {os.path.abspath(executable_path)}")
while True:
    fmin = input("Amin\n")
    if fmin == "q":
        break
    fmax = input("Amax\n")
    if fmax == "q":
        break
    n = input("n\n")
    if int(n) <= 0:
        break

    m = input("m (integer)\n")
    if m == "q":
        break
    m = int(m)

    C_values = []
    for i in range(m):
        val = input(f"C value {i + 1}\n")
        if val == "q":
            break
        C_values.append(float(val))

    if len(C_values) != m:
        break

    fmin = float(fmin)
    fmax = float(fmax)
    B = 1.0

    fig, ax = plt.subplots(1, 1, tight_layout=True)

    for C in C_values:
        result = subprocess.run(
            [executable_path, str(B), str(C), str(fmin), str(fmax), str(n)],
            capture_output=True,
            text=True,
            cwd="build",
        )
        if result.returncode != 0:
            print(f"Return Code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        else:
            print(f"Return Code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

            t, v = np.genfromtxt(output_file, delimiter=",", unpack=True)

            ax.scatter(
                t,
                v,
                s=3.0,
                alpha=0.95,
                label=f"C: {C:.2f}",
            )

    ax.grid()
    ax.legend()
    print("done")
    plt.show()
    plt.close()
