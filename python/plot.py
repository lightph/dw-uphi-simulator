import numpy as np
import subprocess
import os
from matplotlib import pyplot as plt


def sol_analitica(f):
    if f <= 1.0:
        return 0.0
    else:
        return np.sqrt(f**2 - 1)


executable_path = "./solveODPend"
output_file = "output/forzado/xt/run_03.csv"
if not os.path.exists(executable_path):
    print(f"Error: Executable not found at {os.path.abspath(executable_path)}")
while True:
    A = input()
    if A == "q":
        break
    B = input()
    if B == "q":
        break
    C = input()
    if C == "q":
        break

    result = subprocess.run(
        [executable_path, str(A), str(B), str(C)],
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

        t, x, v = np.genfromtxt(output_file, delimiter=",", unpack=True)

        fig, ax = plt.subplots(2, 1, tight_layout=True)
        ax[0].scatter(
            t,
            x,
            color="red",
            s=3.0,
            alpha=0.95,
            label=f"A: {float(A):.2f}\nB: {float(B):.2f}\nC: {float(C):.2f}",
        )
        ax[0].grid()

        ax[1].scatter(t, v, color="black", s=3.0)
        ax[1].grid()

        ax[0].legend()

        # plt.savefig("output/forzado/xt/comp.jpg", dpi=200)
        plt.show()
        print("done")
        plt.close()
