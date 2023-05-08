from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    path = Path.cwd() / "schwefel" / "gauss"

    filenames = []
    plt.subplots()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(path.name)
    for file in sorted(path.glob("*_best.txt")):
        plt.plot(np.genfromtxt(file), label=file.name)
        filenames.append(file.name.removesuffix("_best.txt"))
    plt.legend()

    for filename in filenames:
        best = np.genfromtxt(path / f"{filename}_best.txt")
        avg = np.genfromtxt(path / f"{filename}_avg.txt")

        fig, ax = plt.subplots()
        ax.plot(best, label="Best")
        ax.plot(avg, label="Average")
        ax.set_title(filename)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend()
    plt.show()
