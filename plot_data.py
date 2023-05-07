from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    path = Path.cwd() / "tsp" / "not_fucked_question_mark" / "delta"

    filenames = []
    fig, ax = plt.subplots()
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(path.name)
    for file in sorted(path.glob("*_best")):
        ax.plot(np.genfromtxt(file), label=file.name)
        filenames.append(file.name.removesuffix("_best"))
    ax.legend()

    for filename in filenames:
        best = np.genfromtxt(path / f"{filename}_best")
        avg = np.genfromtxt(path / f"{filename}_avg")

        fig, ax = plt.subplots()
        ax.plot(best, label="Best")
        ax.plot(avg, label="Average")
        ax.set_title(filename)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend()
    plt.show()
