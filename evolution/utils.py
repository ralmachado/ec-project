import random
from pathlib import Path


def gen_seeds(n: int, domain: list[int] | None = None) -> list[int]:
    if domain is None:
        domain = [1_000, 100_000]
    return [random.randrange(*domain) for _ in range(n)]


def save_seeds(file: str | Path, n: int, domain: list[int] | None = None) -> None:
    seeds = gen_seeds(n, domain)
    if not isinstance(file, Path):
        file = Path(file)
    with file.open('w') as f:
        f.write('\n'.join(map(str, seeds)))


def read_seeds(file: str | Path) -> list[int]:
    if not isinstance(file, Path):
        file = Path(file)
    with file.open('r') as f:
        seeds = list(map(int, f.readlines()))
    return seeds
