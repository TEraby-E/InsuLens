"""Deprecated dataset-module command wrapper.

Use :mod:`insulens_perception.insulator_dataset`. Historical class-count
constants and builder functions are intentionally not re-exported, preventing
stale checkpoint and label-index dependencies.
"""

from .insulator_dataset import main


if __name__ == "__main__":
    main()
