import numpy as np

from src.univariate_garch import gjr_persistence


def test_gjr_persistence() -> None:
    result = gjr_persistence(
        alpha=0.05,
        gamma=0.10,
        beta=0.85,
    )

    assert np.isclose(
        result,
        0.95,
    )