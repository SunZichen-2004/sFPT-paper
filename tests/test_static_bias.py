import numpy as np
from fes_time import mean_bias_weights


def test_mean_bias_matches_average_of_cumulative_deposition_histories():
    # Mean of the cumulative biases after depositions6,7,8,9,10.
    coefficients = np.arange(1, 11, dtype=float)
    actual = np.dot(mean_bias_weights(10, 6), coefficients)
    expected = np.cumsum(coefficients)[5:].mean()
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(mean_bias_weights(10, 10), np.ones(10))
