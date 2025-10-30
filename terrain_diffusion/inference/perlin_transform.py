import numpy as np

def build_quantiles(values, n_quantiles=32, eps=1e-4):
    """
    Build quantile values for a distribution.

    Parameters
    ----------
    values : array-like
        Samples from the distribution.
    n_quantiles : int
        Number of quantile knots to use.
        Larger -> smoother, but a bit more setup cost.
    eps : float
        Avoids extreme tails (0 and 1) where empirical quantiles are unstable.

    Returns
    -------
    quantiles : np.ndarray
        The quantile values (strictly increasing).
    """
    v = np.asarray(values).ravel()

    # Drop NaNs if present
    v = v[~np.isnan(v)]

    # Quantile grid (avoid exact 0/1 for stability)
    q = np.linspace(eps, 1.0 - eps, n_quantiles)

    # Empirical quantile function
    v_q = np.quantile(v, q)

    # Ensure strictly increasing (np.interp requires increasing; ties can occur with discrete/flat regions)
    diffs = np.diff(v_q)
    min_diff = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else 1e-10
    for i in range(1, len(v_q)):
        if v_q[i] <= v_q[i-1]:
            v_q[i] = v_q[i-1] + min_diff * 0.1

    return v_q

def transform_perlin(perlin_map, source_quantiles, target_quantiles):
    if len(source_quantiles) != len(target_quantiles):
        raise ValueError("Source and target quantiles must have the same length")
    return np.interp(perlin_map, source_quantiles, target_quantiles, left=target_quantiles[0], right=target_quantiles[-1])