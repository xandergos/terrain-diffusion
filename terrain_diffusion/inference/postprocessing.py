import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def d8_flow(z, tol=1e-3):
    z = np.asarray(z)
    H, W = z.shape
    dy = np.array([-1,  1,  0,  0, -1, -1,  1,  1], dtype=int)
    dx = np.array([ 0,  0, -1,  1, -1,  1, -1,  1], dtype=int)
    dist = np.array([1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)], dtype=z.dtype)

    zpad = np.pad(z, 1, mode='edge')
    nbrs = np.stack([zpad[1+dy[k]:1+dy[k]+H, 1+dx[k]:1+dx[k]+W] for k in range(8)], axis=0)
    slopes = (z[None] - nbrs) / dist[:, None, None]  # positive = downhill
    slopes[slopes < tol] = -np.inf

    # Ocean handling
    # - Centers that are NaN or <= 0 are sinks (ocean)
    # - Neighbors that are NaN or <= 0 act as ocean sinks: prefer routing into them
    center_ocean = np.isnan(z) | (z <= 0)
    neighbor_ocean = np.isnan(nbrs) | (nbrs <= 0)

    # Prepare two slope tensors:
    # 1) prefer_nan: prefers routing into NaN neighbors (treat as +inf slope)
    # 2) ignore_nan: ignores NaN neighbors (treat as -inf) to decide internal sinks
    prefer_ocean = slopes.copy()
    prefer_ocean[:, center_ocean] = -np.inf
    prefer_ocean[neighbor_ocean & (~center_ocean[None])] = np.inf

    ignore_ocean = slopes.copy()
    ignore_ocean[:, center_ocean] = -np.inf
    ignore_ocean[neighbor_ocean] = -np.inf

    # Chosen directions prefer draining into NaN neighbors (coast/ocean)
    kmax = np.argmax(prefer_ocean, axis=0)
    max_slope_prefer = np.take_along_axis(prefer_ocean, kmax[None], axis=0)[0]

    # is_sink: true only if center is NaN OR there is no downhill route ignoring NaNs
    max_slope_ignore = np.take_along_axis(ignore_ocean, np.argmax(ignore_ocean, axis=0)[None], axis=0)[0]
    has_ocean_neighbor = np.any(neighbor_ocean, axis=0)
    is_sink = center_ocean | ((~has_ocean_neighbor) & (~np.isfinite(max_slope_ignore)))

    rr = np.clip(np.arange(H)[:, None] + dy[kmax], 0, H - 1)
    cc = np.clip(np.arange(W)[None, :] + dx[kmax], 0, W - 1)
    return rr, cc, is_sink, kmax

def flow_accumulation(z, rr, cc, is_sink):
    H, W = z.shape
    invalid = np.isnan(z) | (z <= 0)
    # Initialize with ones for valid cells only
    A = np.zeros((H, W), dtype=np.float32)
    A[~invalid] = 1.0

    # Process cells from high to low elevation, ignoring NaNs
    flat_idx = np.flatnonzero(~invalid)
    if flat_idx.size:
        vals = z.ravel()[flat_idx]
        order = flat_idx[np.argsort(vals)[::-1]]
        r, c = order // W, order % W
        for i, j in zip(r, c):
            if not is_sink[i, j]:
                ti, tj = rr[i, j], cc[i, j]
                if not invalid[ti, tj]:
                    A[ti, tj] += A[i, j]
    return A

def plot_flow_indicator(z, max_pool_kernel=1):
    z = np.asarray(z)
    rr, cc, is_sink, kmax = d8_flow(z)
    A = flow_accumulation(z, rr, cc, is_sink)
    # Ensure ocean (NaN or <= 0) remain non-contributing in the indicator
    invalid = np.isnan(z) | (z <= 0)
    A[invalid] = 0.0
    
    # Perform max pooling on A, configurable by max_pool_kernel
    if max_pool_kernel > 1:
        # Downsampling max pool (non-overlapping, stride = kernel size)
        new_H = A.shape[0] // max_pool_kernel
        new_W = A.shape[1] // max_pool_kernel
        A = A[:new_H * max_pool_kernel, :new_W * max_pool_kernel]
        A = A.reshape(new_H, max_pool_kernel, new_W, max_pool_kernel)
        A = A.max(axis=(1, 3))
        
    return np.log1p(A)

def smooth_river_bumps(
    height,
    slope_thresh=50,     # below this, considered "flat"
    smooth_strength=0.3,    # fraction of smoothing applied
    iterations=3            # few iterations are enough
):
    """
    Removes small upslope bumps in rivers while preserving steep slopes.
    """
    h = height.copy().astype(np.float32)
    nan_mask = np.isnan(h)

    for _ in range(iterations):
        # Compute gradients on a NaN-filled-safe surface (treat NaNs as 0 for ops)
        h_safe = np.where(nan_mask, 0.0, h)
        grad_y, grad_x = np.gradient(h_safe)
        slope = np.sqrt(grad_x**2 + grad_y**2)

        # Build Laplacian ignoring NaN neighbors (4-neighbor)
        valid = ~nan_mask
        up_valid = np.roll(valid, 1, 0)
        dn_valid = np.roll(valid, -1, 0)
        lf_valid = np.roll(valid, 1, 1)
        rt_valid = np.roll(valid, -1, 1)

        up = np.where(up_valid, np.roll(h_safe, 1, 0), 0.0)
        dn = np.where(dn_valid, np.roll(h_safe, -1, 0), 0.0)
        lf = np.where(lf_valid, np.roll(h_safe, 1, 1), 0.0)
        rt = np.where(rt_valid, np.roll(h_safe, -1, 1), 0.0)

        neighbor_sum = up + dn + lf + rt
        neighbor_cnt = (
            up_valid.astype(np.float32)
            + dn_valid.astype(np.float32)
            + lf_valid.astype(np.float32)
            + rt_valid.astype(np.float32)
        )
        laplace = neighbor_sum - neighbor_cnt * h_safe
        laplace[nan_mask] = 0.0

        # Weight by (low slope) regions only; do not update NaN cells
        w = np.exp(- (slope / slope_thresh) ** 2)
        w[nan_mask] = 0.0

        # Apply selective smoothing, preserve NaNs
        h += smooth_strength * w * laplace
        h[nan_mask] = np.nan

    return h

import heapq

def fill_depressions_priority_flood(
    height: np.ndarray,
    epsilon: float = 1e-3,      # tiny gradient injected across flats
    max_raise: float | None = None,  # H_max: maximum allowed basin fill depth
    connectivity: int = 8,      # 4 or 8
    in_place: bool = False,
    nodata: float | None = None # treat NaNs (or this value) as barriers
) -> np.ndarray:
    """
    Priority-Flood selective depression fill.
    Fills pits only up to a maximum basin depth H_max ("max_raise").
    If the required fill depth exceeds H_max, the basin is left as a true
    depression (no further raising). Epsilon ensures drainage across flats.

    Args:
        height: 2D elevation array.
        epsilon: Small increment to ensure drainage across flats.
        connectivity: 4 or 8-neighbor graph.
        in_place: Modify input array in place if True.
        nodata: If provided, cells equal to this value are treated as invalid.
                NaNs are always treated as invalid.

    Returns:
        Filled elevation array (same shape).
    """
    h = height if in_place else height.copy()
    h = h.astype(np.float32, copy=False)
    # Preserve original heights; needed to track basin minima
    base = height.astype(np.float32, copy=False).copy()
    H, W = h.shape

    if nodata is None:
        ocean = np.isnan(h) | (h <= 0)
    else:
        ocean = np.isnan(h) | (h <= 0) | (h == nodata)
    invalid = ocean

    visited = np.zeros((H, W), dtype=bool)
    # Track the minimum original elevation encountered along the flood path
    # to each cell; used to measure basin fill depth relative to its minimum
    basin_min = np.full((H, W), np.inf, dtype=np.float32)
    heap: list[tuple[float, int, int]] = []

    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Seed with valid outer border cells
    for i in range(H):
        for j in (0, W - 1):
            if not invalid[i, j] and not visited[i, j]:
                heapq.heappush(heap, (h[i, j], i, j))
                visited[i, j] = True
                basin_min[i, j] = base[i, j]
    for j in range(W):
        for i in (0, H - 1):
            if not invalid[i, j] and not visited[i, j]:
                heapq.heappush(heap, (h[i, j], i, j))
                visited[i, j] = True
                basin_min[i, j] = base[i, j]

    # Also seed coast-adjacent valid cells (adjacent to ocean) as outlets
    if connectivity == 4:
        nbrs_seed = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs_seed = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(H):
        for c in range(W):
            if invalid[r, c] or visited[r, c]:
                continue
            # If any neighbor is ocean, treat this as coastal outlet seed
            coastal = False
            for dr, dc in nbrs_seed:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if ocean[nr, nc]:
                    coastal = True
                    break
            if coastal:
                elev_seed = max(h[r, c], 0.0)
                heapq.heappush(heap, (elev_seed, r, c))
                visited[r, c] = True
                basin_min[r, c] = base[r, c]

    # Priority-Flood
    while heap:
        elev, r, c = heapq.heappop(heap)
        bm_cur = basin_min[r, c]
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if visited[nr, nc] or invalid[nr, nc]:
                continue

            ne = h[nr, nc]
            # Propagate basin minimum along the flood path
            bm_next = bm_cur if base[nr, nc] >= bm_cur else base[nr, nc]
            if ne <= elev:
                # Selective fill: stop raising if basin depth exceeds H_max
                if (max_raise is not None) and (elev - bm_cur >= max_raise):
                    heapq.heappush(heap, (ne, nr, nc))
                else:
                    new_e = elev + epsilon
                    # Ensure we never exceed the allowed basin depth
                    if max_raise is not None:
                        max_level = bm_cur + max_raise
                        if new_e > max_level:
                            new_e = max_level
                    if new_e > ne:
                        h[nr, nc] = new_e
                    heapq.heappush(heap, (h[nr, nc], nr, nc))
            else:
                heapq.heappush(heap, (ne, nr, nc))
            visited[nr, nc] = True
            basin_min[nr, nc] = bm_next

    return h

def local_baseline_temperature_torch(
    T: torch.Tensor,
    e: torch.Tensor,
    win: int = 3,
    beta_clip=(-0.012, 0.0),   # °C per meter
    fallback_beta=-0.0065,     # °C per meter
    eps=1e-6,
    fallback_threshold=0.3
):
    """
    Estimate local sea-level baseline temperature using a windowed regression.

    Args:
        T, e: 2D tensors (H, W) or batched (B, 1, H, W) of temperature [°C] and elevation [m].
        win: window size (odd integer)
        beta_clip: allowed lapse-rate range (°C/m)
        fallback_beta: used if local elevation variance ~ 0
        eps: small constant for stability

    Returns:
        T_sea: local baseline temperature map (B, 1, H-(win-1), W-(win-1))
        beta:   local lapse-rate map      (same shape)
    """
    if T.ndim == 2:
        T = T.unsqueeze(0).unsqueeze(0)
        e = e.unsqueeze(0).unsqueeze(0)
    elif T.ndim == 3:
        T = T.unsqueeze(1)
        e = e.unsqueeze(1)

    # Land mask (1 = land, 0 = ocean)
    w = (e > 0).float()

    # Compute weighted means with valid convolution (no padding)
    def wavg(x):
        num = F.avg_pool2d(x * w, win, stride=1, padding=0)
        den = F.avg_pool2d(w,      win, stride=1, padding=0)
        return num / (den + eps), den

    mu_T, sum_w = wavg(T)
    mu_e, _ = wavg(e)
    mu_e2, _ = wavg(e * e)
    mu_eT, _ = wavg(e * T)

    var_e  = mu_e2 - mu_e**2
    cov_eT = mu_eT - mu_e * mu_T

    # Local slope β (°C per meter)
    beta = cov_eT / (var_e + eps)

    # Flat or water-dominated windows → fallback β
    invalid = (var_e < 1.0) | (sum_w < fallback_threshold)  # <30% land
    beta = torch.where(invalid, torch.tensor(fallback_beta, device=beta.device), beta)
    beta = torch.clamp(beta, beta_clip[0], beta_clip[1])

    # Sea-level baseline using raw T and e (no averaging); crop to valid region
    pad = (win - 1) // 2
    T_c = T[:, :, pad:-pad, pad:-pad]
    e_c = e[:, :, pad:-pad, pad:-pad]

    T_sea = T_c - beta * e_c

    return T_sea.squeeze(1), beta.squeeze(1)