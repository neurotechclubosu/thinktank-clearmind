def DistanceFunc(selectedPoint, pointArray, responseArray, nNearest,
                 metric="euclidean", rng=None):
    """Inverse-distance-weighted interpolation of ``responseArray`` at ``selectedPoint``.

    The value at a virtual node is the inverse-distance weighted mean of its
    ``nNearest`` electrode/node neighbours, plus Gaussian jitter whose SD is the
    sample SD of those neighbours' values (a visual-density device: it is applied
    only to the *rendered* virtual nodes, never to the electrode-level data that
    enter the statistics).

    v1.1.0
    ------
    * ``metric``: "euclidean" (default; what the Methods text always described) or
      "manhattan" (what v1.0.0 computed).
    * ``rng``: a ``numpy.random.Generator`` for reproducible rendering (default:
      NumPy's global state, as before).
    * Weights are inverse-distance (nearer neighbours count for more) --
      corrected in PR #2; v1.0.0 weighted each neighbour by d / sum(d).
    """
    import numpy as np
    import statistics

    if metric not in ("euclidean", "manhattan"):
        raise ValueError("metric must be 'euclidean' or 'manhattan'")
    sel = np.asarray(selectedPoint, dtype=float)

    distances = []
    for point, response in zip(pointArray, responseArray):
        diff = np.abs(sel - np.asarray(point, dtype=float))
        dist = float(diff.sum()) if metric == "manhattan" else float(np.sqrt((diff ** 2).sum()))
        distances.append((dist, response))

    distances.sort(key=lambda x: x[0])
    nearest = distances[:nNearest]

    if any(d[0] == 0 for d in nearest):
        # Selected point coincides with an electrode; take the exact value(s).
        weighted_average = np.mean([d[1] for d in nearest if d[0] == 0])
    else:
        inv = [1.0 / d[0] for d in nearest]
        total_inv = sum(inv)
        weighted_average = sum(w / total_inv * d[1] for w, d in zip(inv, nearest))

    responses = [d[1] for d in nearest]
    std_dev = statistics.stdev(responses) if len(responses) > 1 else 0.01

    draw = np.random.normal if rng is None else rng.normal
    return draw(loc=weighted_average, scale=std_dev)
