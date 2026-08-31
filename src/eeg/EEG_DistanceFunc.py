def DistanceFunc(selectedPoint, pointArray, responseArray, nNearest):
    import numpy as np
    import statistics

    # Track distances and associated responses
    distances = []
    for point, response in zip(pointArray, responseArray):
        dist = abs(selectedPoint[0] - point[0]) + abs(selectedPoint[1] - point[1]) + abs(selectedPoint[2] - point[2])
        distances.append((dist, response))

    # Sort and select nearest n
    distances.sort(key=lambda x: x[0])
    nearest = distances[:nNearest]

    # Inverse-distance weighting: nearer neighbours must count for more. The
    # previous form weighted each neighbour by d / sum(d), which gave the most
    # distant neighbour the largest share.
    if any(d[0] == 0 for d in nearest):
        # Selected point coincides with an electrode; take the exact value(s).
        weighted_average = np.mean([d[1] for d in nearest if d[0] == 0])
    else:
        inv = [1.0 / d[0] for d in nearest]
        total_inv = sum(inv)
        weighted_average = sum(w / total_inv * d[1] for w, d in zip(inv, nearest))

    # For standard deviation, use only the response values of nearest neighbors
    responses = [d[1] for d in nearest]
    if len(responses) > 1:
        std_dev = statistics.stdev(responses)
    else:
        std_dev = 0.01  # Small default noise

    return np.random.normal(loc=weighted_average, scale=std_dev)
