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

    # Calculate weighted average
    total_distance = sum([d[0] for d in nearest])
    if total_distance == 0:
        weighted_average = np.mean([d[1] for d in nearest])
    else:
        weighted_average = sum([(d[0] / total_distance) * d[1] for d in nearest])

    # For standard deviation, use only the response values of nearest neighbors
    responses = [d[1] for d in nearest]
    if len(responses) > 1:
        std_dev = statistics.stdev(responses)
    else:
        std_dev = 0.01  # Small default noise

    return np.random.normal(loc=weighted_average, scale=std_dev)
