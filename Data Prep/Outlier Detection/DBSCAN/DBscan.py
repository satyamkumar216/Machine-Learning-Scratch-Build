import math

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def get_neighbours(data, idx, epsilon):
    neighbours = []
    for i in range(len(data)):
        if i != idx and euclidean_distance(data[idx], data[i]) <= epsilon:
            neighbours.append(i)
    return neighbours

def dbscan(data, epsilon, min_points):
    n = len(data)
    labels = [-1] * n
    visited = [False] * n
    is_core = [False] * n
    cluster_id = -1

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True
        neighbours = get_neighbours(data, i, epsilon)

        if len(neighbours) < min_points - 1:
            continue

        is_core[i] = True
        cluster_id += 1
        labels[i] = cluster_id
        queue = list(neighbours)

        while queue:
            j = queue.pop(0)

            if not visited[j]:
                visited[j] = True
                j_neighbours = get_neighbours(data, j, epsilon)

                if len(j_neighbours) >= min_points - 1:
                    is_core[j] = True
                    queue.extend(j_neighbours)

                if labels[j] == -1:
                    labels[j] = cluster_id
            else:
                if labels[j] == -1:
                    labels[j] = cluster_id

    return labels, is_core

def classify_points(labels, is_core):
    result = []
    for i, (lbl, core) in enumerate(zip(labels, is_core)):
        if lbl == -1:
            point_type = "NOISE"
        elif core:
            point_type = f"CORE  (cluster {lbl})"
        else:
            point_type = f"BORDER (cluster {lbl})"
        result.append((i, point_type))
    return result
