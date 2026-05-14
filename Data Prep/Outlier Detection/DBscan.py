import math

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def get_neighbours(data, idx, epsilon):
    neighbours = []

    for i in range(len(data)):
        if i != idx and euclidean_distance(data[idx], data[i]) <= epsilon:
            neighbours.append(i)
    return neighbours # Return the indices of the neighboring points within epsilon distance


def dbscan(data, epsilon, min_points):
    n = len(data)

    # ── Phase 1: mark EVERYTHING as noise, nothing visited ──────────────────
    labels  = [-1] * n        # -1 = NOISE  (every point starts here)
    visited = [False] * n
    is_core = [False] * n

    cluster_id = -1

    # ── Phase 2: visit each point ────────────────────────────────────────────
    for i in range(n):

        if visited[i]:
            continue

        visited[i] = True
        neighbours = get_neighbours(data, i, epsilon)

        # ── Not enough neighbours → stays NOISE (for now) ───────────────────
        # Do NOT label it border here! It might get rescued later.
        # Just leave it as labels[i] = -1 and move on.
        if len(neighbours) < min_points - 1:
            # labels[i] stays -1 (NOISE)
            # is_core[i] stays False
            continue   # <── that's it. no border label here.

        # ── Enough neighbours → this is a CORE point ─────────────────────────
        is_core[i] = True
        cluster_id += 1
        labels[i] = cluster_id      # core point joins its own cluster

        # Queue starts with all neighbours of this core point
        queue = list(neighbours)

        # ── Expand the cluster ────────────────────────────────────────────────
        while queue:
            j = queue.pop(0)

            # ── Case A: j not yet visited ─────────────────────────────────────
            if not visited[j]:
                visited[j] = True
                j_neighbours = get_neighbours(data, j, epsilon)

                if len(j_neighbours) >= min_points - 1:
                    # j is ALSO a core point → its neighbours join the queue
                    is_core[j] = True
                    queue.extend(j_neighbours)   # ← border points NEVER do this
                # else: j is a BORDER point → joins cluster but queue NOT extended

                # Either way, j joins the cluster if not already in one
                if labels[j] == -1:             # was noise, now rescued
                    labels[j] = cluster_id

            # ── Case B: j already visited ─────────────────────────────────────
            else:
                # j was visited before (possibly labelled noise when first seen)
                # If a core point is now reaching it, rescue it
                if labels[j] == -1:             # still noise → rescue as border
                    labels[j] = cluster_id      # ← border: joins cluster, not core

    # ── Phase 3: anything still labelled -1 is permanent noise ───────────────
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
