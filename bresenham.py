
def bresenham_path(start, end, shape):
    """
    Bresenham's Line Algorithm
    Produces an numpy array
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end

    x1 = max(0, min(round(x1), shape[0]-1))
    y1 = max(0, min(round(y1), shape[1]-1))
    x2 = max(0, min(round(x2), shape[0]-1))
    y2 = max(0, min(round(y2), shape[1]-1))

    dx = x2 - x1
    dy = y2 - y1

    # Prepare output array
    path = []

    if (start == end):
        return path

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    for x in range(x1, x2 + 1):
        if is_steep:
            path.append([y, x])
        else:
            path.append([x, y])
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    return path
