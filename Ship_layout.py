import numpy as np
import random

def create_ship_layout(size=30):
    grid = np.zeros((size, size), dtype=int)
    
    grid[0, :] = grid[:, 0] = grid[size-1, :] = grid[:, size-1] = 0
    
    start_x, start_y = random.randint(1, size-2), random.randint(1, size-2)
    grid[start_x, start_y] = 1

    while True:
        candidates = []
        for x in range(1, size-1):
            for y in range(1, size-1):
                if grid[x, y] == 0:
                    open_neighbors = sum([grid[x+dx, y+dy] for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]])
                    if open_neighbors == 1:
                        candidates.append((x, y))

        if not candidates:
            break

        new_x, new_y = random.choice(candidates)
        grid[new_x, new_y] = 1

    dead_ends = [(x, y) for x in range(1, size-1) for y in range(1, size-1) 
                 if grid[x, y] == 1 and sum([grid[x+dx, y+dy] for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]) == 1]
    
    random.shuffle(dead_ends)
    for x, y in dead_ends[:len(dead_ends) // 2]:
        closed_neighbors = [(x+dx, y+dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)] if grid[x+dx, y+dy] == 0]
        if closed_neighbors:
            nx, ny = random.choice(closed_neighbors)
            grid[nx, ny] = 1

    return grid

ship_layout = create_ship_layout()
