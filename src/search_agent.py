# Task 1: warehouse robot route planning with BFS, UCS and A*.
import heapq
import itertools
import json
from collections import deque

from common import REPORTS

ROWS, COLS = 12, 20
START, GOAL = (0, 0), (11, 19)


def build_grid():
    grid = [['.' for _ in range(COLS)] for _ in range(ROWS)]
    for r in (2, 3, 6, 7):            # two double rows of shelving
        for c in range(2, 18):        # aisles at columns 0-1 and 18-19
            grid[r][c] = '#'
    return grid


def neighbours(grid, node):
    r, c = node
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] != '#':
            yield (nr, nc)


def reconstruct(parent, node):
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def bfs(grid, start, goal):
    frontier = deque([start])
    parent = {start: None}
    expanded = 0
    while frontier:
        node = frontier.popleft()
        expanded += 1
        if node == goal:
            path = reconstruct(parent, node)
            return path, len(path) - 1, expanded
        for nb in neighbours(grid, node):
            if nb not in parent:
                parent[nb] = node
                frontier.append(nb)
    return None, None, expanded


def zero_heuristic(a, b):
    return 0


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal, h):
    tie = itertools.count()
    h0 = h(start, goal)
    # Priority = (f, h, insertion order): prefer low f, then nodes closer to goal
    frontier = [(h0, h0, next(tie), start)]
    g = {start: 0}
    parent = {start: None}
    closed = set()
    expanded = 0
    while frontier:
        _f, _h, _t, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node == goal:
            return reconstruct(parent, node), g[node], expanded
        for nb in neighbours(grid, node):
            ng = g[node] + 1
            if ng < g.get(nb, float('inf')):
                g[nb] = ng
                parent[nb] = node
                hn = h(nb, goal)
                heapq.heappush(frontier, (ng + hn, hn, next(tie), nb))
    return None, None, expanded


def render(grid, path):
    canvas = [row[:] for row in grid]
    for r, c in path:
        canvas[r][c] = '*'
    canvas[START[0]][START[1]] = 'S'
    canvas[GOAL[0]][GOAL[1]] = 'G'
    return '\n'.join(''.join(row) for row in canvas)


if __name__ == '__main__':
    grid = build_grid()
    runs = {
        'bfs': ('BFS', lambda: bfs(grid, START, GOAL)),
        'ucs': ('UCS (h=0)', lambda: astar(grid, START, GOAL, zero_heuristic)),
        'astar': ('A* (Manhattan)', lambda: astar(grid, START, GOAL, manhattan)),
    }
    results = {}
    for key, (label, fn) in runs.items():
        path, cost, expanded = fn()
        results[key] = {'cost': cost, 'expanded': expanded}
        print(f'{label:<15} path cost: {cost:>3}  nodes expanded: {expanded:>4}')
    print()
    print(render(grid, path))
    (REPORTS / 'search_results.json').write_text(json.dumps(results, indent=2))
