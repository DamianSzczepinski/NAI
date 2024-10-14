from PIL import Image
import numpy as np
import math
import random
from collections import deque
import time

def search_path_with_astar(start, goal, accessible_fn, h, callback_fn):
    open_set = {tuple(start)}
    closed_set = set()
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): h(start, goal)}

    while open_set:
        callback_fn(closed_set, open_set)

        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

        if current == tuple(goal):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(tuple(start))
            return path[::-1]

        open_set.remove(current)
        closed_set.add(current)

        for neighbor in accessible_fn(current):
            if tuple(neighbor) in closed_set:
                continue

            tentative_g_score = g_score.get(current, float('inf')) + h(neighbor, current)

            if tuple(neighbor) not in open_set:
                open_set.add(tuple(neighbor))
            elif tentative_g_score >= g_score.get(tuple(neighbor), float('inf')):
                continue

            came_from[tuple(neighbor)] = current
            g_score[tuple(neighbor)] = tentative_g_score
            f_score[tuple(neighbor)] = g_score[tuple(neighbor)] + h(neighbor, goal)

    return []

def manhattan_heuristic(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])

def euclidean_heuristic(start, end):
    return math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

def random_heuristic(start, end):
    return random.random()

def getpixel(image, dims, position):
    if any(p < 0 or p >= dims[i] for i, p in enumerate(position)):
        return None
    return image[position[1], position[0]]

def setpixel(image, dims, position, value):
    if any(p < 0 or p >= dims[i] for i, p in enumerate(position)):
        return
    image[position[1], position[0]] = value

def accessible(bitmap, dims, point):
    neighbors = []
    height, width = dims
    for i in range(len(point)):
        for delta in [-1, 1]:
            neighbor = list(point)
            neighbor[i] += delta
            neighbor = tuple(neighbor)

            x, y = neighbor[0], neighbor[1]
            if 0 <= x < width and 0 <= y < height:
                if bitmap[y, x][0] == 0:
                    neighbors.append(neighbor)
    return neighbors

def load_world_map(fname):
    img = Image.open(fname)
    img = img.convert("RGBA")
    pixels = np.array(img)
    dims = pixels.shape[:2]
    return dims, pixels

def save_world_map(fname, image):
    img = Image.fromarray(image)
    img.save(fname)

def find_pixel_position(image, dims, value):
    for y in range(dims[0]):
        for x in range(dims[1]):
            if tuple(image[y, x]) == value:
                return [x, y]
    raise ValueError("Nie znaleziono piksela o podanej wartości!")

if __name__ == "__main__":
    dims, bitmap = load_world_map("img.png")

    start = find_pixel_position(bitmap, dims, (255, 0, 255, 255))
    goal = find_pixel_position(bitmap, dims, (255, 255, 0, 255))

    heuristics = {
        'manhattan': manhattan_heuristic,
        'euclidean': euclidean_heuristic,
        'random': random_heuristic
    }

    for name, heuristic in heuristics.items():
        setpixel(bitmap, dims, start, (0, 0, 0, 255))
        setpixel(bitmap, dims, goal, (0, 0, 0, 255))

        start_time = time.time()
        path = search_path_with_astar(start, goal, lambda p: accessible(bitmap, dims, p), heuristic, lambda x, y: None)
        end_time = time.time()

        if path:
            print(f"Heurystyka: {name}, Długość trasy: {len(path)}, Czas obliczeń: {end_time - start_time:.6f} sekund")
            for p in path:
                setpixel(bitmap, dims, p, (255, 0, 0, 255))
        else:
            print(f"Heurystyka: {name}, Nie znaleziono trasy.")

        save_world_map(f"result_{name}.png", bitmap)
