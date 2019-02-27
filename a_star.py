#!/usr/bin/env python3

from gridenv import GridEnv
from navigator import Navigator
import numpy as np
import time

from queue import PriorityQueue

def _dist_between(tile1, tile2):
    """
    Calculates the distance between two tiles.

    Args:
    	tile1: First tile.
    	tile2: Second tile.
    """
    return ((tile1[0] - tile2[0])**2 + (tile1[1] - tile2[1])**2)**0.5

def _a_star_estimate(tile, goal):
    """
    Estimates the cost of getting from one tile to another.
    Uses the distance between the tiles as the estimate.

    Args:
    	tile: Coordinates of tile we originate at.
    	goal: Coordinates of tile we want to get to.

    Returns:
        Cost of getting from tile to goal.
    """
    return _dist_between(tile, goal)

def _a_star_get_neighbors(env, tile):
    """
    Gets all tiles neighboring a tile.

    Args:
    	tile: The tile to find neighbors for.

    Returns:
        Coordinates of tiles to the north, south, east, and west of the tile.
    """
    maybe = []
    maybe.append((tile[0],     tile[1] + 1)) # North
    maybe.append((tile[0] + 1, tile[1]    )) # South
    maybe.append((tile[0],     tile[1] - 1)) # East
    maybe.append((tile[0] - 1, tile[1]    )) # West

    # Only include tiles that are within the grid world's bounds and are walkable.
    neighbors = []
    for neigh in maybe:
        if env.is_in_grid(neigh[0], neigh[1]) and env.is_walkable(neigh[0], neigh[1]):
            neighbors.append(neigh)

    return neighbors

def a_star(env, start, goal):
    """
    Calculates a path from one tile to another using A* search.

    Args:
    	start: Coordinates of tile agent starts at.
    	goal: Coordinates of tile to end on.
	env: GridEnv object that tiles exist within.

    Returns:
    	Path to goal as an ordered list of tile coordinates, starting at the start and ending at
    	the goal.
    """

    # The set of discovered nodes not yet evaluated.
    open_set = PriorityQueue()
    open_set.put(start, 0) # Initial priority does not matter.

    new_2d_array = lambda default: [[default for y in range(0, env.height)] for x in range(0, env.width)]
    # Indicates which tiles have been discovered, i.e. which tiles are or have been in open_set.
    discovered = new_2d_array(False)
    discovered[start[0]][start[1]] = True

    # Indicates which tiles have already been evaluated.
    # True for have been and False for have not.
    closed_set = new_2d_array(False)

    # For each tile, which tile it can most efficiently be reached from.
    came_from = new_2d_array(None)

    # Cost of getting to a tile from the start.
    g_score = new_2d_array(None)
    g_score[start[0]][start[1]] = 0

    # Cost of getting from the start to the goal by passing through a tile.
    f_score = new_2d_array(None)
    f_score[start[0]][start[1]] = _a_star_estimate(start, goal)

    while not open_set.empty():
        current = open_set.get()
        if closed_set[current[0]][current[1]]:
            continue

        # Check if we have reached our goal.
        if current[0] == goal[0] and current[1] == goal[1]:
            # Construct path from start to goal.
            current = goal
            path = [current]
            while current[0] != start[0] or current[1] != start[1]:
                current = came_from[current[0]][current[1]]
                path.insert(0, current)
            return path

        closed_set[current[0]][current[1]] = True

        for neigh in _a_star_get_neighbors(env, current):
            # G-scores of None are treated as infinite.
            cur_g_score = g_score[current[0]][current[1]] + 1
            neigh_g_score = g_score[neigh[0]][neigh[1]]

            if not discovered[neigh[0]][neigh[1]]:
                # We have found an undiscovered node.
                open_set.put(neigh, f_score[neigh[0]][neigh[1]])
                discovered[neigh[0]][neigh[1]] = True
            elif (not (neigh_g_score is None)) and cur_g_score >= neigh_g_score:
                continue

            came_from[neigh[0]][neigh[1]] = current
            g_score[neigh[0]][neigh[1]] = cur_g_score
            f_score[neigh[0]][neigh[1]] = cur_g_score + _a_star_estimate(neigh, goal)

def main():
    env = GridEnv()
    try:
        while True:
            env.reset()
            goal = env.find_goals()[0]
            path = a_star(env, env.get_agent_pos(), goal)
            nav = Navigator(env, path)
            print("Executing plan...")
            while (nav.has_next()):
                nav.do_next_step()
                env.render()
                time.sleep(0.1)
            print("Plan completed")

            pos = env.get_agent_pos()
            if pos[0] == goal[0] and pos[1] == goal[1]:
                print("Goal reached")
            else:
                print("Goal not reached")
            print()
    except KeyboardInterrupt:
        print("program exited...")
    print("done")

if __name__ == "__main__":
    main()
