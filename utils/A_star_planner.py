import math
import numpy as np
import json

'''This file contains the A* planner class, to compute the path 
from a starting node to a goal node exploiting the A* algorithm.
The heuristic is given by the Eucledian distance between the current node and the goal node.'''

# Describes the node structure (x,y,cost and parent index)
class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = int(x)  
            self.y = int(y)  
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

class A_star_planner():

    def __init__(self, min_x, max_x, min_y, max_y,resolution):

        self.res = resolution
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.motion = self.get_motion_model()
        self.obstacles_map = 0

    def planning(self, sx, sy, gx, gy,costmap):

        self.obstacles_map = costmap
        start_node = Node(sx, sy, 0.0, -1)
        goal_node = Node(gx, gy, 0.0, -1)
        
        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(open_set[o],goal_node))
            current = open_set[c_id]

            if (current.x == goal_node.x and current.y == goal_node.y):
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                next_x = current.x + self.motion[i][0]
                next_y = current.y + self.motion[i][1]
                cost = current.cost + self.motion[i][2]
                node = Node(next_x, next_y, cost, c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        path = self.calc_final_path(goal_node, closed_set)

        return path

    # It extracts the path from the beginning node to the goal one
    def calc_final_path(self, goal_node, closed_set):
        path = []
        path.append([goal_node.x,goal_node.y])
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            path.append([n.x,n.y])
            parent_index = n.parent_index
        path.reverse()
        return path

    def calc_heuristic(self,node,goal):
        cost_to_goal = math.hypot(node.x-goal.x,node.y-goal.y)
        return cost_to_goal

    # Unique key that identifies a specific node in the dictionary
    def calc_grid_index(self, node):
        return str(node.x)+","+str(node.y)

    # Check collisions and map limits
    def verify_node(self, node):
        px = node.x
        py = node.y

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False
        
        # collision check
        if self.obstacles_map[py][px]!=0:
            return False

        return True

    # Returns the admissible movements direction from a node
    def get_motion_model(self):
        motion = [[self.res, 0,self.res],
                  [0, self.res,self.res],
                  [-self.res, 0,self.res],
                  [0, -self.res,self.res],
                  [-self.res, -self.res,self.res],
                  [-self.res, self.res,self.res],
                  [self.res, -self.res,self.res],
                  [self.res, self.res,self.res]]

        return motion

