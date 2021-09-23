import math
import cv2 as cv
import numpy as np
from PIL import Image
from pylab import *
import json
from scipy import io

'''This planner is responsible for finding the minimum cost path 
between a starting point to an end point exploiting the gradient descent principle.'''

class Gradient():
    def __init__(self,im,resolution,k_goal,k_blurring,obstacles):
        self.K_goal = k_goal
        self.res = resolution
        self.obstacles = obstacles
        self.motion = self.get_motion_model()
        self.x_dimension, self.y_dimension = self.get_dimensions(im)
        self.K_blurring = k_blurring
        self.im = im
        self.visited = {}
    
    # It retrives the mask image dimensions
    def get_dimensions(self,im):
        rows = size(im,0)
        columns = size(im,1)
        return columns, rows

    # It computes the costmap related to the blurred image
    def compute_costmap_blurring(self,im,kernel_size):
        im_array = array(im)
        costmap_blurring = 255 - cv.GaussianBlur(im_array,(kernel_size,kernel_size),0)
        return costmap_blurring

    # Customizable motion model
    def get_motion_model(self):

        motion = [[-self.res, 0],
                    [0, self.res],
                    [self.res, 0],
                    [0, -self.res],
                    [-self.res, -self.res],
                    [-self.res, self.res],
                    [self.res, -self.res],
                    [self.res, self.res]]

        return motion

    # It computes the costmap according to the distance from the goal. 
    # In 3D is a cone with the cusp located on the goal point.
    def calc_costmap_wrt_goal(self,min_x,max_x,min_y,max_y,gx,gy):
        max_value = -1
        z = np.ones((self.x_dimension,self.y_dimension),dtype=float)*float("inf")
        for y in range(min_y,max_y):
            for x in range(min_x,max_x):
                temp = math.hypot(x-gx,y-gy)
                if temp > max_value:
                    max_value = temp
                z[y][x] = temp
        
        z = self.K_goal*(z/max_value*255)
        return z

    # It expands a parent node and retrives the minimum cost child
    def expand_node(self,node_x,node_y,costmap,gx,gy):
        actual_cost = float("inf")
        actual_x = -1
        actual_y = -1
        temp = float("inf")
        for motion in self.motion:
            next_x = node_x + motion[0]
            next_y = node_y + motion[1]
            flag_visited = 0
            flag_obst = 0
            if (next_x,next_y) in self.visited.keys():
                flag_visited = 1
            if (next_x,next_y) in self.obstacles.keys():
                flag_obst = 1
            if costmap[next_y][next_x] < actual_cost and flag_visited==0 and flag_obst==0:
                actual_cost = costmap[next_y][next_x]
                actual_x = next_x
                actual_y = next_y

            self.visited[(next_x,next_y)] = 1
        return actual_x, actual_y

    # It computes the minimum cost path bewteen a start point and a goal point
    def calc_min_cost_path(self,costmap,sx,sy,gx,gy):
        last_node_x = sx
        last_node_y = sy

        # It stores the total path
        path = []
        path.append([sx,sy])

        # It stores the visited nodes as a dictionary {key=(x_node,y_node),value=1} 
        # in order to have average searching cost=1 
        # thank to heap organization of a dictionary in python
        self.visited = {}
        self.visited[(sx,sy)] = 1

        costmap[gy][gx] = -1 # It ensures to find the goal
        distance_goal = float("inf")
        
        while (last_node_x != gx or last_node_y != gy) and (distance_goal > 2*self.res):
            last_node_x,last_node_y = self.expand_node(last_node_x,last_node_y,costmap,gx,gy)
            
            if last_node_x == -1 and last_node_y == -1:
                print("Unable to find the goal!")
                break
            else:
                path.append([last_node_x,last_node_y])
                distance_goal = math.hypot(last_node_x-gx,last_node_y-gy)
        
        path.append([gx,gy])
        return path
    
    # It works as a interface between the hybrid planner and the gradient class
    def path_planning(self,sx,sy,gx,gy,min_x, max_x, min_y, max_y,kernel_size):
        costmap_goal = self.calc_costmap_wrt_goal(min_x,max_x,min_y,max_y,gx,gy)
        costmap_blurring = self.K_blurring*self.compute_costmap_blurring(self.im,kernel_size)
        tot_costmap = self.K_blurring*self.compute_costmap_blurring(self.im,kernel_size) + costmap_goal

        data = {"costmap_goal" : costmap_goal, "costmap_blurring": costmap_blurring, "costmap_tot": tot_costmap}
        io.savemat("costmaps.mat",data)
        path = self.calc_min_cost_path(tot_costmap,sx,sy,gx,gy)
        return path
