'''
Author: Simone Cerrato
'''
from A_star_planner import *
from Gradient import *
import time
from scipy.ndimage import convolve

'''This planner is intended for orchestrating the total path planning problem. It loops over
the waypoints array selecting the A* planner or the Gradient one according to
considered waypoint. The A* planner has been used in the curved portion of the path,
while the Gradient planner has been used in the straight portion of the path.'''

REDUCED_MARGIN_LIMIT = 10

# This method extracts the obstacles position (in pixel) in the provided mask
def obstacles_extraction(im,threshold):
    # x on columns and y on rows
    im_array = np.array(im)
    rows = size(im,0)
    columns = size(im,1)
    obstacles = []
    obstacles_reduced = []
    obst_dict = {}
    for i in range(0,rows):
        for j in range(0,columns):
            if(im_array[i][j]<threshold):
                obstacles.append([j,i])
                obst_dict[(j,i)] = 1
            if (im_array[i][j]<threshold) and not (im_array[i+1][j]<threshold and im_array[i-1][j]<threshold
            and im_array[i][j+1]<threshold and im_array[i][j-1]<threshold):
                obstacles_reduced.append([j,i])
                
    return obstacles,obstacles_reduced,obst_dict,columns,rows

# Compute the new boundary limits of the mask based on the obstacles position
#  in order to cut out large blank parts of the mask boundaries
def margin_reduction(obstacles,x_dimension,y_dimension):

    max_x = max(obstacles[:,0])
    min_x = min(obstacles[:,0])
    max_y = max(obstacles[:,1])
    min_y = min(obstacles[:,1])
    max_x = max_x + int((x_dimension-max_x)/2)
    min_x = min_x - int((min_x)/2)
    max_y = max_y + int((y_dimension-max_y)/2)
    min_y = min_y - int((min_y)/2)

    return min_x, max_x, min_y, max_y

# It computes the occupancy map used by the A* algorithm
def compute_costmap_a_star(image_path,kernel_size):
    img = cv.imread(image_path,0)
    y_dimension = size(array(img),0)
    x_dimension = size(array(img),1)

    # Dynamic threshold
    dynamic_th = 180 + (kernel_size-1)/2*10
    if dynamic_th > 240:
        dynamic_th = 240

    im_array = array(img)
    blur_image = cv.GaussianBlur(im_array,(kernel_size,kernel_size),0)
    # It transforms the mask image in a bit-like image in order to detect occupied cells and free ones
    # 255 = obstacle, 0 = free space
    ret, bw_img = cv.threshold(blur_image,dynamic_th,255,cv.THRESH_BINARY_INV)
    #ret, bw_img_save = cv.threshold(blur_image,dynamic_th,255,cv.THRESH_BINARY)
    #cv.imwrite("costmap_astar.png",bw_img_save)

    return bw_img


# Compute the kernel size based on the minimum distance between obstacles and start/goal point.
def compute_kernel_size(control_points,obstacles):
    min_distance = []
    for point in control_points:
        x_differences = (obstacles[:,0] - point[0])**2
        y_differences = (obstacles[:,1] - point[1])**2
        min_distance +=np.sqrt(x_differences+y_differences).tolist()   
    min_absolute = int(round(min(min_distance)))

    kernel_size = (min_absolute -  min_absolute%2)*2 + 1

    return kernel_size

if __name__=="__main__":
    # Read configuration file
    with open("config.json") as json_data_file:
        data = json.load(json_data_file)
    
    # Opening the mask image and converting it to grayscale
    im = Image.open(data["base_path"]+data["mask_image"]).convert("L")

    # Obstacles position extraction: 
    # obst_list = total obstacles, obst_reduced = only external pixel representing row crops, 
    # obst_dict = total obstacles in a dictionary for fast search
    obst_list,obst_reduced,obst_dict, x_dimension, y_dimension = obstacles_extraction(im,data["threshold"])
    obstacles = np.array(obst_list)
    print("Obstacle extraction ended!")

    # Load the waypoints array
    control_array = np.load(data["base_path"]+data["control_points_file"])

    # Select a reduced portion of the mask based on the minimum and maximum obstacle positions
    min_x, max_x, min_y, max_y = margin_reduction(obstacles,x_dimension,y_dimension)

    # Creation of planners
    gradient_planner = Gradient(im,data["Resolution"],data["K_goal_gradient"],data["K_blurring_gradient"],obst_dict)
    a_star = A_star_planner(min_x, max_x, min_y, max_y,data["Resolution"])
    
    total_path = []
    print("Start path planning!")
    start_time = time.time()
    # Loop on the waypoints array
    for i in range(0,len(control_array)-1):
        sx = control_array[i][0]
        sy = control_array[i][1]
        gx = control_array[i+1][0]
        gy = control_array[i+1][1]
        print(f"sx: {sx},sy: {sy},gx: {gx},gy: {gy}")
        kernel_size = compute_kernel_size([[sx,sy],[gx,gy]],np.array(obst_reduced))
        if i%2==0:
            min_x = min(sx,gx) - 2*REDUCED_MARGIN_LIMIT
            max_x = max(sx,gx) + 2*REDUCED_MARGIN_LIMIT
            min_y = min(sy,gy) - 2*REDUCED_MARGIN_LIMIT
            max_y = max(sy,gy) + 2*REDUCED_MARGIN_LIMIT
            print(f"min_x: {min_x}, min_y: {min_y}, max_x: {max_x}, max_y: {max_y}")
            # This is the case when control points are very close to vine rows
            if kernel_size == 1:
                kernel_size = data["kernel_size_gradient"]
            total_path += gradient_planner.path_planning(sx,sy,gx,gy,min_x,max_x,min_y,max_y,kernel_size)
        
        else:
            costmap_a_star = compute_costmap_a_star(data["base_path"]+data["mask_image"],kernel_size)
            total_path += a_star.planning(sx, sy, gx, gy,costmap_a_star)

        print("Goal %d found"%(i+1))

    end_time = time.time()
    print("End path planning in %.2f seconds!"%(end_time-start_time))
    path = np.array(total_path)
    np.save(data["base_path"]+data["file_total_path"],path)

    # Plot the founded path
    plt.figure()
    plt.plot(path[:,0],path[:,1],'or',obstacles[:,0],obstacles[:,1],'.k',control_array[:,0],control_array[:,1],'ob')
    plt.legend(['Esimated Path','Row Crops','Waypoints'])
    plt.show()
    
    


    
