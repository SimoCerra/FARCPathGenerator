
# Fast Adaptive Row Crops Path Generator (FARCPathGenerator)
This path planner is intended for computing a global path in a row-based crops mantaininig a safe distance from the rows (ideally in the mean). It is composed by two different planners:
1. Gradient based planner: used to find the minimum cost path inside the rows
2. A* based planner: used to compute the path between two different vine rows

## Content of the directory
- _utils_ contains the classes and methods used for path planning.
- _main.py_ is the entry point and orchestrator of the overall system, switching from the two planners according to the provided control points.
- _dataset_ is a directory containing different masks with the related waypoints and it is organized in the following manner:
    1. _satellite_ and _train_ contain the real word and synthetic dataset used for experimentation, respectively.
        1. _imgX_ are the mask images numbered from 1 to 100.
        2. _waypointsX_ are the corresponding waypoints arrays estimated by the deep neural network.
        3. _waypoints.csv_ contain the ground truth waypoints for each mask image.
- _config.json_ is a configuration file containinig:
    1. _base\_path_ is the path to the directory where are stored the mask and the waypoints to be tested.
    2. _mask\_image_ is the name of the mask to be tested.
    3. _control\_points\_file_ is the name of the file containing the waypoints related to the _mask\_image_(above)
    4. _threshold_ is used to distinguish free space and obstacles in the provided mask
    5. _K\_goal\_gradient_ indicates how much the gradient based planner attempts to reach the goal
    6. _K\_blurring\_gradient_ indicates how much the blurring is taken into account in gradient based planner for the global path computation. The higher this value the more central will be the path
    7. _Resolution_ is a measure of how many pixels the planner will move formward at each step. The higher this value the faster will be the computation.
    8. _kernel\_size\_gradient_ is the minimum value of the kernel dimension used to apply the blurring on the mask image
    9. _file\_total\_path_ is the filename where will be stored the computed total path
