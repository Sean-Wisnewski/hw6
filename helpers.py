import numpy as np
import math

##################
# Helper Classes
##################
class Point:
    def __init__(self, odom):
        self.x = odom.pose.pose.position.x
        self.y = odom.pose.pose.position.y
        self.w = odom.pose.pose.orientation.w
        self.z = odom.pose.pose.orientation.z
        self.theta = self.current_angle()

    def current_angle(self):
        return 2*math.atan2(self.z, self.w)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"({self.x},{self.y},{self.theta})"

class SimplePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"({self.x},{self.y})"



##################
# Bresenham's 
##################
def plotLineLow(pt0, pt1, points, num_cells_beyond=5):
    dx = pt1.x - pt0.x
    dy = pt1.y - pt0.y
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = pt0.y

    for x in range(pt0.x, pt1.x + num_cells_beyond):
        points.append(SimplePoint(x, y))
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy
    return points

def plotLineHigh(pt0, pt1, points, num_cells_beyond=5):
    dx = pt1.x - pt0.x
    dy = pt1.y - pt0.y
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = pt0.x

    for y in range(pt0.y, pt1.y + num_cells_beyond):
        points.append(SimplePoint(x, y))
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2 * dx
    return points

def plotLine(pt0, pt1, num_cells_beyond=5):
    points = []
    if abs(pt1.y - pt0.y) < abs(pt1.x - pt0.x):
        if pt0.x > pt1.x:
            points = plotLineLow(pt1, pt0, points, num_cells_beyond)
        else:
            points = plotLineLow(pt0, pt1, points, num_cells_beyond)
    else:
        if pt0.y > pt1.y:
            points = plotLineHigh(pt1, pt0, points, num_cells_beyond)
        else:
            points = plotLineHigh(pt0, pt1, points, num_cells_beyond)
    return points


##################
# Misc helpers
##################
def l2_dist(pt0, pt1):
    return math.sqrt((pt1.x - pt0.x)**2 + (pt1.y - pt0.y)**2)

def replace_nans(lmsg):
    lmsg.ranges = list(map(replace_nan, lmsg.ranges))

def replace_nan(val):
    math.inf if math.isnan(val) else val

def compute_interior_angle(pt0, pt1, pt2):
    """
    Computes the interior angle of a vertex given 3 points.
    NOTE: Assumes that *pt0* is the vertex you would like to find the interior angle of
    :param pt0:
    :param pt1:
    :param pt2:
    :return:
    """
    angle = math.atan2(pt1.y - pt0.y, pt1.x-pt0.x) - math.atan2(pt2.y - pt0.y, pt2.x - pt0.x)
    if angle < 0:
        angle += 2*math.pi
    elif angle > 2*math.pi:
        angle -= 2*math.pi
    return angle

def compute_obstacle_point(robot_pt, d, theta, alpha):
    """
    Computed in real world, not rectified points - points should still be in the real world
    coordinate system at thisn point, not the map world
    """
    # check for math.inf here, change it to be more reasonable
    # TODO swap to use NaN?
    if d == math.inf:
        d = 5 
        print(f"changing a math.inf to {d}")
    obs_x = robot_pt.x + d * math.cos(theta + alpha)
    obs_y = robot_pt.y + d * math.sin(theta + alpha)
    return SimplePoint(obs_x, obs_y)

def compute_alpha_from_idx(scan, idx):
    assert idx <= len(scan.ranges)
    return scan.angle_min + idx * scan.angle_increment

def rectify_pos_to_map_cell(pt):
    # this will need to take an odom (x,y) pt to a map (x,y)
    # odom will be (float, float) while map will be (int, int)
    # the returned pt will be used in BH to get the new odds for a bunch of cells along the ray
    x = round(pt.x, 1)
    y = round(pt.y, 1)
    return SimplePoint(int(x * 10), int(y * 10))

def get_d_line_from_points(points, start_pt):
    d_lines = [l2_dist(pt, start_pt) for pt in points]
    return d_lines

def convert_pt_list_to_np_idx(pts, ms):
    xs = [pt.x+ms//2 for pt in pts]
    ys = [pt.y+ms//2 for pt in pts]
    return(xs, ys)

def get_points_at_sonar_end(robot_map_pt, d, w, theta, alpha):
    xdl = robot_map_pt.x + d*math.cos(theta+alpha+w/2)
    ydl = robot_map_pt.y + d*math.sin(theta+alpha+w/2)
    xdr = robot_map_pt.x + d*math.cos(theta+alpha-w/2)
    ydr = robot_map_pt.y + d*math.sin(theta+alpha-w/2)
    point_left = SimplePoint(xdl, ydl)
    point_right = SimplePoint(xdr, ydr)
    return point_left, point_right

def get_cell_map(robot_map_pt, point_left, point_right, eps=5):
    min_x = min(robot_map_pt.x, point_left.x, point_right.x)
    max_x = max(robot_map_pt.x, point_left.x, point_right.x)
    min_y = min(robot_map_pt.y, point_left.y, point_right.y)
    max_y = max(robot_map_pt.y, point_left.y, point_right.y)
    grid = np.mgrid[min_x-eps:max_x+eps, min_y-eps:max_y+eps].T.reshape(-1, 2)
    return grid

def change_pt_to_zero_idx(x, y, mapsize):
    return x+mapsize//2, y+mapsize//2

##################
# Laser Update Fn 
##################

# TODO fix this to be not be some random magic number
# realistically the max distance an object is sensed at, will probably be math.inf with the robot
# corresponds to about 10 meters in the map, maybe make it smaller?
MAX_D = 100 
# WOoO random constants for proability! (technically odds)
# will likely need to update these to make map change appropriately
#LOW_PROB = 0.25
#HIGH_PROB = 1.5
#MED_HIGH_PROB = 1.2
#MED_PROB = 0.5
#NO_UPDATE = 1
LOW_PROB = 0.9
HIGH_PROB = 1.1
MED_HIGH_PROB = 1.05
NO_UPDATE = 1

# TODO: tweak this to work with the map which looks like the scale will be in the 100's, not the 1's
def laser_odds(d_line, d, eps=30):
    # handle infinity
    if d_line == math.inf:
        d_line = MAX_D
    # i.e. obstacle detected and we can/should update the map
    if d < MAX_D:
        if d_line < d-eps:
            return LOW_PROB
        elif d-eps <= d_line <= d+eps:
            return HIGH_PROB
        elif d+eps <= d_line <= d+2*eps:
            return MED_HIGH_PROB
        else:
            return NO_UPDATE
    # no obstacle detected, only update pts close to the the robot
    else:
        # Max dist will probably need to drop to something sane like 12m for the actual implementation
        if d_line < d:
            return LOW_PROB
        else:
            return NO_UPDATE


##################
# Sonar Update Fn 
##################
# TODO make these class variables
MAX_DIST = 48
NO = 1.0
LOWEST = 0.85
LOW = 0.9
MED = 1.05
HIGH = 1.1


# TODO check that this works in all quadrants, no matter which direction the robot is facing
# This works for at least quadrant one, need to check other quadrants
def sonar_odds(cell_pt, rb_pt, pt_left, d, w=math.radians(30.2), w2=math.radians(5), eps=5):
    """
    Computes the prob/odds for a sonar pt
    :param cell_pt:
    :param rb_pt:
    :param pt_left:
    :param d: dist to the obstacle
    :param w: width of the sonar cone
    :param w2: indirectly, width of an "inner" sonar cone. Specified as how far in to go from the outer points
    :param eps:
    :return:
    """
    angle = compute_interior_angle(rb_pt, pt_left, cell_pt)
    delta = l2_dist(cell_pt, rb_pt)
    # left is slightly below zero, right slightly above 30 to account for floating pt precision nonsense
    leftmost_angle = math.radians(-0.2)
    inner_left_angle = leftmost_angle + w2
    rightmost_angle = w
    inner_right_angle = rightmost_angle - w2

    # obstacle detected
    if d != MAX_DIST:
        # cell in the "low" zone
        if delta < d - eps:
            # print("in low zone")
            if angle < leftmost_angle:
                return NO
            elif leftmost_angle <= angle < inner_left_angle:
                return LOW
            elif inner_left_angle <= angle < inner_right_angle:
                return LOWEST
            elif inner_right_angle <= angle <= rightmost_angle:
                return LOW
            else: # i.e. angle >= rightmost_angle
                return NO
        # cell in the "high" zone
        elif d - eps <= delta <= d + eps:
            # print("In high zone")
            if angle < leftmost_angle:
                    return NO
            elif leftmost_angle <= angle < inner_left_angle:
                return MED
            elif inner_left_angle <= angle < inner_right_angle:
                return HIGH
            elif inner_right_angle <= angle <= rightmost_angle:
                return MED
            else: # i.e. angle >= rightmost_angle
                return NO
        # cell beyond the high zone - no update
        else:
            # print("beyond high zone")
            return NO
    # no obstacle detected, update map with low probs
    else:
        if angle < leftmost_angle:
            return NO
        elif leftmost_angle <= angle < inner_left_angle:
            return LOW
        elif inner_left_angle <= angle < inner_right_angle:
            return LOWEST
        elif inner_right_angle <= angle < rightmost_angle:
            return LOW
        else:
            return NO

