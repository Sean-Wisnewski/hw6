#!/usr/bin/python
'''
  Some Tkinter/PIL code to pop up a window with a gray-scale
  pixel-editable image, for mapping purposes.  Does not run
  until you fill in a few things.

  Does not do any mapping.

  Z. Butler, 3/2016, updated 3/2018 and 3/2020
'''

import tkinter as tk
from PIL import Image, ImageTk
import random, sys
import rospy
from sensor_msgs.msg import LaserScan
from p2os_msgs.msg import SonarArray
from nav_msgs.msg import Odometry
import time
import numpy as np
from helpers import *

# a reasonable size? depends on the scale of the map and the
# size of the environment, of course:
MAPSIZE = 500 
PIXEL_SIZE = 2

class Mapper(tk.Frame):    

    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.master.title("I'm the map!")
        self.master.minsize(width=MAPSIZE*PIXEL_SIZE,height=MAPSIZE*PIXEL_SIZE)

        # makes a grey-scale image filled with 50% grey pixels
        # you can change the image type if you want color, check
        # the PIL (actually, Pillow) documentation
        self.themap = Image.new("L",(MAPSIZE*PIXEL_SIZE,MAPSIZE*PIXEL_SIZE),128)
        self.mapimage = ImageTk.PhotoImage(self.themap)

        # this gives us directly memory access to the image pixels:
        self.mappix = self.themap.load()
        # keeping the odds separately saves one step per cell update:
        #self.oddsvals = [[1.0 for _ in range(MAPSIZE)] for _ in range(MAPSIZE)]
        self.oddsvals = np.ones((MAPSIZE, MAPSIZE))

        self.canvas = tk.Canvas(self,width=MAPSIZE*PIXEL_SIZE, height=MAPSIZE*PIXEL_SIZE)

        self.map_on_canvas = self.canvas.create_image(MAPSIZE*PIXEL_SIZE/2, 
                MAPSIZE*PIXEL_SIZE/2, image = self.mapimage)
        self.canvas.pack()
        self.pack()

        ##############
        # My stuff
        ##############

        ##############
        # queues and flags
        ##############
        self.sonar_q = []
        self.laser_q = []
        self.odom_q  = []
        self.use_sonar = False 
        self.use_laser = False 
        # Counter to keep track of number of updates
        # reset to zero when LARGE_UPDATES hit, indicating to clear old values from queues
        self.num_updates = 0
        # alphas for the SONAR
        self.alphas = [math.radians(90), math.radians(50), math.radians(30), math.radians(10), math.radians(-10), math.radians(-30), math.radians(-50), math.radians(-90)]
        self.w = math.radians(30)

        ##############
        # NAMED CONSTANTS
        ##############
        self.OLD_TIME = 5.0
        # clear stuff after 50 updates, or roughly 5 seconds
        self.LARGE_UPDATES = 50


    def update_image(self):
        self.mapimage = ImageTk.PhotoImage(self.themap)       
        self.canvas.create_image(MAPSIZE*PIXEL_SIZE/2, MAPSIZE*PIXEL_SIZE/2, image = self.mapimage)

    def odds_to_pixel_value(self, value):
        # This seems backwards for some reason rn
        # 0 = black, 255 = white
        if value < 1:
            return math.floor(255 - 255*value)
        elif 1 < value:
            val = min(1.999, value)
            return 255-math.floor(255*val)
        else:
            return 128

    def sonar_update(self, smsg):
        self.sonar_q.append(smsg)

    def odom_update(self, omsg):
        self.odom_q.append(omsg)
        self.map_update()

    def laser_update(self,lmsg):
        #print("Before", lmsg.ranges)
        replace_nans(lmsg)
        #print("After", lmsg.ranges)
        self.laser_q.append(lmsg)

    # here I am putting the map update in the laser callback
    # of course you will need a sonar callback and odometry as well
    # actually, a completely separate function on a timer might be
    # a good idea, either way be careful if it takes longer to run
    # than the time between calls...

    def laser_map_update(self, odom, scan):
        robot_pt = Point(odom)
        robot_map_pt = rectify_pos_to_map_cell(robot_pt)
        for idx, dist in enumerate(scan.ranges):
            if dist is None:
                continue
            alpha = compute_alpha_from_idx(scan, idx)
            # TODO take math.inf into account probably
            # TODO probably take it into account here, maybe just throw math.inf measures away?
            obs_pt = compute_obstacle_point(robot_pt, dist, robot_pt.theta, alpha)
            obs_map_pt = rectify_pos_to_map_cell(obs_pt)
            # go .3m before/after the object
            map_pts = plotLine(robot_map_pt, obs_map_pt, 20)
            d_lines = get_d_line_from_points(map_pts, robot_map_pt)
            d_map_obs = l2_dist(robot_map_pt, obs_map_pt)
            # TODO call the laser odds fn
            lsr_odds = [laser_odds(d_line, d_map_obs, eps=3) for d_line in d_lines]
            # this now converts to the map coordinates to be at the correct index to not go out of bounds
            # tuple of two lists: ([x_coords], [y_coords])
            map_pts_idx = convert_pt_list_to_np_idx(map_pts, MAPSIZE)
            self.oddsvals[map_pts_idx] *= lsr_odds

    def sonar_map_update(self, odom, ranges):
        #print("in sonar")
        robot_pt = Point(odom)
        robot_map_pt = rectify_pos_to_map_cell(robot_pt)
        for idx, dist in enumerate(ranges.ranges):
            alpha = self.alphas[idx]
            # to avoid having to convert dist to map dist, find the points in the
            # real world then rectify them to the map world
            # also note that these points are still gonna be negative, and will need to be 
            # changed ot the correct index before updating the map/oddsvals
            pt_l, pt_r = get_points_at_sonar_end(robot_pt, dist, self.w, robot_pt.theta, alpha)
            pt_map_l = rectify_pos_to_map_cell(pt_l)
            pt_map_r = rectify_pos_to_map_cell(pt_r)
            # at this point, work in the map system, not the real world
            cell_map = get_cell_map(robot_map_pt, pt_map_l, pt_map_r)
            map_xs = cell_map[:, 0]
            map_ys = cell_map[:, 1]
            # haha I'm an idiot and still need to convert real world dist to map dist
            map_dist = dist*10
            snr_odds = [sonar_odds(SimplePoint(pt[0], pt[1]), robot_map_pt, pt_map_l, map_dist) for pt in cell_map] 

            # I'm just gonna apologize in advance for the numpy nonsense
            # [[x, y, sonar_odds]] - points NOT rectified
            loc_plus_odds = np.array([map_xs, map_ys, np.array(snr_odds)]).T
            # get idxes to actually update, i.e. places where the odds != 1
            just_update = np.asarray(loc_plus_odds[:, 2] != 1.0).nonzero()
            # offset the coords to be at the correct index in the map st (0,0) is in the middle
            offset_coords = [change_pt_to_zero_idx(pt[0], pt[1], MAPSIZE) for pt in loc_plus_odds]
            xs = [int(pt[0]) for pt in offset_coords]
            ys = [int(pt[1]) for pt in offset_coords]
            offset_as_idx = (xs, ys)
            self.oddsvals[offset_as_idx] *= snr_odds

    def both_map_update(self, odom, scan, ranges):
        # call the two updates in sequence, using values from (roughly) the same timestamp
        self.laser_map_update(odom, scan)
        self.sonar_map_update(odom, ranges)

    ##############
    # MAP UPDATE FUNCTION
    ##############
    def map_update(self):
        """
        Updates the map! called from the odom callback, update speed will depend on if it is using laser/sonar/both
        """
        #rospy.loginfo("Updating the map!")
        # TODO update to pick the correct queues to pop from
        # jk just grab everything b/c I'm lazy
        # this will almost certainly bite me in the ass later but oh well
        result = self.get_vals_from_queue(self.laser_q, self.sonar_q)

        # a pair or triplet of data was successfully obtained
        if result is not None:
            if self.use_laser and self.use_sonar:
                #rospy.loginfo("Using both to update")
                self.both_map_update(result[0], result[1], result[2])
            elif self.use_laser:
                #rospy.loginfo("Using laser to update")
                self.laser_map_update(result[0], result[1])
            else:
                #rospy.loginfo("Using sonar to update")
                self.sonar_map_update(result[0], result[2])
            self.num_updates += 1
            if self.num_updates == self.LARGE_UPDATES:
                rospy.loginfo("CLEAN QUEUES")
                self.clean_queues([self.odom_q, self.laser_q, self.sonar_q])
            ######
            # Actually updating the map
            ######
            dilated = dilate_map(self.oddsvals, PIXEL_SIZE)
            for x in range(0, MAPSIZE*PIXEL_SIZE):
                for y in range(0, MAPSIZE*PIXEL_SIZE):
                    val = self.odds_to_pixel_value(dilated[x,y])
                    self.mappix[x,y] = self.odds_to_pixel_value(self.oddsvals[x,y])



            
        # this puts the image update on the GUI thread, not ROS thread!
        # also note only one image update per scan, not per map-cell update
        self.after(0,self.update_image)    

    def get_vals_from_queue(self, queue, queue_two=None, t_eps=0.15):
        matching_flag = False
        result = None
        if queue_two is None:
            while len(self.odom_q) > 0 and len(queue) > 0 and not matching_flag:
                odom = self.odom_q.pop()
                val = queue.pop()
                duration = abs(odom.header.stamp - val.header.stamp)
                t = duration.to_sec()
                if t <= t_eps:
                    matching_flag = True
                    result = (odom, val)
                    #rospy.loginfo("got a matching pair")
                    return result
                else:
                    rospy.loginfo("Throwing a pair away")
        else:
            while len(self.odom_q) > 0 and len(queue) > 0 and len(queue_two) > 0 and not matching_flag:
                odom = self.odom_q.pop()
                val = queue.pop()
                val2 = queue_two.pop()
                duration = abs(odom.header.stamp - val.header.stamp)
                t = duration.to_sec()
                duration2 = abs(odom.header.stamp - val2.header.stamp)
                t2 = duration.to_sec()
                duration3 = abs(val.header.stamp - val2.header.stamp)
                t3 = duration.to_sec()
                if t <= t_eps and t2 < t_eps and t3 <= t_eps:
                    matching_flag = True
                    result = (odom, val, val2)
                    #rospy.loginfo("got a matching triplet")
                    return result
                else:
                    x = 5
                    #rospy.loginfo("Throwing a pair away")

    def clean_queues(self, queues):
        lambda_fn = lambda x: (abs(time.time() - x.header.stamp.to_sec()) <= self.OLD_TIME)
        for queue in queues:
            rospy.loginfo(f"len before clear: {len(queue)}")
            queue = list(filter(lambda_fn, queue))
            rospy.loginfo(f"len after clear: {len(queue)}")




######
# Main
#####
def main():


    rospy.init_node("mapper")

    root = tk.Tk()
    m = Mapper(master=root,height=MAPSIZE,width=MAPSIZE)
    if len(sys.argv) != 2:
        print("specify which device to use")
        sys.exit()
    else:
        if len(sys.argv) == 2:
            if sys.argv[1] == "laser":
                print("use laser")
                m.use_laser = True
            elif sys.argv[1] == "sonar":
                print("use sonar")
                m.use_sonar = True
            # make the lazy assumption that anything else means to use both
            else:
                m.use_laser = True
                m.use_sonar = True
    rospy.Subscriber("/scan/",LaserScan,m.laser_update)
    rospy.Subscriber("/sonar/",SonarArray,m.sonar_update)
    rospy.Subscriber("/pose/",Odometry,m.odom_update)
    
    # the GUI gets the main thread, so all your work must be in callbacks.
    root.mainloop()

if __name__ == "__main__":
    main()
