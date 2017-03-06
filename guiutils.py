import cv2
from camera_calibration import calib, undistort
from threshold import get_combined_gradients, get_combined_hls, combine_grad_hls
from line import Line, get_perspective_transform, get_lane_lines_img, illustrate_driving_lane, illustrate_info_panel, illustrate_driving_lane_with_topdownview
import numpy as np

class EdgeFinder:
    def __init__(self, image, filter_size=1, threshold1=0, threshold2=0):
        self.image = image
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2

        def onchangeThreshold1(pos):
            self._threshold1 = pos
            self._render()

        def onchangeThreshold2(pos):
            self._threshold2 = pos
            self._render()

        def onchangeFilterSize(pos):
            self._filter_size = pos
            self._filter_size += (self._filter_size + 1) % 2
            self._render()

        cv2.namedWindow('edges')

        cv2.createTrackbar('threshold1', 'edges', self._threshold1, 255, onchangeThreshold1)
        cv2.createTrackbar('threshold2', 'edges', self._threshold2, 255, onchangeThreshold2)
        cv2.createTrackbar('filter_size', 'edges', self._filter_size, 20, onchangeFilterSize)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('edges')
        cv2.destroyWindow('smoothed')

    def threshold1(self):
        return self._threshold1

    def threshold2(self):
        return self._threshold2

    def filterSize(self):
        return self._filter_size

    def edgeImage(self):
        return self._edge_img

    def smoothedImage(self):
        return self._smoothed_img

    def _render(self):
        self._smoothed_img = cv2.GaussianBlur(self.image, (self._filter_size, self._filter_size), sigmaX=0, sigmaY=0)
        self._edge_img = cv2.Canny(self._smoothed_img, self._threshold1, self._threshold2)
        cv2.imshow('smoothed', self._smoothed_img)
        cv2.imshow('edges', self._edge_img)



class LaneDetector:
    def __init__(self, image, th_sobelx_X=35, th_sobelx_Y=100, th_sobely_X=30, th_sobely_Y=255, th_mag_X=30, th_mag_Y=255, th_dir_X=0.7, th_dir_Y=1.3, th_h_X=10, th_h_Y=100, th_l_X=0, th_l_Y=60, th_s_X=85, th_s_Y=255):

        self.left_line = Line()
        self.right_line = Line()

        # camera matrix & distortion coefficient
        self.mtx, self.dist = calib()

        self.image = image
        self._th_sobelx_X = th_sobelx_X
        self._th_sobelx_Y = th_sobelx_Y
        self._th_sobely_X = th_sobely_X
        self._th_sobely_Y = th_sobely_Y
        self._th_mag_X = th_mag_X
        self._th_mag_Y = th_mag_Y
        self._th_dir_X = th_dir_X
        self._th_dir_Y = th_dir_Y
        self._th_h_X = th_h_X
        self._th_h_Y = th_h_Y
        self._th_l_X = th_l_X
        self._th_l_Y = th_l_Y
        self._th_s_X = th_s_X
        self._th_s_Y = th_s_Y
        
        def onchange_th_sobelx_X(pos):
            self._th_sobelx_X = pos
            self._render()

        def onchange_th_sobelx_Y(pos):
            self._th_sobelx_Y = pos
            self._render()

        def onchange_th_sobely_X(pos):
            self._th_sobely_X = pos
            self._render()

        def onchange_th_sobely_Y(pos):
            self._th_sobely_Y = pos
            self._render()

        def onchange_th_mag_X(pos):
            self._th_mag_X = pos
            self._render()

        def onchange_th_mag_Y(pos):
            self._th_mag_Y = pos
            self._render()

        def onchange_th_dir_X(pos):
            self._th_mag_X = pos
            self._render()

        def onchange_th_dir_Y(pos):
            self._th_mag_Y = pos
            self._render()

        def onchange_th_h_X(pos):
            self._th_h_X = pos
            self._render()

        def onchange_th_h_Y(pos):
            self._th_h_Y = pos
            self._render()

        def onchange_th_l_X(pos):
            self._th_l_X = pos
            self._render()

        def onchange_th_l_Y(pos):
            self._th_l_Y = pos
            self._render()

        def onchange_th_s_X(pos):
            self._th_s_X = pos
            self._render()

        def onchange_th_s_Y(pos):
            self._th_s_Y = pos
            self._render()

        cv2.namedWindow('LaneLines')

        cv2.createTrackbar('th_sobelx_X', 'LaneLines', self._th_sobelx_X, 255, onchange_th_sobelx_X)
        cv2.createTrackbar('th_sobelx_Y', 'LaneLines', self._th_sobelx_X, 255, onchange_th_sobelx_Y)
        cv2.createTrackbar('th_sobely_X', 'LaneLines', self._th_sobely_X, 255, onchange_th_sobely_X)
        cv2.createTrackbar('th_sobely_Y', 'LaneLines', self._th_sobely_Y, 255, onchange_th_sobely_Y)
        cv2.createTrackbar('th_mag_X',    'LaneLines', self._th_mag_X,     255, onchange_th_mag_X)
        cv2.createTrackbar('th_mag_Y',    'LaneLines', self._th_mag_Y,     255, onchange_th_mag_Y)
        #cv2.createTrackbar('th_dir_X',    'LaneLines', self._th_dir_X,     20, onchange_th_dir_X)
        #cv2.createTrackbar('th_dir_Y',    'LaneLines', self._th_dir_Y,     20, onchange_th_dir_Y)
        cv2.createTrackbar('th_h_X',      'LaneLines', self._th_h_X,          255, onchange_th_h_X)
        cv2.createTrackbar('th_h_Y',      'LaneLines', self._th_h_Y,          255, onchange_th_h_Y)
        cv2.createTrackbar('th_l_X',      'LaneLines', self._th_l_X,          255, onchange_th_l_X)
        cv2.createTrackbar('th_l_Y',      'LaneLines', self._th_l_Y,          255, onchange_th_l_Y)
        cv2.createTrackbar('th_s_X',      'LaneLines', self._th_s_X,          255, onchange_th_s_X)
        cv2.createTrackbar('th_s_Y',      'LaneLines', self._th_s_Y,          255, onchange_th_s_Y)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('LaneLines')

    def th_sobelx_X(self):
        return self._th_sobelx_X

    def th_sobelx_Y(self):
        return self._th_sobelx_Y

    def th_sobely_X(self):
        return self._th_sobely_X 

    def th_sobely_Y(self):
        return self._th_sobely_Y

    def th_mag_X(self):
        return self._th_mag_X

    def onchange_th_mag_Y(pos):
        return self._th_mag_Y 

    def onchange_th_dir_X(pos):
        return self._th_mag_X 

    def onchange_th_dir_Y(pos):
        return self._th_mag_Y 

    def onchange_th_h_X(pos):
        return self._th_h_X 

    def onchange_th_h_Y(pos):
        return self._th_h_Y 

    def onchange_th_l_X(pos):
        return self._th_l_X 

    def onchange_th_l_Y(pos):
        return self._th_l_Y 

    def onchange_th_s_X(pos):
        return self._th_s_X 

    def onchange_th_s_Y(pos):
        return self._th_s_Y 


    def _render(self):
        # Correcting for Distortion
        undist_img = undistort(self.image, self.mtx, self.dist)
        # resize video
        undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
        rows, cols = undist_img.shape[:2]

        combined_gradient = get_combined_gradients(undist_img, (self._th_sobelx_X, self._th_sobelx_Y), (self._th_sobely_X,self._th_sobely_Y), (self._th_mag_X, self._th_mag_Y), (self._th_dir_X, self._th_dir_Y))

        combined_hls = get_combined_hls(undist_img, (self._th_h_X,self._th_h_Y) , (self._th_l_X, self._th_l_Y), (self._th_s_X, self._th_s_Y))

        combined_result = combine_grad_hls(combined_gradient, combined_hls)

        c_rows, c_cols = combined_result.shape[:2]
        s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
        s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

        src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
        dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

        warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (720, 720))

        searching_img = get_lane_lines_img(warp_img, self.left_line, self.right_line)

        w_comb_result, w_color_result = illustrate_driving_lane(searching_img, self.left_line, self.right_line)

        # Drawing the lines back down onto the road
        color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
        lane_color = np.zeros_like(undist_img)
        lane_color[220:rows - 12, 0:cols] = color_result

        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)

        info_panel, birdeye_view_panel = np.zeros_like(result),  np.zeros_like(result)
        info_panel[5:110, 5:325] = (255, 255, 255)
        birdeye_view_panel[5:110, cols-111:cols-6] = (255, 255, 255)
    
        info_panel = cv2.addWeighted(result, 1, info_panel, 0.2, 0)
        birdeye_view_panel = cv2.addWeighted(info_panel, 1, birdeye_view_panel, 0.2, 0)
        road_map = illustrate_driving_lane_with_topdownview(w_color_result, self.left_line, self.right_line)
        birdeye_view_panel[10:105, cols-106:cols-11] = road_map
        birdeye_view_panel = illustrate_info_panel(birdeye_view_panel, self.left_line, self.right_line)
    
        cv2.imshow('LaneLines', birdeye_view_panel)

        
