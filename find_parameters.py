"""
How to run:
python find_edges.py <image path>
"""

import argparse
import cv2
import os
import numpy as np
from guiutils import EdgeFinder, LaneDetector


def main():

    parser = argparse.ArgumentParser(description='Visualize Lane Finding Params.')
    parser.add_argument('filename')

    args = parser.parse_args()

    img = cv2.imread(args.filename)

    #cv2.imshow('input', img)

    lane_detector = LaneDetector(img)

    print ("Lane Detector parameters:")
    print ("th_sobelx_X: ", lane_detector._th_sobelx_X)
    print ("th_sobelx_Y: ", lane_detector._th_sobelx_Y)
    print ("th_sobely_X: ", lane_detector._th_sobely_X)
    print ("th_sobely_Y: ", lane_detector._th_sobely_Y)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
