import numpy as np
import cv2
import glob
import matplotlib.image as mpimg

# Reading all images in a list using glob
image_files = glob.glob('camera_cal/calibration*.jpg')

"""
Implementation Notes:
--------------------
We need to map Image Points to Object Points.

Image Points: The coordinates of the corners in these 2D images 
- To get image points, I am using cv2.findChessboardCorners.

Object Points: The 3D coordinates of real undistorted chess board corners
- The object points are known; they are the known coordinates of the chessboard corners for a 9x6 board. 
- This points will be 3D coordinates.
- For an 9x6 board:
     Top left corner = (0,0,0)
     Bottom right corner = (8,5,0)
     The z, in (x,y,z) is 0 for all corners, since the chessboard is a flat 2D surface.

"""

# Array to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

def calib():
    """
    #--------------------
    # To get an undistorted image, we need camera matrix & distortion coefficient
    # Calculate them with 9*6 20 chessboard images
    #
    """

    # Prepare object points
    # This object points will be the same for all calibration images
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # x,y coordinates

    for curr_file in image_files:

        img = mpimg.imread(curr_file)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            continue

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

def undistort(img, mtx, dist):
    """ 
    #--------------------
    # undistort image 
    #
    """
    return cv2.undistort(img, mtx, dist, None, mtx)
