#Advanced Lane Detection for Self-Driving Cars

[![Video White](output_videos/gif_out_track1.gif?raw=true)](https://youtu.be/Boe5HvpGnMQ)  
> Click the GIF abobe for full video.

---

###Dependencies:
* Python 3.5.x
* NumPy
* CV2
* matplotlib
* glob
* PIL
* moviepy


###Objectives:

* Compute the camera calibration matrix and distortion coefficients using a set of chessboard images.
* Apply a distortion correction to video frames.
* Use color transforms, gradients, to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image/Video/Code File References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image1b]: ./output_images/undistort_road_output.png "Undistorted"
[input_img]: ./test_images/test3.jpg "Input Image"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"

[video_in]: ./project_video.mp4 "Video"
[video_out]: ./output_videos/video_out_track1.mp4 "Track 1 Video"

[gif_track1]:   ./output_videos/gif_out_track1.gif "Track 1 GIF"

[calib1]: ./test_images/calibration1.jpg "Chessboard"
[calib_chesss]: ./output_images/undist_img_calibration1.png "Calibration Chessboard"
[01_calib_road]: ./output_images/01_undist_img.png "Calibration Road"
[02_cropped]: ./output_images/02_cropped.png "Cropped Road"
[03_sobel_thresh]: ./output_images/03_combined_gradient_img.png "Sobel Threshold"
[04_hls_thresh]: ./output_images/04_combined_hls_img.png "HLS Threshold"
[05_combined_thresh]: ./output_images/05_combined_thresh_result_img.png "Combined Threshold"
[07_warped]: ./output_images/07_warped_img.png "Warped Thresholded"
[08_sliding_window]: ./output_images/08_searching_img.png "Sliding Window"
[10_lane_illustration]: ./output_images/10_w_color_result_img.png "Lane Illustration"
[11_lane_rewarped]: ./output_images/11_color_result.png "Original Perspective"
[13_result]: ./output_images/13_final_result.png "Result"

###Implementation Details:

[`camera_calibration.py`](camera_calibration.py) : To calculate Calibration Matrix <br />
[`line.py`](line.py) : Line class, contains functions to detect lane lines <br />
[`threshold.py`](threshold.py) : Contains functions for thresholding an image <br />
[`process.py`](process.py) : Contains the image processing pipeline and main function <br />
[`find_parameters.py`](find_parameters.py) : Run GUI tool to find right parameters for various inputs <br />
[`guiutils.py`](guiutils.py) : GUI builder class


---

###Pipeline 

<p align="center">
    <img src="output_images/Pipeline/pipeline.png" alt="Pipeline" /><br>
    <b>[Pipeline for detecting Lanes]</b><br>
</p>

---

###1. Camera Calibration
The camera that is being used may have distortion, which can cause erros in calculations. So we first need to calibrate the camera and calculate the calibration matrix. Camera looks at World-Points (3D) and converts them to Image-Points (2D). Using some chessboard images, that would have predictable patterns, I am calibrating the camera. <br /> 
The code for camera calibration step is contained in the [`camera_calibration.py`](camera_calibration.py).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted Chessboard][image1]
![Undistorted Road][image1b]














####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][input_img]



####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

