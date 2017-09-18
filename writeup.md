## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./doc_images/chessboard_corners.png "Chessboard corners"
[image1]: ./doc_images/chessboard_undist.png "Undistorted chessboard"
[image2]: ./doc_images/test3_undist.png "Undistorted test3"

[image3]: ./doc_images/thres_x_grad.png "Threshold x gradient"
[image4]: ./doc_images/thres_y_grad.png "Threshold y gradient"
[image5]: ./doc_images/thres_mag.png "Threshold magnitude"
[image6]: ./doc_images/thres_grad.png "Threshold gradient"
[image7]: ./doc_images/thres_hls_s.png "Threshold HLS S channel"
[image8]: ./doc_images/thres_combined.png "Threshold Combined"

[image9]: ./doc_images/pipeline_teset_images.png "Threshold test with all test images"
[image10]: ./doc_images/warp_test_images.png "Warper test with all test images"

[image11]: ./examples/color_fit_lines.jpg "Fit Visual"
[image12]: ./doc_images/find_lines.png "Find lane-line pixels"

[image13]: ./output_images/Curv_and_Dist-test4.png "Example of a result"

[video1]: ./project_video_processed.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 63 through 135 of the file called `project.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detected all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Here is the corner detection results. The black indicates that the corner detection was failed. But this is completely OK since there are many other images that can be used for the calibration.

![alt_text][image0]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Even if the effect of distortion correction is subtle, I can see the corrections in the corner of the test image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 136 through 262 in `project.py`).  Here are examples of my output for this step. 

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

And then finally I tested my image processing pipeline with all the test images in the `test_images` folder.

![alt text][image9]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 268 through 275 in the file `project.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner as a function:

```python
def get_src_dst_for_transform(img):
    h, w = img.shape[:2]
    img_size = (w, h)
    
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

    return src, dst
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto test images and their warped counterparts to verify that the lines appear parallel in the warped image.

![alt text][image10]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image11]

The code for finding lane-line pixels and their positions includes `sliding_window` and `find_lines`, which appears in lines 299 through 460 in the file `project.py`.

![alt text][image12]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 461 through 501 in my code in `project.py`. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 502 through 563 in my code in `project.py` in the functions `draw_lanes()` and `draw_lane_info()`.  Here is an example of my result on a test image:

![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link](./project_video_processed.mp4) to my video result.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

According to my experiemnts with the threshold techniques in a test image, the HLS threshold in the S channel showed clear output for distinct line segments. So I decided to use the HLS (S channel) as one major feature to determine lanes. But the HLS threshold often missed small line segments that can be captured by gradient or magnitude threshold methods. Therefore, I combined those features with the HLS to detect lanes.

To make my lane detection more robust, sanity checks will be implemented if time allows.

