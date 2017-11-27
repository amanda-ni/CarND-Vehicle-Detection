# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2a]: ./examples/car-hog.png
[image2b]: ./examples/notcar-hog.png
[image3]: ./examples/perspective-single-box.png
[image4a]: ./examples/perspective-multi-box.png
[image4b]: ./examples/all-search-windows.png
[image5a]: ./examples/heatmap.png
[image5b]: ./examples/detections.png
[image6]: ./examples/thresholded.png
[image7]: ./examples/final-detections.png
[video1]: ./project_video_output.mp4


###Histogram of Oriented Gradients (HOG)

####1. HOG feature extraction from the training images.

The code for this step is contained in the code cell [9] of the IPython notebook `Pipeline.ipynb` (or in lines 47 through 95 of the file called `lesson_functions.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images in code cell [3] of `Pipeline.ipynb`.  The below is a set of images, where the left sets are vehicles and the right sets are not. The function `show_images` was taken from a previous project is adapted to show these training data.

![alt text][image1]

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2a]
![alt text][image2b]


####2. Final choice of HOG parameters.

I tried various combinations of parameters, but in terms of computation and effort, I thought that simplest was the best. But also, the resolution (since the image was quite rich) needed to be higher, so the number orientations should be relatively high. Because the image is pretty large, I took the HOG feature to be a bit larger also, spanning sixteen pixels. Here are the parameters that I chose:

```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
save_params = False
```

####3. Training the SVM.

I trained a linear SVM in code cell [9]. Because training takes a long time, especially because I'm using the full set, I had the option of either loading parameters in or re-training the parameters. After training, there is an option to save the parameters.

To train, I used a test split of 0.8 to 0.2. This is after I concatenated features together, using a normalizing factor for spatial and histogram bins.

###Sliding Window Search

####1. Sliding Window Search

I searched through the image based on how far I believe the detection box to be away from the camera. It was relatively easy to take a look through the image and see where the lane lines matched the detection box. This is seen below. These three boxes show the far, mid, and close up boxes, each with a different scale based on how close the bottom portion of the box is away from the bottom of the image.

![alt text][image3]

####2. Examples of test images.

Ultimately I searched a continuously derived number of scales with a function called `y_window` that takes in a window size and gives a *y* offset: the bottom and the top of a box. This is in code box [5]:

```
def y_window( winsize, w_max, w_min, y_bottom, y_top ):
    winslope = (w_max - w_min) / (np.array(y_bottom) - np.array(y_top))
    return list(np.round( (winsize - w_min)/winslope + np.array(y_top) ).astype(np.int))
```

I used YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4a]
![alt text][image4b]
---

### Video Implementation

####1. Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image5a]
![alt text][image5b]

### The thresholded box is found below

![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The parameter tuning was especially difficult. Also, there were some times when the detector dropped detections in certain frames. This was likely due to the fact that the offset of the box detector was fairly large. To remedy this, tracks were laid down with consistent detections from previous frames. This is not very robust of a solution, which is acknowledged, but it helped out considerably in this case.
