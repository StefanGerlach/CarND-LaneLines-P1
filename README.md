# **Finding Lane Lines on the Road** 

### This is the first project of the Udacity Self Driving Car Nanodegree Program and implements a basic pipeline to detect road lanes in a video.

[//]: # (Image References)

[image1]: ./writeup_images/pipeline_overview.png "Overview"
[image2]: ./writeup_images/detection.png "Road Lanes Detection"

![Road Lanes Detection][image2]


## Reflection

### 1. Description of the pipeline.

To tackle the problem I constructed the a pipeline that consists of several sequentially processed steps. This process can be run in visualistion-mode to display the intermediate steps with the help of the matplolib.pyplot-lib. The basic description of the pipeline is:

1. Initialize parameter for intermediate steps.
2. Convert the image from RGB colorspace to grayscale.
3. Create a mask (mask_brightness), based on all pixels above a predefined threshold, since usually the road lanes are quite bright in the images.
4. Create a mask (mask_vertices), based on a predefined shape in the center of the field of view.
5. Preprocess the grayscale image with gaussian blur.

6. Apply the Canny Operation to detect edges in the image.
7. Mask the Canny-Output with mask_brightness.
8. Mask the Canny-Output with mask_vertices.
9. Use the Hough-Transformation to detect lines in the image.
10. Compute a histogramm over the lines with respect to their orientation and the summed lengths.
11. Select to two most frequent orientations, where one of them must be with negative and the other positive slope. Then average these two clusters of lines with opencv.fitLine() in the endpoints and midpoints. 
12. The two leftover lines are clamped to a field of view and painted into the image.

The following overview will describe the pipeline:

![Overview of the pipeline][image1]


### 2. Potential shortcomings and improvements

One problem of this pipeline are the tuned parameters, so that this sequence of processing steps works on the given dataset. The videos show in sum a ride of 46 seconds, and only one type of road modality. To improve this algorithm, there should be a much larger dataset with ground-trouth lanes for evaluating the score of my approach. 

Another problem might be the assumption, that there must be 1 line with positive slope and 1 line with negative slope. This will not hit the reality in all situations. So maybe the first most frequent slope plus the second most frequent slopes with spacial distance in the image would be better.

Additionall the lines in the image are flickering a lot and are disturbed in the challenge video at the brighter road segment. To smoothen the detected lines, a filtering over several frames would help alot. Maybe the last 5 to 10 frames could be averaged (avg, median) to reduce noise in the position of the lines.

