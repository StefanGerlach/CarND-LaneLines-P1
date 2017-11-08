import numpy as np
import glob
import os
import math
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def euclidian_distance(p1, p2):
    """ Calculates the euclidian distance between two points. """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns hough lines.
    """
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                           maxLineGap=max_line_gap)


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def lines_histogram(lines):
    if len(lines) == 0:
        raise Exception("No Lines in lines_histogram given.")

    # A histogram of binned slopes will help us to find to dominant lane in the image.
    rounded_slopes_hist = {}
    indexing_helper = {}

    for i, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            # Calc slope and length of this line
            slope = np.arctan2(y2 - y1, x2 - x1)
            length = euclidian_distance((x1, y1), (x2, y2))

            # Then round the slope and convert to integer key for our dictionary.
            rounded_slope = int(np.around(slope, decimals=1) * 10)
            if rounded_slope not in rounded_slopes_hist:
                rounded_slopes_hist[rounded_slope] = 0

            # Add the current length
            rounded_slopes_hist[rounded_slope] += length

            # Remember the lines in a mapping dictionary
            if rounded_slope not in indexing_helper:
                indexing_helper[rounded_slope] = []

            indexing_helper[rounded_slope].append(i)

    # Returning the histogram of slopes and a mapping/indexing structure
    return rounded_slopes_hist, indexing_helper


def vertical_clip_lines(lines, min_y, max_y):
    clipped_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            vec = [x2 - x1, y2 - y1]
            vec = vec / np.linalg.norm(vec)
            p1 = (x1, y1)
            # find xs for min_y, max_y
            xs = []
            for y_t in [min_y, max_y]:
                direction = -(p1[1] - y_t) / vec[1]
                xs.append(p1[0] + (vec[0] * direction))
                xs.append(y_t)

            new_line = (int(xs[0]), int(xs[1]), int(xs[2]), int(xs[3]))
            clipped_lines.append([new_line])

    return clipped_lines


def average_lanes(lines, hist, indexing):
    # Split into Slopes < 0 and Slopes >= 0
    positives = dict((key, value) for key, value in hist.items() if key >= 0)
    negatives = dict((key, value) for key, value in hist.items() if key < 0)

    # Sort dictionaries by value
    positives = sorted(positives.items(), key=lambda x: x[1])
    negatives = sorted(negatives.items(), key=lambda x: x[1])

    maxima = []
    if len(positives) > 0:
        maxima.append(positives[-1][0])
    if len(negatives) > 0:
        maxima.append(negatives[-1][0])

    slope_points = {}
    for slope_keys in maxima:
        slope_points[slope_keys] = []
        for idx in indexing[slope_keys]:
            for line in lines[idx]:
                x1, y1, x2, y2 = line
                slope_points[slope_keys].append([x1, y1])
                slope_points[slope_keys].append([x2, y2])
                slope_points[slope_keys].append([int((x1 + x2) / 2.), int((y1 + y2) / 2.)])

    relevant_lines = []
    length = 42
    for slope_keys in slope_points:
        xv, yv, x, y = cv2.fitLine(np.array(slope_points[slope_keys]), cv2.DIST_L2, 0, 0.01, 0.01)
        relevant_lines.append(
            [(int(x + (length * xv)), int(y + (length * yv)), int(x - (length * xv)), int(y - (length * yv)))])

    return relevant_lines


def find_lanes_in_image(image, display_infos=True):
    """
    This is the main function to be called for finding the street lanes in a single frame.
    """
    # Initialize some parameters for the pipeline:
    gauss_kernel_size = 3
    canny_low_t = 50
    canny_high_t = 150
    hough_rho = 2
    hough_degrees = 4.
    hough_theta = (np.pi / 180) * hough_degrees
    hough_t = 8
    hough_min_len = 16
    hough_max_gap = 4

    # The mask for our field of view (something like a truncated cone)
    poly_verts = np.array([[(0, image.shape[0]),
                            ((image.shape[1] / 2) - (image.shape[1] * 0.05), image.shape[0] * 0.62),
                            ((image.shape[1] / 2) + (image.shape[1] * 0.05), image.shape[0] * 0.62),
                            (image.shape[1], image.shape[0])
                            ]], dtype=np.int32)

    # DEBUG Information display
    if display_infos:
        fig = plt.figure(figsize=(14, 8))

        # Now put together all components from helper function- section!
    # Convert to grayscale - make a copy
    img = grayscale(np.copy(image))

    # Blur image as preprocessing for canny-operator
    img = gaussian_blur(img, gauss_kernel_size)

    # Apply canny operation
    img = canny(img, canny_low_t, canny_high_t)

    # DEBUG Information display
    if display_infos:
        sub2 = fig.add_subplot(221)
        sub2.set_title('Canny')
        plt.imshow(img, cmap='gray')

    mask = np.ones(shape=img.shape) * 255
    mask = region_of_interest(mask, poly_verts)

    img = region_of_interest(img, poly_verts)

    # Use hough transform to detect road lanes in the image
    lines = hough_lines(img, hough_rho, hough_theta, hough_t, hough_min_len, hough_max_gap)

    # DEBUG Information display
    if display_infos:
        draw_lines(mask, lines, [128])
        sub2 = fig.add_subplot(222)
        sub2.set_title('Mask with Hough Lines')
        plt.imshow(mask, cmap='gray')

    # Calculate a Histogram over the Line-Slopes with respect to their summed line lengths
    hist, indexing = lines_histogram(lines)

    # Average lanes from collection of lines
    lines = average_lanes(lines, hist, indexing)

    # Clip the Lines to a limit
    lines = vertical_clip_lines(lines, image.shape[0] * 0.62, image.shape[0])

    # DEBUG Information display
    if display_infos:
        sub3 = fig.add_subplot(223)
        sub3.set_title('Histogram of lines-pixels with slope')
        y_data = np.array([float(hist[k]) for k in hist])
        y_data = y_data / np.max(y_data)
        plt.bar(range(len(hist)), y_data, align='center')
        x_axis = np.array([float(k) / 10. for k in hist])
        plt.xticks(range(len(hist)), x_axis)

    img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(img, lines, thickness=18)
    img = region_of_interest(img, poly_verts)

    img = np.clip(weighted_img(img, image, 1, 1), 0, 255)

    # DEBUG Information display
    if display_infos:
        sub4 = fig.add_subplot(224)
        sub4.set_title('Averaged Lines')
        plt.imshow(img)

    return img


def process_folder(path):
    # for each jpg-element in the folder, apply processing pipeline
    for elem in glob.glob(os.path.join(path, '*.jpg')):
        image = cv2.imread(elem)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result_img = find_lanes_in_image(image)


process_folder('test_images')