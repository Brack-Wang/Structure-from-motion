'''
Questions 2-4. Fundamental matrix estimation

Question 2. Eight-point Estimation
For this question, your task is to implement normalized and unnormalized eight-point algorithms to find out the fundamental matrix between two cameras.
We've provided a method to compute the average geometric distance, which is the distance between each projected keypoint from one image to its corresponding epipolar line in the other image.
You might consider reading that code below as a reminder for how we can use the fundamental matrix.
For more information on the normalized eight-point algorithm, please see this link: https://en.wikipedia.org/wiki/Eight-point_algorithm#Normalized_algorithm
Note that the normalized version may not necessarily perform better.

Question 3. RANSAC
Your task is to implement RANSAC to find out the fundamental matrix between two cameras if the correspondences are noisy.

Please report the average geometric distance based on your estimated fundamental matrix, given 1, 100, and 10000 iterations of RANSAC.
Please also visualize the inliers with your best estimated fundamental matrix in your solution for both images (we provide a visualization function).
In your PDF, please also explain why we do not perform SVD or do a least-square over all the matched key points.

Question 4. Visualizing Epipolar Lines
Please visualize the epipolar lines for both images for your estimated F in Q2 and Q3.

To draw on images, cv2.line, cv2.circle are useful to plot lines and circles.
Check our Lecture 4, Epipolar Geometry, to learn more about equation of epipolar line.
Our Lecture 4 and 5 cover most of the concepts here.
This link also gives a thorough review of epipolar geometry:
    https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
'''

import os
import cv2 # our tested version is 4.5.5
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import random
from pathlib import Path

basedir= Path('assets/fountain')
img1 = cv2.imread(str(basedir / 'images/0000.png'), 0)
img2 = cv2.imread(str(basedir /'images/0005.png'), 0)

f, axarr = plt.subplots(2, 1)
axarr[0].imshow(img1, cmap='gray')
axarr[1].imshow(img2, cmap='gray')
# plt.show()

# --------------------- Question 2

def calculate_geometric_distance(all_matches, F):
    """
    Calculate average geometric distance from each projected keypoint from one image to its corresponding epipolar line in another image.
    Note that you should take the average of the geometric distance in two direction (image 1 to 2, and image 2 to 1)
    Arguments:
        all_matches: all matched keypoint pairs that loaded from disk (#all_matches, 4).
        F: estimated fundamental matrix, (3, 3)
    Returns:
        average geometric distance.
    """
    ones = np.ones((all_matches.shape[0], 1))
    all_p1 = np.concatenate((all_matches[:, 0:2], ones), axis=1)
    all_p2 = np.concatenate((all_matches[:, 2:4], ones), axis=1)
    # Epipolar lines.
    F_p1 = np.dot(F, all_p1.T).T  # F*p1, dims [#points, 3].
    F_p2 = np.dot(F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].

    # Geometric distances.
    p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
    p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
    d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
    d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]

    # Final distance.
    dist1 = d1.sum() / all_matches.shape[0]
    dist2 = d2.sum() / all_matches.shape[0]

    dist = (dist1 + dist2)/2
    return dist

# Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
# For each row, it consists (k1_x, k1_y, k2_x, k2_y).
# If necessary, you can convert float to int to get the integer coordinate
eight_good_matches = np.load('assets/eight_good_matches.npy')
all_good_matches = np.load('assets/all_good_matches.npy')

def estimate_fundamental_matrix(matches, normalize=False):
    """
    Arguments:
        matches: Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4).
        normalize: Boolean flag for using normalized or unnormalized alg.
    Returns:
        F: Fundamental matrix, dims (3, 3).
    """
    F = np.eye(3)
    # --------------------------- Begin your code here ---------------------------------------------
    A = []
    for i in range(matches.shape[0]):
        x1 = matches[i, 0]
        y1 = matches[i, 1]
        x2 = matches[i, 2]
        y2 = matches[i, 3]
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    A = np.array(A)
    u, s, vt = np.linalg.svd(A)  
    F = vt[len(vt) - 1].reshape(3,3)
    if normalize == True:
        uf, sf, vtf = np.linalg.svd(F)
        s_normalized = [sf[0], sf[1], 0] * np.eye(3)
        tmp = np.dot(uf, s_normalized)
        F_normalized = np.dot(tmp, vtf)
        F = F_normalized.reshape(3,3)
    # --------------------------- End your code here   ---------------------------------------------
    return F

F_with_normalization = estimate_fundamental_matrix(eight_good_matches, normalize=True)
F_without_normalization = estimate_fundamental_matrix(eight_good_matches, normalize=False)

# Evaluation (these numbers should be quite small)
print(f"F_with_normalization average geo distance: {calculate_geometric_distance(all_good_matches, F_with_normalization)}")
print(f"F_without_normalization average geo distance: {calculate_geometric_distance(all_good_matches, F_without_normalization)}")


# --------------------- Question 3

def ransac(all_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold):
    """
    Arguments:
        all_matches: coords of matched keypoint pairs in image 1 and 2, dims (# matches, 4).
        num_iteration: total number of RANSAC iteration
        estimate_fundamental_matrix: your eight-point algorithm function but use normalized version
        inlier_threshold: threshold to decide if a point is inlier
    Returns:
        best_F: best Fundamental matrix, dims (3, 3).
        inlier_matches_with_best_F: (#inliers, 4)
        avg_geo_dis_with_best_F: float
    """

    best_F = np.eye(3)
    inlier_matches_with_best_F = None
    avg_geo_dis_with_best_F = 0.0

    ite = 0
    # --------------------------- Begin your code here ---------------------------------------------
    number_all_matches = all_matches.shape[0]
    max_inlier_number = 0
    #while ite < num_iteration:
    while ite < num_iteration:
        # random sample correspondences
        eight_matches_index = random.sample(range(0, number_all_matches), 8)
        eight_matches = all_matches[eight_matches_index]
        # estimate the minimal fundamental estimation problem
        F = estimate_fundamental_matrix(eight_matches, normalize=True)
        # compute # of inliers
        ones = np.ones((all_matches.shape[0], 1))
        all_p1 = np.concatenate((all_matches[:, 0:2], ones), axis=1)
        all_p2 = np.concatenate((all_matches[:, 2:4], ones), axis=1)
        # Epipolar lines.
        F_p1 = np.dot(F, all_p1.T).T  # F*p1, dims [#points, 3].
        F_p2 = np.dot(F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].
        # Geometric distances.
        p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
        p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
        d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
        d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]
        dist = ((d1 + d2) / 2).flatten()
        current_inlier_number = np.count_nonzero(dist < inlier_threshold)
        if current_inlier_number > max_inlier_number:
            max_inlier_number = current_inlier_number
            best_F = F
            index = (dist < inlier_threshold).flatten()
            inlier_matches_with_best_F = all_matches[index, :]
        ite += 1
    avg_geo_dis_with_best_F = calculate_geometric_distance(all_matches, best_F)

    # --------------------------- End your code here   ---------------------------------------------
    return best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F

def visualize_inliers(im1, im2, inlier_coords):
    for i, im in enumerate([im1, im2]):
        plt.subplot(1, 2, i+1)
        plt.imshow(im, cmap='gray')
        plt.scatter(inlier_coords[:, 2*i], inlier_coords[:, 2*i+1], marker="x", color="red", s=10)
    plt.show()

num_iterations = [1, 100, 10000]
inlier_threshold = 0.01 # TODO: change the inlier threshold by yourself
for num_iteration in num_iterations:
    best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F = ransac(all_good_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold)
    if inlier_matches_with_best_F is not None:
        print(f"num_iterations: {num_iteration}; avg_geo_dis_with_best_F: {avg_geo_dis_with_best_F};")
        # visualize_inliers(img1, img2, inlier_matches_with_best_F)

# --------------------- Question 4

def visualize(estimated_F, img1, img2, kp1, kp2):
    # --------------------------- Begin your code here ---------------------------------------------
    ones = np.ones((kp1.shape[0], 1))
    all_p1 = np.concatenate((kp1, ones), axis=1)
    all_p2 = np.concatenate((kp2, ones), axis=1)
    # # Epipolar lines.
    F_p1 = np.dot(estimated_F, all_p1.T).T  # F*p1, dims [#points, 3].
    F_p2 = np.dot(estimated_F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].
    tmp1 = np.linalg.norm(F_p1, axis=1)[:, np.newaxis]
    line_img1 = F_p1 / tmp1
    tmp2 = np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
    line_img2 = F_p2 / tmp2

    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for line in line_img1:
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [img1.shape[1], -(line[2] + line[0] * img1.shape[1]) / line[1]])
        img1 = cv2.circle(img1, (x0, y0), 5, (0, 0, 255), 20)
        img1 = cv2.circle(img1, (x1, y1), 5, (0, 0, 255), 20)
        img1 = cv2.line(img1, (x0, y0), (x1, y1), (0, 255, 0), 1)
    for line in line_img2:
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [img2.shape[1], -(line[2] + line[0] * img2.shape[1]) / line[1]])
        img2 = cv2.circle(img2, (x0, y0), 5, (0, 0, 255), 20)
        img2 = cv2.circle(img2, (x1, y1), 5, (0, 0, 255), 20)
        img2 = cv2.line(img2, (x0, y0), (x1, y1), (0, 255, 0), 1)
    image = np.concatenate((img1, img2), axis=1) 
    cv2.imshow('Epipolar lines: img1(left), img2(right)', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --------------------------- End your code here   ---------------------------------------------
    pass

all_good_matches = np.load('assets/all_good_matches.npy')
F_Q2 = estimate_fundamental_matrix(eight_good_matches, normalize=True) # link to your estimated F in Q3
F_Q3 = ransac(all_good_matches, 10000, estimate_fundamental_matrix, inlier_threshold) # link to your estimated F in Q3
visualize(F_Q2, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])
# visualize(F_Q3, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])