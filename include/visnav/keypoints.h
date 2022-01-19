/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <bitset>
#include <set>
#include <stdexcept>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <pangolin/image/managed_image.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <visnav/common_types.h>
namespace keypoints_intern {
// Rotates a 2d vector by angle
Eigen::Vector2d rotate(const double angle, const Eigen::Vector2d& vector) {
  auto cos_of_angle = std::cos(angle);
  auto sin_of_angle = std::sin(angle);
  Eigen::Matrix2d rotation;
  rotation << cos_of_angle, -sin_of_angle, sin_of_angle, cos_of_angle;
  return rotation * vector;
}

int compute_hamming_distance(const std::bitset<256>& bitset_1,
                             const std::bitset<256>& bitset_2) {
  // XOR
  return (bitset_1 ^ bitset_2).count();
}

std::vector<std::pair<int, int>> match_implementation(
    const std::vector<std::bitset<256>>& corner_descriptors_1,
    const std::vector<std::bitset<256>>& corner_descriptors_2, int threshold,
    double dist_2_best) {
  std::vector<std::pair<int, int>> matches;

  // Iterate over the set of descriptors P
  for (size_t i = 0; i < corner_descriptors_1.size(); ++i) {
    // Saves the value of the minimal distance. The initial value is
    // descriptor_size + 1; is chosen such that the first distance value is
    // always smaller than this value
    int minimal_distance = 257;
    // Distance to the second best match
    int second_minimal_distance = 257;

    // Current index in set of descriptors Q
    int index_q_minimal;
    // Set of descriptors P
    const auto& bitset_p = corner_descriptors_1[i];

    for (int j = 0; j < static_cast<int>(corner_descriptors_2.size()); ++j) {
      // Set of descriptors Q
      const auto& bitset_q = corner_descriptors_2[j];
      auto distance = compute_hamming_distance(bitset_p, bitset_q);
      // Check if the distance between the current bitset_p and current bitset_q
      // is smaller than the previous distances
      if (minimal_distance >= distance) {
        second_minimal_distance = minimal_distance;
        minimal_distance = distance;
        index_q_minimal = j;

      } else {
        // Check if the distance between the current bitset_p and current
        // bitset_q is smaller than the current second smallest distance
        if (distance <= second_minimal_distance) {
          second_minimal_distance = distance;
        }
      }
    }
    // Discard matches with distance larger or equal to the threshold.

    if (minimal_distance < threshold) {
      /*
       * Discard matches if the distance to the second best match is smaller
       * than the smallest distance multiplied by dist 2 best.
       */
      if (minimal_distance * dist_2_best <= second_minimal_distance) {
        matches.emplace_back(i, index_q_minimal);
      }
    }
  }

  return matches;
}
}  // namespace keypoints_intern
namespace visnav {

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

typedef std::bitset<256> Descriptor;

constexpr int descriptor_size = 256;
char pattern_31_x_a[256] = {
    8,   4,   -11, 7,   2,   1,   -2,  -13, -13, 10,  -13, -11, 7,   -4,  -13,
    -9,  12,  -3,  -6,  11,  4,   5,   3,   -8,  -2,  -13, -7,  -4,  -10, 5,
    5,   1,   9,   4,   2,   -4,  -8,  4,   0,   -13, -3,  -6,  8,   0,   7,
    -13, 10,  -6,  10,  -13, -13, 3,   5,   -1,  3,   2,   -13, -13, -13, -7,
    6,   -9,  -2,  -12, 3,   -7,  -3,  2,   -11, -1,  5,   -4,  -9,  -12, 10,
    7,   -7,  -4,  7,   -7,  -13, -3,  7,   -13, 1,   2,   -4,  -1,  7,   1,
    9,   -1,  -13, 7,   12,  6,   5,   2,   3,   2,   9,   -8,  -11, 1,   6,
    2,   6,   3,   7,   -11, -10, -5,  -10, 8,   4,   -10, 4,   -2,  -5,  7,
    -9,  -5,  8,   -9,  1,   7,   -2,  11,  -12, 3,   5,   0,   -9,  0,   -1,
    5,   3,   -13, -5,  -4,  6,   -7,  -13, 1,   4,   -2,  2,   -2,  4,   -6,
    -3,  7,   4,   -13, 7,   7,   -7,  -8,  -13, 2,   10,  -6,  8,   2,   -11,
    -12, -11, 5,   -2,  -1,  -13, -10, -3,  2,   -9,  -4,  -4,  -6,  6,   -13,
    11,  7,   -1,  -4,  -7,  -13, -7,  -8,  -5,  -13, 1,   1,   9,   5,   -1,
    -9,  -1,  -13, 8,   2,   7,   -10, -10, 4,   3,   -4,  5,   4,   -9,  0,
    -12, 3,   -10, 8,   -8,  2,   10,  6,   -7,  -3,  -1,  -3,  -8,  4,   2,
    6,   3,   11,  -3,  4,   2,   -10, -13, -13, 6,   0,   -13, -9,  -13, 5,
    2,   -1,  9,   11,  3,   -1,  3,   -13, 5,   8,   7,   -10, 7,   9,   7,
    -1};

char pattern_31_y_a[256] = {
    -3,  2,   9,   -12, -13, -7,  -10, -13, -3,  4,   -8,  7,   7,   -5,  2,
    0,   -6,  6,   -13, -13, 7,   -3,  -7,  -7,  11,  12,  3,   2,   -12, -12,
    -6,  0,   11,  7,   -1,  -12, -5,  11,  -8,  -2,  -2,  9,   12,  9,   -5,
    -6,  7,   -3,  -9,  8,   0,   3,   7,   7,   -10, -4,  0,   -7,  3,   12,
    -10, -1,  -5,  5,   -10, -7,  -2,  9,   -13, 6,   -3,  -13, -6,  -10, 2,
    12,  -13, 9,   -1,  6,   11,  7,   -8,  -7,  -3,  -6,  3,   -13, 1,   -1,
    1,   -9,  -13, 7,   -5,  3,   -13, -12, 8,   6,   -12, 4,   12,  12,  -9,
    3,   3,   -3,  8,   -5,  11,  -8,  5,   -1,  -6,  12,  -2,  0,   -8,  -6,
    -13, -13, -8,  -11, -8,  -4,  1,   -6,  -9,  7,   5,   -4,  12,  7,   2,
    11,  5,   -4,  9,   -7,  5,   6,   6,   -10, 1,   -2,  -12, -13, 1,   -10,
    -13, 5,   -2,  9,   1,   -8,  -4,  11,  6,   4,   -5,  -5,  -3,  -12, -2,
    -13, 0,   -3,  -13, -8,  -11, -2,  9,   -3,  -13, 6,   12,  -11, -3,  11,
    11,  -5,  12,  -8,  1,   -12, -2,  5,   -1,  7,   5,   0,   12,  -8,  11,
    -3,  -10, 1,   -11, -13, -13, -10, -8,  -6,  12,  2,   -13, -13, 9,   3,
    1,   2,   -10, -13, -12, 2,   6,   8,   10,  -9,  -13, -7,  -2,  2,   -5,
    -9,  -1,  -1,  0,   -11, -4,  -6,  7,   12,  0,   -1,  3,   8,   -6,  -9,
    7,   -6,  5,   -3,  0,   4,   -6,  0,   8,   9,   -4,  4,   3,   -7,  0,
    -6};

char pattern_31_x_b[256] = {
    9,   7,  -8, 12,  2,   1,  -2,  -11, -12, 11,  -8,  -9,  12,  -3,  -12, -7,
    12,  -2, -4, 12,  5,   10, 6,   -6,  -1,  -8,  -5,  -3,  -6,  6,   7,   4,
    11,  4,  4,  -2,  -7,  9,  1,   -8,  -2,  -4,  10,  1,   11,  -11, 12,  -6,
    12,  -8, -8, 7,   10,  1,  5,   3,   -13, -12, -11, -4,  12,  -7,  0,   -7,
    8,   -4, -1, 5,   -5,  0,  5,   -4,  -9,  -8,  12,  12,  -6,  -3,  12,  -5,
    -12, -2, 12, -11, 12,  3,  -2,  1,   8,   3,   12,  -1,  -10, 10,  12,  7,
    6,   2,  4,  12,  10,  -7, -4,  2,   7,   3,   11,  8,   9,   -6,  -5,  -3,
    -9,  12, 6,  -8,  6,   -2, -5,  10,  -8,  -5,  9,   -9,  1,   9,   -1,  12,
    -6,  7,  10, 2,   -5,  2,  1,   7,   6,   -8,  -3,  -3,  8,   -6,  -5,  3,
    8,   2,  12, 0,   9,   -3, -1,  12,  5,   -9,  8,   7,   -7,  -7,  -12, 3,
    12,  -6, 9,  2,   -10, -7, -10, 11,  -1,  0,   -12, -10, -2,  3,   -4,  -3,
    -2,  -4, 6,  -5,  12,  12, 0,   -3,  -6,  -8,  -6,  -6,  -4,  -8,  5,   10,
    10,  10, 1,  -6,  1,   -8, 10,  3,   12,  -5,  -8,  8,   8,   -3,  10,  5,
    -4,  3,  -6, 4,   -10, 12, -6,  3,   11,  8,   -6,  -3,  -1,  -3,  -8,  12,
    3,   11, 7,  12,  -3,  4,  2,   -8,  -11, -11, 11,  1,   -9,  -6,  -8,  8,
    3,   -1, 11, 12,  3,   0,  4,   -10, 12,  9,   8,   -10, 12,  10,  12,  0};

char pattern_31_y_b[256] = {
    5,   -12, 2,   -13, 12,  6,   -4,  -8,  -9,  9,   -9,  12,  6,   0,  -3,
    5,   -1,  12,  -8,  -8,  1,   -3,  12,  -2,  -10, 10,  -3,  7,   11, -7,
    -1,  -5,  -13, 12,  4,   7,   -10, 12,  -13, 2,   3,   -9,  7,   3,  -10,
    0,   1,   12,  -4,  -12, -4,  8,   -7,  -12, 6,   -10, 5,   12,  8,  7,
    8,   -6,  12,  5,   -13, 5,   -7,  -11, -13, -1,  2,   12,  6,   -4, -3,
    12,  5,   4,   2,   1,   5,   -6,  -7,  -12, 12,  0,   -13, 9,   -6, 12,
    6,   3,   5,   12,  9,   11,  10,  3,   -6,  -13, 3,   9,   -6,  -8, -4,
    -2,  0,   -8,  3,   -4,  10,  12,  0,   -6,  -11, 7,   7,   12,  2,  12,
    -8,  -2,  -13, 0,   -2,  1,   -4,  -11, 4,   12,  8,   8,   -13, 12, 7,
    -9,  -8,  9,   -3,  -12, 0,   12,  -2,  10,  -4,  -13, 12,  -6,  3,  -5,
    1,   -11, -7,  -5,  6,   6,   1,   -8,  -8,  9,   3,   7,   -8,  8,  3,
    -9,  -5,  8,   12,  9,   -5,  11,  -13, 2,   0,   -10, -7,  9,   11, 5,
    6,   -2,  7,   -2,  7,   -13, -8,  -9,  5,   10,  -13, -13, -1,  -9, -13,
    2,   12,  -10, -6,  -6,  -9,  -7,  -13, 5,   -13, -3,  -12, -1,  3,  -9,
    1,   -8,  9,   12,  -5,  7,   -8,  -12, 5,   9,   5,   4,   3,   12, 11,
    -13, 12,  4,   6,   12,  1,   1,   1,   -13, -13, 4,   -2,  -3,  -2, 10,
    -9,  -1,  -2,  -8,  5,   10,  5,   5,   11,  -6,  -12, 9,   4,   -2, -2,
    -11};

void detectKeypoints_in_patch(const cv::Mat& patch,
                              const pangolin::ManagedImage<uint8_t>& img_raw,
                              KeypointsPositions& kd, int num_features,
                              FeaturePatchPair& fpp, PatchID patchID) {
  cv::Mat image(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);
  std::vector<cv::Point2f> points;
  goodFeaturesToTrack(patch, points, num_features, 0.01, 8);

  int x_increment = patchID % 2 == 0 ? img_raw.w : 0;
  int y_increment = patchID >= 3 ? img_raw.h : 0;

  FeatureId temp = kd.size();
  for (size_t i = 0; i < points.size(); i++) {
    if (img_raw.InBounds(points[i].x, points[i].y, EDGE_THRESHOLD)) {
      kd.emplace_back(points[i].x + x_increment, points[i].y + y_increment);
      fpp.insert(std::make_pair(temp, patchID));
      temp++;
    }
  }
  std::cout << "Size of kd: " << kd.size() << " and size of fpp: " << fpp.size()
            << std::endl;
}

void detectKeypoints(const pangolin::ManagedImage<uint8_t>& img_raw,
                     KeypointsData& kd, int num_features) {
  cv::Mat image(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);
  std::vector<cv::Point2f> points;
  goodFeaturesToTrack(image, points, num_features, 0.01, 8);

  kd.corners.clear();
  kd.corner_angles.clear();
  kd.corner_descriptors.clear();

  for (size_t i = 0; i < points.size(); i++) {
    if (img_raw.InBounds(points[i].x, points[i].y, EDGE_THRESHOLD)) {
      kd.corners.emplace_back(points[i].x, points[i].y);
    }
  }
}

void computeAngles(const pangolin::ManagedImage<uint8_t>& img_raw,
                   KeypointsData& kd, bool rotate_features) {
  kd.corner_angles.resize(kd.corners.size());

  for (size_t i = 0; i < kd.corners.size(); i++) {
    const Eigen::Vector2d& p = kd.corners[i];

    const int cx = p[0];
    const int cy = p[1];

    double angle = 0;
    // m_pq should be signed int because x and y can be negative and
    // img_raw(x_coordinate,y_cordinate) is an unsigned char
    int m_01 = 0;
    int m_10 = 0;
    if (rotate_features) {
      // Iterate over all points inside the square with side length PATCH_SIZE
      // because the circular patch around (cx, cy) is inscribed in this square
      for (auto x = -HALF_PATCH_SIZE; x <= HALF_PATCH_SIZE; ++x) {
        for (auto y = -HALF_PATCH_SIZE; y <= HALF_PATCH_SIZE; ++y) {
          // x and y are relative to (cx, cy)
          // compute the absolute coordinates of the pixel
          auto x_coordinate = cx + x;
          auto y_coordinate = cy + y;
          // Check if the current pixel is a valid one
          if (img_raw.InBounds(x_coordinate, y_coordinate)) {
            // Check if the current pixel is in the circular patch around (cx,
            // cy)
            if (x * x + y * y <= HALF_PATCH_SIZE * HALF_PATCH_SIZE) {
              m_10 += x * img_raw(x_coordinate, y_coordinate);
              m_01 += y * img_raw(x_coordinate, y_coordinate);
            }
          }
        }
      }
    }
    angle = std::atan2(m_01, m_10);
    kd.corner_angles[i] = angle;
  }
}

void computeDescriptors(const pangolin::ManagedImage<uint8_t>& img_raw,
                        KeypointsData& kd) {
  kd.corner_descriptors.resize(kd.corners.size());

  for (size_t i = 0; i < kd.corners.size(); i++) {
    std::bitset<256> descriptor;

    const Eigen::Vector2d& p = kd.corners[i];
    const double angle = kd.corner_angles[i];

    const int cx = p[0];
    const int cy = p[1];

    // TODO SHEET 3: compute descriptor

    for (auto i = 0; i < descriptor_size; ++i) {
      // Take the offset_a and offset_b around c and rotate them by angle
      auto p_a_rotated = keypoints_intern::rotate(
          angle, Eigen::Vector2d{pattern_31_x_a[i], pattern_31_y_a[i]});
      auto p_b_rotated = keypoints_intern::rotate(
          angle, Eigen::Vector2d{pattern_31_x_b[i], pattern_31_y_b[i]});
      // Get integer coordinates of the point with the rotated offset point p_a
      int im_x_a = cx + round(p_a_rotated[0]);
      int im_y_a = cy + round(p_a_rotated[1]);
      // Get integer coordinates of the point with the rotated offset point p_b
      int im_x_b = cx + round(p_b_rotated[0]);
      int im_y_b = cy + round(p_b_rotated[1]);

      unsigned char intensity_a = 0;
      unsigned char intensity_b = 0;
      // Is it possible for the point to be out of bounds?
      if (img_raw.InBounds(im_x_a, im_y_a)) {
        intensity_a = img_raw(im_x_a, im_y_a);
      } else
        throw std::runtime_error("Point is out of bouds!");
      if (img_raw.InBounds(im_x_b, im_y_b)) {
        intensity_b = img_raw(im_x_b, im_y_b);
      } else
        throw std::runtime_error("Point is out of bouds!");

      if (intensity_a < intensity_b) {
        descriptor[i] = 1;
      } else {
        descriptor[i] = 0;
      }
    }

    kd.corner_descriptors[i] = descriptor;
  }
}

void detectKeypointsAndDescriptors(
    const pangolin::ManagedImage<uint8_t>& img_raw, KeypointsData& kd,
    int num_features, bool rotate_features) {
  detectKeypoints(img_raw, kd, num_features);
  computeAngles(img_raw, kd, rotate_features);
  computeDescriptors(img_raw, kd);
}

void matchDescriptors(const std::vector<std::bitset<256>>& corner_descriptors_1,
                      const std::vector<std::bitset<256>>& corner_descriptors_2,
                      std::vector<std::pair<int, int>>& matches, int threshold,
                      double dist_2_best) {
  matches.clear();

  // Match set of descriptors P to Q
  auto matches_pq = keypoints_intern::match_implementation(
      corner_descriptors_1, corner_descriptors_2, threshold, dist_2_best);
  // Match set of descriptors Q to P
  auto matches_qp = keypoints_intern::match_implementation(
      corner_descriptors_2, corner_descriptors_1, threshold, dist_2_best);
  // Get only matches that can be found in both matches
  for (size_t i = 0; i < matches_pq.size(); ++i) {
    // index in set P from the matches from P to Q
    auto indexP_pq = matches_pq[i].first;
    // index in set Q from the matches from P to Q
    auto indexQ_pq = matches_pq[i].second;

    for (int j = 0; j < static_cast<int>(matches_qp.size()); ++j) {
      // index in set P from the matches from Q to P
      auto indexP_qp = matches_qp[j].second;
      // index in set Q from the matches from Q to P
      auto indexQ_qp = matches_qp[j].first;

      if (indexP_pq == indexP_qp) {
        if (indexQ_pq == indexQ_qp) {
          matches.emplace_back(indexP_pq, indexQ_pq);
        }
      }
    }
  }
}

}  // namespace visnav
