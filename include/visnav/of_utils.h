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

#include <set>
#include <map>
#include <visnav/common_types.h>

#include <visnav/calibration.h>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/serialization.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/tracking.hpp>
using namespace cv;

namespace visnav {
// https://learnopencv.com/cropping-an-image-using-opencv/
void divide_image_into_patches() {
  /*
   * Given an image, divide it into patches. Number patches? Parameter or always
   * 4
   *
   * Call detect_keypoints from keypoint.h
   * for each patch
   * Define a pair: patch and keypoints in patch
   * keypoints is determined from FeatureID  or smth like KeypointData
   *
   * Update OpticalFlowData
   * */
}

void filter_optical_flow(std::vector<uchar> status,
                         std::vector<Point2f> tracking_points[2],
                         std::vector<Point2f> tracking_points_backward[2]) {
  size_t i, k;
  for (i = k = 0; i < tracking_points[1].size(); i++) {
    if (!status[i]) continue;
    k++;
    tracking_points[1][k] = tracking_points[1][i];
    tracking_points_backward[0][k] = tracking_points[1][i];
  }
  tracking_points[1].resize(k);
  tracking_points_backward[0].resize(k);
}
/*
 * Remove tracking_points
 * Add opticalFlowData
 *
 * */
void find_optical_flow(std::vector<Point2f> tracking_points[2],
                       cv::Mat image_previous, cv::Mat image_next) {
  std::vector<uchar> status;
  std::vector<float> err;
  calcOpticalFlowPyrLK(image_previous, image_next, tracking_points[0],
                       tracking_points[1], status, err);
  std::vector<Point2f> tracking_points_backward[2];
  tracking_points_backward[0].resize(tracking_points[1].size());

  filter_optical_flow(status, tracking_points, tracking_points_backward);

  status.clear();
  err.clear();
  calcOpticalFlowPyrLK(image_next, image_previous, tracking_points_backward[0],
                       tracking_points_backward[1], status, err);

  // /!\ remove tracking_points_backward as 3. parameter!

  filter_optical_flow(status, tracking_points_backward,
                      tracking_points_backward);

  // Compare tracking_points to tracking_points_backward! Should we use a
  // threshold? 3  pixels -> 1
}

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {}

void localize_camera(const Sophus::SE3d& current_pose,
                     const std::shared_ptr<AbstractCamera<double>>& cam,
                     const double ransac_thresh,
                     std::vector<Point2f> tracking_points[2],
                     Sophus::SE3d& T_i_j) {
  opengv::bearingVectors_t vectors_P;
  opengv::bearingVectors_t vectors_Q;

  for (size_t i = 0; i < tracking_points[0].size(); i++) {
    // Get the respective 2D  points
    auto p_2d_x = tracking_points[0][i].x;
    auto p_2d_y = tracking_points[0][i].y;

    auto q_2d_x = tracking_points[1][i].x;
    auto q_2d_y = tracking_points[1][i].y;

    Eigen::Vector2d p_2d(p_2d_x, p_2d_y);
    Eigen::Vector2d q_2d(q_2d_x, q_2d_y);

    // Find the respective 3D point to the corner point
    auto p_3d_P = cam->unproject(p_2d);
    auto p_3d_Q = cam->unproject(q_2d);

    vectors_P.emplace_back(p_3d_P);
    vectors_Q.emplace_back(p_3d_Q);
  }

  // Define the central relative adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(vectors_P, vectors_Q);
  // Create a Ransac object
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  // Create a CentralRelativePoseSacProblem with STEWENIUS 5 point algorithm
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::STEWENIUS));
  // Run Ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  // What is a good number of iterations?
  ransac.max_iterations_ = 10;
  ransac.computeModel();
  // Get the results:
  const auto& translation = ransac.model_coefficients_.topRightCorner(3, 1);
  const auto& rotation = ransac.model_coefficients_.topLeftCorner(3, 3);

  adapter.sett12(translation);
  adapter.setR12(rotation);
  // Nonlinear optimization to refine the model parameters using ALL inliers
  const opengv::transformation_t nonlinear_transformation =
      opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);
  // Update the set of inliers using the refined relative pose
  // Select the inliers that are within threshold from the model
  // sac_model_->selectWithinDistance( model_coefficients, threshold_, inliers
  // );
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac_thresh, ransac.inliers_);
  // Get refined pose
  // Normalize translation vector
  const auto& refined_tranlation =
      nonlinear_transformation.topRightCorner(3, 1).normalized();
  const auto& refined_rotation = nonlinear_transformation.topLeftCorner(3, 3);

  // Store the final refined relative pose.
  T_i_j = Sophus::SE3d(refined_rotation, refined_tranlation);
}

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  /*
   *
   * kdl saves the tracked keypoints in left stereo frame
   * kdr saves the tracked keypoints in right stereo frame
   *
   * triangulate to compute the 3D positions and add new landmark.
   *
   * /!\ There are duplicates!
   * */
}

void remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          std::set<FrameId>& kf_frames) {}
}  // namespace visnav
