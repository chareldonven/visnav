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

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/keypoints.h>
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
void divide_image_into_patches(const pangolin::ManagedImage<uint8_t>& img_raw,
                               KeypointsPositions& kd, int num_features,
                               FeaturePatchPair& fpp) {
  kd.clear();
  fpp.clear();
  cv::Mat image(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);

  const auto rowrange_0 = cv::Range(0, image.rows / 2);
  const auto colrange_0 = cv::Range(0, image.cols / 2);

  const auto rowrange_1 = cv::Range(image.rows / 2, image.rows);
  const auto colrange_1 = cv::Range(image.cols / 2, image.cols);
  cv::Mat subimage1(image, rowrange_0, colrange_0);
  cv::Mat subimage2(image, rowrange_0, colrange_1);
  cv::Mat subimage3(image, rowrange_1, colrange_0);
  cv::Mat subimage4(image, rowrange_1, colrange_1);

  detectKeypoints_in_patch(subimage1, img_raw, kd, num_features, fpp, 1);
  detectKeypoints_in_patch(subimage2, img_raw, kd, num_features, fpp, 2);
  detectKeypoints_in_patch(subimage3, img_raw, kd, num_features, fpp, 3);
  detectKeypoints_in_patch(subimage4, img_raw, kd, num_features, fpp, 4);
}
void find_patchID(const pangolin::ManagedImage<uint8_t>& img_raw,
                  KeypointsPositions& kd, FeaturePatchPair& fpp) {}
bool check_threshold(const cv::Point2f& position0, const cv::Point2f& position1,
                     const int threshold) {
  return true;
}
void find_opticalflow_matches(FeaturePatchPair& fpp, KeypointsPositions& kdl,
                              KeypointsPositions& kdr,
                              const pangolin::ManagedImage<uint8_t>& img_raw_0,
                              const pangolin::ManagedImage<uint8_t>& img_raw_1,
                              MatchData& stereo_trackedPoints) {
  cv::Mat image_0(img_raw_0.h, img_raw_0.w, CV_8U, img_raw_0.ptr);
  cv::Mat image_1(img_raw_1.h, img_raw_1.w, CV_8U, img_raw_1.ptr);

  OpticalFlowPairs forward_tracking, backward_tracking;
  for (const auto& position : kdl) {
    // forward_tracking.source_points.emplace_back(position);
  }
  std::vector<uchar> status;
  std::vector<float> err;
  calcOpticalFlowPyrLK(image_0, image_1, forward_tracking.source_points,
                       forward_tracking.target_points, status, err);

  for (int i = 0; i < forward_tracking.target_points.size(); i++) {
    FeatureId featureID = i;
    if (!status[i]) {
      forward_tracking.outliers.emplace_back(featureID);
    } else {
      forward_tracking.inliers.emplace_back(featureID);
      backward_tracking.source_points.emplace_back(
          forward_tracking.target_points[i]);
    }
  }

  status.clear();
  err.clear();
  calcOpticalFlowPyrLK(image_1, image_0, backward_tracking.source_points,
                       backward_tracking.target_points, status, err);

  size_t last_index = 0;
  for (int k = 0; k < backward_tracking.target_points.size(); k++) {
    FeatureId featureID = forward_tracking.inliers[k];
    if (!status[k]) {
      backward_tracking.outliers.emplace_back(featureID);
    } else {
      backward_tracking.inliers.emplace_back(featureID);
      backward_tracking.target_points[last_index] =
          backward_tracking.target_points[k];
      last_index++;
    }
  }
  backward_tracking.target_points.resize(last_index);

  // Compare tracking_points to tracking_points_backward! Should we use a
  // threshold? 3  pixels -> 1

  for (size_t i = 0; i < backward_tracking.inliers.size(); i++) {
    for (size_t j = 0; j < forward_tracking.inliers.size(); j++) {
      if (backward_tracking.inliers[i] == forward_tracking.inliers[j]) {
        const FeatureId& featureID = forward_tracking.inliers[j];
        if (check_threshold(backward_tracking.target_points[i],
                            forward_tracking.source_points[featureID], 3)) {
          /*
        stereo_trackedPoints.left_image.emplace_back(
            forward_tracking.source_points[featureID]);
        stereo_trackedPoints.right_image.emplace_back(
            forward_tracking.target_points[featureID]);
            */
        }
      }
    }
  }
}
/// Stereo:
void add_new_landmarks(Landmarks& landmarks, MatchData& stereo_trackedPoints,
                       const Corners& feature_corners, FeaturePatchPair& fpp,
                       TrackedPoints& trackedPoints) {
  // Also update trackedpoints
}
/// Frame to frame: füge neue obs hinzu
void update_landmarks(const std::vector<TrackId>& projected_track_ids,
                      Landmarks& landmarks, TrackedPoints& trackedPoints,
                      const FrameCamId fcidl) {}
void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.

  // First transform landmarks to camera coordinates
  for (const auto& landmark : landmarks) {
    // Get 3D coordinates in camera reference
    auto landmark_3D_world = landmark.second.p;
    auto landmark_3D_camera = current_pose.inverse() * landmark_3D_world;

    // Check all the given constraints
    auto z_coordinate = landmark_3D_camera.z();
    // Check if the point is behing the camera
    if (z_coordinate >= cam_z_threshold) {
      auto projected_landmark = cam->project(landmark_3D_camera);

      // Get width and height of the image
      const auto& image_width = cam->width();
      const auto& image_height = cam->height();
      // Check whether the projected point is outside the image
      if (0 <= projected_landmark.x() &&
          projected_landmark.x() <= image_width &&
          0 <= projected_landmark.y() &&
          projected_landmark.y() <= image_height) {
        // Save the 2D location and TrackID of the landmark the point originates
        // from
        const auto& track_id = landmark.first;

        projected_points.emplace_back(projected_landmark);
        projected_track_ids.emplace_back(track_id);
      }
    }
  }
}

void localize_camera(const Sophus::SE3d& current_pose,
                     const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     LandmarkMatchData& md) {
  md.inliers.clear();

  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

  // TODO SHEET 5: Find the pose (md.T_w_c) and the inliers (md.inliers) using
  // the landmark to keypoints matches and PnP. This should be similar to the
  // localize_camera in exercise 4 but in this exercise we don't explicitly have
  // tracks.

  opengv::bearingVectors_t bearing_vectors;
  opengv::points_t points;

  for (const auto& match : md.matches) {
    auto landmark = landmarks.at(match.second);
    auto point3d = landmark.p;
    points.push_back(point3d);

    auto corner_point2d = kdl.corners[match.first];
    auto corner_point3d = cam->unproject(corner_point2d);
    bearing_vectors.push_back(corner_point3d.normalized());
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors,
                                                        points);

  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));

  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel / 500.0));
  ransac.max_iterations_ = 100;

  ransac.computeModel();
  const auto& translation = ransac.model_coefficients_.topRightCorner(3, 1);
  const auto& rotation = ransac.model_coefficients_.topLeftCorner(3, 3);
  adapter.setR(rotation);
  adapter.sett(translation);

  const opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac.threshold_, ransac.inliers_);

  const auto& refined_tranlation =
      nonlinear_transformation.topRightCorner(3, 1);
  const auto& refined_rotation = nonlinear_transformation.topLeftCorner(3, 3);

  md.T_w_c = Sophus::SE3d(refined_rotation, refined_tranlation);
  for (size_t i = 0; i < ransac.inliers_.size(); ++i) {
    md.inliers.push_back(md.matches[ransac.inliers_[i]]);
  }
}

void remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  // TODO SHEET 5: Remove old cameras and observations if the number of
  // keyframe pairs (left and right image is a pair) is larger than
  // max_num_kfs. The ids of all the keyframes that are currently in the
  // optimization should be stored in kf_frames. Removed keyframes should be
  // removed from cameras and landmarks with no left observations should be
  // moved to old_landmarks.

  // Is this the right condition? This the condition the test checks!
  while (max_num_kfs < int(kf_frames.size())) {
    // Assume the first is the oldest just like in the odometry.cpp
    const auto& frameID = *kf_frames.begin();
    // Remove old keyframes from cameras
    for (const auto& camera : cameras) {
      if (camera.first.frame_id == frameID) {
        cameras.erase(cameras.find(camera.first));
      }
    }
    // Remove old keyframes from current keyframes
    kf_frames.erase(kf_frames.begin());
  }

  Landmarks new_landmarks;
  for (auto& landmark : landmarks) {
    auto& observations = landmark.second.obs;
    FeatureTrack new_observations;
    // save the correct observations in new_observations
    for (auto& observation : observations) {
      for (const auto& frameID : kf_frames) {
        if (observation.first.frame_id == frameID) {
          new_observations.emplace(observation);
        }
      }
    }
    // First clear and then save new_observations to observations
    observations.clear();

    for (auto& observation : new_observations) {
      observations.emplace(observation);
    }

    // Move landmarks with no observations to old_landmarks
    // Save temporarly landmarks with observations in new_landmarks
    if (!landmark.second.obs.empty()) {
      new_landmarks.emplace(landmark);
    } else {
      old_landmarks.emplace(landmark);
    }
  }
  // First clear and then save new_landmarks to landmarks
  landmarks.clear();
  for (auto& landmark : new_landmarks) {
    landmarks.emplace(landmark);
  }
}
}  // namespace visnav
