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
/// This method saves in kdl.corners the keypoints of the left stereo image.
/// It also saves in fpp the pair (featureID, patchID)
/// It saves all points from trackedPoints as keypoints and it finds new
/// keypoints in the image

void divide_image_into_patches(const pangolin::ManagedImage<uint8_t>& img_raw,
                               KeypointsPositions& kd, int num_features,
                               FeaturePatchPair& fpp,
                               const TrackedPoints& trackedPoints,
                               const Corners& corners, const FrameCamId fcidl) {
  kd.clear();
  fpp.clear();
  // It saves all points from trackedPoints as keypoints

  for (const auto& trackedPoint : trackedPoints) {
    const auto& featureID = trackedPoint.featureID_current_frame;
    kd.emplace_back(corners.at(fcidl).corners.at(featureID));
    // This should not be neccessary
    fpp.emplace(featureID, trackedPoint.patchID_current_frame);
  }

  // It finds new keypoints in the image
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

/// This methode returns the corresponding patch to a given point in an image
PatchID find_patchID(const pangolin::ManagedImage<uint8_t>& img_raw,
                     cv::Point2f point) {
  PatchID result = 0;
  if (point.x > img_raw.w / 2) {
    result = result + 1;
  }
  if (point.y > img_raw.h / 2) {
    result = result + 2;
  }
  return result;
}

/// This Methode checks if two given points a close enough to each other
bool check_threshold(const cv::Point2f& position0, const cv::Point2f& position1,
                     const int threshold) {
  double norm = cv::norm(cv::Mat(position0), cv::Mat(position1));
  return norm < threshold;
}

/// This method finds optical flow matches
/// It saves the 2d positions of the right/next frame in kdr
/// It does not change kdl
/// It saves the ids of matches in stereo_trackedPoints
void find_opticalflow_stereo_matches(
    FeaturePatchPair& fpp, KeypointsPositions& kdl, KeypointsPositions& kdr,
    const pangolin::ManagedImage<uint8_t>& img_raw_0,
    const pangolin::ManagedImage<uint8_t>& img_raw_1,
    MatchData& stereo_trackedPoints) {
  cv::Mat image_0(img_raw_0.h, img_raw_0.w, CV_8U, img_raw_0.ptr);
  cv::Mat image_1(img_raw_1.h, img_raw_1.w, CV_8U, img_raw_1.ptr);

  OpticalFlowPairs forward_tracking, backward_tracking;
  // alle kp von allen position speichern in kdl
  for (const auto& position : kdl) {
    cv::Point2f points;
    points.x = position[0];
    points.y = position[1];
    forward_tracking.source_points.emplace_back(points);
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
          // sourcePoints: cv::Point2f
          // Eigen::Vector2d
          Eigen::Vector2d forward_point;
          forward_point[0] = forward_tracking.source_points[featureID].x;
          forward_point[1] = forward_tracking.source_points[featureID].y;
          kdl.emplace_back(forward_point);

          Eigen::Vector2d backward_point;
          backward_point[0] = backward_tracking.source_points[featureID].x;
          backward_point[1] = backward_tracking.source_points[featureID].y;
          kdr.emplace_back(backward_point);

          stereo_trackedPoints.inliers.emplace_back(
              std::make_pair(featureID, featureID));
          stereo_trackedPoints.matches.emplace_back(
              std::make_pair(featureID, featureID));

          PatchID pID = find_patchID(
              img_raw_1, backward_tracking.source_points[featureID]);

          fpp[featureID] = pID;
        }
      }
    }
  }
}

/// This method computes frame to frame optical flow
/// It save the new positions in the next frame of the tracked points in kdr
/// It also finds the new patchID of the tracked points
/// And the FeatureID in the next frame
/// It saves the new feature-patch pairs in fpp
void find_opticalflow_matches(FeaturePatchPair& fpp, KeypointsPositions& kdl,
                              KeypointsPositions& kdr,
                              const pangolin::ManagedImage<uint8_t>& img_raw_0,
                              const pangolin::ManagedImage<uint8_t>& img_raw_1,
                              TrackedPoints& trackedPoints) {}

/// This Method checks if all patch contains enough points
bool enough_points_in_patch(const FeaturePatchPair& fpp,
                            int min_points_per_patch) {
  int patch_counter[4];
  for (const auto elem : fpp) {
    patch_counter[elem.second - 1] += 1;
  }

  bool res = false;

  for (int i = 0; i < 4; i++) {
    res = res || (patch_counter[i] < min_points_per_patch);
  }

  return !res;
}

/// This checks if sterio matching is nessesary.
/// In the First frame it has to do it in the following enough_points_in_patch
/// is called
bool track_into_stereo(int current_frame, const FeaturePatchPair& fpp,
                       int min_points_per_patch) {
  if (current_frame == 0) return true;
  return enough_points_in_patch(fpp, min_points_per_patch);
}

/// Add new observations to the existing landmarks and add new landmarks to the
/// map
void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       Landmarks& landmarks, MatchData& stereo_trackedPoints,
                       FeaturePatchPair& fpp, TrackedPoints& trackedPoints,
                       const Calibration& calib_cam,
                       const Sophus::SE3d& current_pose,
                       TrackId& next_landmark_id) {
  // Also update trackedpoints

  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  opengv::bearingVectors_t vectors_0;
  opengv::bearingVectors_t vectors_1;
  // Prepare bearing vectors for the adapter
  for (size_t i = 0; i < stereo_trackedPoints.inliers.size(); i++) {
    const auto& stereo_l_featureID = stereo_trackedPoints.inliers[i].first;
    const auto& stereo_r_featureID = stereo_trackedPoints.inliers[i].second;

    const auto& point2D_0 = kdl.corners.at(stereo_l_featureID);
    const auto& point2D_1 = kdr.corners.at(stereo_r_featureID);

    auto point3D_0 =
        calib_cam.intrinsics.at(fcidl.cam_id)->unproject(point2D_0);
    auto point3D_1 =
        calib_cam.intrinsics.at(fcidr.cam_id)->unproject(point2D_1);

    vectors_0.emplace_back(point3D_0.normalized());
    vectors_1.emplace_back(point3D_1.normalized());
  }

  opengv::relative_pose::CentralRelativeAdapter adapter(vectors_0, vectors_1);

  adapter.setR12(R_0_1);
  adapter.sett12(t_0_1);
  // Triangulate new landmarks from stereo matches
  for (size_t i = 0; i < stereo_trackedPoints.inliers.size(); i++) {
    const auto& triangulated_point =
        opengv::triangulation::triangulate(adapter, i);
    next_landmark_id = next_landmark_id + 1;
    const auto& new_trackID = next_landmark_id;
    landmarks[new_trackID].p = current_pose * triangulated_point;
    landmarks.at(new_trackID)
        .obs.emplace(fcidl, stereo_trackedPoints.inliers[i].first);
    landmarks.at(new_trackID)
        .obs.emplace(fcidr, stereo_trackedPoints.inliers[i].second);
    // feature id of the left image

    OpticalFlowData new_point;
    new_point.trackID = new_trackID;
    new_point.featureID_current_frame = stereo_trackedPoints.inliers[i].first;
    new_point.trackID = fpp.at(new_point.featureID_current_frame);
    // Find new patch
    trackedPoints.emplace_back(new_point);
  }
  // Add matchdata in md
  // Update tracked points
}

/// Frame to frame: add new observations to landmarks and fill md.matches
void update_landmarks(Landmarks& landmarks, TrackedPoints& trackedPoints,
                      const FrameCamId fcidl, LandmarkMatchData& md) {
  for (const auto& trackedPoint : trackedPoints) {
    const auto& trackID = trackedPoint.trackID;
    // Find corresponding landmark
    auto landmark = landmarks.at(trackID);

    landmark.obs.emplace(fcidl, trackedPoint.featureID_current_frame);
  }
}

/// Not changed for this project
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

/// Not changed for this project
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

/// Not changed for this project
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
