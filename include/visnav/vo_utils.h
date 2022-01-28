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
namespace vo_intern {
int compute_hamming_distance(const std::bitset<256>& bitset_1,
                             const std::bitset<256>& bitset_2) {
  // XOR
  return (bitset_1 ^ bitset_2).count();
}
unsigned long int find_index_best_match(
    const std::bitset<256>& given_descriptor,
    const std::vector<std::bitset<256>>& descriptors, int threshold,
    double dist_2_best) {
  unsigned long int index_of_matched_descriptor = descriptors.size();

  // Saves the value of the minimal distance. The initial value is
  // descriptor_size + 1; is chosen such that the first distance value is
  // always smaller than this value
  int minimal_distance = 257;
  // Distance to the second best match
  int second_minimal_distance = 257;

  // Current index in set of descriptors Q
  unsigned long int index_minimal_distance = descriptors.size();

  for (int j = 0; j < static_cast<int>(descriptors.size()); ++j) {
    const auto& current_element_descriptors = descriptors[j];
    auto distance =
        compute_hamming_distance(given_descriptor, current_element_descriptors);
    // Check if the distance between the current_descriptor and current
    // element in descriptors is smaller than the previous distances
    if (minimal_distance >= distance) {
      second_minimal_distance = minimal_distance;
      minimal_distance = distance;
      index_minimal_distance = j;

    } else {
      // Check if the distance between the current_descriptor and
      // current element in descriptors is smaller than the current second
      // smallest distance
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
      index_of_matched_descriptor = index_minimal_distance;
    }
  }

  return index_of_matched_descriptor;
}
unsigned long int find_index_of_minimal_distance_descriptor(
    const std::bitset<256>& given_descriptor,
    const std::vector<std::bitset<256>>& descriptors) {
  int minimal_distance = 257;
  unsigned long int index_minimal_distance = descriptors.size();

  for (int j = 0; j < static_cast<int>(descriptors.size()); ++j) {
    const auto& current_descriptor = descriptors[j];
    auto distance =
        compute_hamming_distance(given_descriptor, current_descriptor);
    // Check if the distance between the given descriptor and current
    // descriptor is smaller than the previous distances
    if (minimal_distance >= distance) {
      minimal_distance = distance;
      index_minimal_distance = j;
    }
  }

  return index_minimal_distance;
}

}  // namespace vo_intern
namespace visnav {

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

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_threshold,
    const double feature_match_dist_2_best, LandmarkMatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location.
  // For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_threshold and feature_match_dist_2_best
  // should be used to filter outliers the same way as in exercise 3. You should
  // fill md.matches with <featureId,trackId> pairs for the successful matches
  // that pass all tests.

  for (size_t keypoint_index = 0; keypoint_index < kdl.corners.size();
       keypoint_index++) {
    const auto& keypoint = kdl.corners[keypoint_index];

    const auto& current_descriptor = kdl.corner_descriptors[keypoint_index];

    std::vector<std::bitset<256>> possible_descriptors;
    std::vector<TrackId> possible_trackIDs;

    for (size_t landmark_index = 0; landmark_index < projected_points.size();
         landmark_index++) {
      const auto& projected_landmark = projected_points[landmark_index];
      std::vector<std::bitset<256>> landmarks_possible_descriptors;

      const auto& x = std::abs(projected_landmark.x() - keypoint.x());
      const auto& y = std::abs(projected_landmark.y() - keypoint.y());

      // Search for matches inside the circle with radius match_max_dist_2d
      if (x * x + y * y <= match_max_dist_2d * match_max_dist_2d) {
        const auto& landmark_trackID = projected_track_ids[landmark_index];
        const auto& observations = landmarks.at(landmark_trackID).obs;
        // Find the distance of the landmark by finding the best match between
        // the current descriptor and all observations of the landmark

        // Save all observed descriptors of the landmark in
        // landmarks_possible_descriptors
        for (const auto& current_observation : observations) {
          const auto& featureID = current_observation.second;
          const auto& frameCamID = current_observation.first;
          const auto& descriptor =
              feature_corners.at(frameCamID).corner_descriptors[featureID];
          landmarks_possible_descriptors.emplace_back(descriptor);
        }

        // Find the index in landmarks_possible_descriptors of the best match
        const auto& index_minimal_descriptor =
            vo_intern::find_index_of_minimal_distance_descriptor(
                current_descriptor, landmarks_possible_descriptors);
        // Check whether the index is valid --> this is not really neccessary
        if (index_minimal_descriptor < landmarks_possible_descriptors.size()) {
          const auto& minimal_descriptor =
              landmarks_possible_descriptors.at(index_minimal_descriptor);
          // Save the best descriptor of the possible matches for the keypoint
          // in possible_descriptors
          possible_descriptors.emplace_back(minimal_descriptor);
          possible_trackIDs.emplace_back(landmark_trackID);
        }
      }
    }
    // Find the best landmark match for the keypoint in the possible_descriptors
    const auto& index_best_landmark_descriptor =
        vo_intern::find_index_best_match(
            current_descriptor, possible_descriptors, feature_match_threshold,
            feature_match_dist_2_best);

    // Check whether the index is valid --> this is not really neccessary
    if (index_best_landmark_descriptor < possible_descriptors.size()) {
      md.matches.emplace_back(
          keypoint_index, possible_trackIDs.at(index_best_landmark_descriptor));
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

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains feature to landmark
  // matches for the left camera (camera 0). For all inlier feature to landmark
  // matches add the observations to the existing landmarks. If the left
  // camera's feature appears also in md_stereo.inliers, then add both
  // observations. For all inlier stereo observations that were not added to the
  // existing landmarks, triangulate and add new landmarks. Here
  // next_landmark_id is a running index of the landmarks, so after adding a new
  // landmark you should always increase next_landmark_id by 1.

  std::vector<std::pair<FeatureId, FeatureId>> other_stereo_points;
  // For all inlier landmark matches
  for (const auto& match : md.inliers) {
    for (size_t i = 0; i < md_stereo.inliers.size(); i++) {
      const auto& stereo_l_featureID = md_stereo.inliers[i].first;

      const auto& match_trackID = match.second;
      const auto& match_feature_id = match.first;
      // Add observations for the left camera to the existing landmarks
      landmarks.at(match_trackID).obs.emplace(fcidl, match_feature_id);
      const auto& stereo_r_featureID = md_stereo.inliers[i].second;
      // If the keypoint appears also in md_stereo.inliers
      if (match.first == stereo_l_featureID) {
        // Add an observation for the right camera as well
        landmarks.at(match_trackID).obs.emplace(fcidr, stereo_r_featureID);
      }
    }
  }
  /*
   * Find inlier stereo matches that were not used in the previous stage
   *
   */
  for (size_t i = 0; i < md_stereo.inliers.size(); i++) {
    bool is_not_used = true;
    for (const auto& match : md.inliers) {
      if (match.first == md_stereo.inliers[i].first) {
        is_not_used = false;
      }
    }
    if (is_not_used) {
      other_stereo_points.emplace_back(md_stereo.inliers[i].first,
                                       md_stereo.inliers[i].second);
    }
  }

  opengv::bearingVectors_t vectors_0;
  opengv::bearingVectors_t vectors_1;
  // Prepare bearing vectors for the adapter
  for (size_t i = 0; i < other_stereo_points.size(); i++) {
    const auto& stereo_l_featureID = other_stereo_points[i].first;
    const auto& stereo_r_featureID = other_stereo_points[i].second;

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
  for (size_t i = 0; i < other_stereo_points.size(); i++) {
    const auto& triangulated_point =
        opengv::triangulation::triangulate(adapter, i);
    next_landmark_id = next_landmark_id + 1;
    const auto& new_trackID = next_landmark_id;
    landmarks[new_trackID].p = md.T_w_c * triangulated_point;
    landmarks.at(new_trackID).obs.emplace(fcidl, other_stereo_points[i].first);
    landmarks.at(new_trackID).obs.emplace(fcidr, other_stereo_points[i].second);
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
