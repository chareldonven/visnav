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
#include <opencv2/video/tracking.hpp>

using namespace cv;

namespace visnav {
/// This method returns the corresponding patch to a given point in an image
PatchID find_patchID(const pangolin::ManagedImage<uint8_t>& img_raw,
                     cv::Point2f point) {
  cv::Mat img(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);

  PatchID result = 1;
  if (point.x > img.cols / 2) {
    result = result + 1;
  }
  if (point.y > img.rows / 2) {
    result = result + 2;
  }
  return result;
}

/// This method checks if all patches contain enough points
bool not_enough_points_in_patch(
    const std::vector<std::pair<FeatureId, TrackId>>& inliers,
    const KeypointsPositions& kd, const pangolin::ManagedImage<uint8_t>& img,
    const unsigned int min_points_per_patch) {
  unsigned int patch_counter[4];
  for (size_t i = 0; i < 4; i++) {
    patch_counter[i] = 0;
  }

  for (const auto& pair : inliers) {
    const auto& featureID = pair.first;
    Point2f point;
    point.x = kd[featureID].x();
    point.y = kd[featureID].y();
    const auto& patchID = find_patchID(img, point);
    patch_counter[patchID - 1] += 1;
  }

  bool not_enough = false;

  for (int i = 0; i < 4; i++) {
    not_enough = not_enough || (patch_counter[i] < min_points_per_patch);
  }

  return not_enough;
}

/// This checks if sterio matching is nessesary.
/// In the first frame it has to do it in the following enough_points_in_patch
/// is called
bool should_track_into_stereo(
    const int current_frame,
    const std::vector<std::pair<FeatureId, TrackId>>& inliers,
    const KeypointsPositions& kd, const pangolin::ManagedImage<uint8_t>& img,
    const int min_points_per_patch) {
  if (current_frame == 0) return true;
  return not_enough_points_in_patch(inliers, kd, img, min_points_per_patch);
}

void find_keypoints_in_patch(const cv::Mat& patch,
                             const pangolin::ManagedImage<uint8_t>& img_raw,
                             KeypointsPositions& kd, const int num_features,
                             const PatchID patchID) {
  cv::Mat image(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);
  std::vector<cv::Point2f> points;
  goodFeaturesToTrack(patch, points, num_features, 0.01, 8);
  // Find real coordinates of point!!!

  const int x_increment = patchID % 2 == 0 ? patch.cols : 0;
  const int y_increment = patchID >= 3 ? patch.rows : 0;
  for (size_t i = 0; i < points.size(); i++) {
    const auto& x = points[i].x + x_increment;
    const auto& y = points[i].y + y_increment;
    if (img_raw.InBounds(x, y, EDGE_THRESHOLD)) {
      kd.emplace_back(x, y);
    }
  }
}
void detect_keypoints(const pangolin::ManagedImage<uint8_t>& img_raw,
                      KeypointsPositions& kd, const int num_features) {
  // Divide image into patches

  cv::Mat image(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);

  const auto rowrange_0 = cv::Range(0, image.rows / 2);
  const auto colrange_0 = cv::Range(0, image.cols / 2);

  const auto rowrange_1 = cv::Range(image.rows / 2, image.rows);
  const auto colrange_1 = cv::Range(image.cols / 2, image.cols);
  cv::Mat subimage1(image, rowrange_0, colrange_0);
  cv::Mat subimage2(image, rowrange_0, colrange_1);
  cv::Mat subimage3(image, rowrange_1, colrange_0);
  cv::Mat subimage4(image, rowrange_1, colrange_1);

  // Find keypoints in each of the subimages

  find_keypoints_in_patch(subimage1, img_raw, kd, num_features, 1);
  find_keypoints_in_patch(subimage2, img_raw, kd, num_features, 2);
  find_keypoints_in_patch(subimage3, img_raw, kd, num_features, 3);
  find_keypoints_in_patch(subimage4, img_raw, kd, num_features, 4);
}
void get_kd_from_trackedPoints(const TrackedPoints& trackedPoints,
                               const KeypointsData& old_kdl,
                               KeypointsData& new_kdl) {
  for (const auto& trackedPoint : trackedPoints) {
    const auto& featureID = trackedPoint.first;
    const auto& position = old_kdl.corners.at(featureID);
    const auto& angle = old_kdl.corner_angles.at(featureID);
    new_kdl.corners.emplace_back(position);
    new_kdl.corner_angles.emplace_back(angle);
  }
}
void get_information_from_trackedPoints(const TrackedPoints& trackedPoints,
                                        const KeypointsData& old_kdl,
                                        KeypointsData& new_kdl,
                                        std::vector<TrackId>& trackIDs) {
  for (const auto& trackedPoint : trackedPoints) {
    const auto& featureID = trackedPoint.first;
    const auto& position = old_kdl.corners.at(featureID);
    const auto& angle = old_kdl.corner_angles.at(featureID);

    new_kdl.corners.emplace_back(position);
    new_kdl.corner_angles.emplace_back(angle);
    const auto& trackID = trackedPoint.second;
    trackIDs.emplace_back(trackID);
  }
}
bool check_threshold(const cv::Point2f& position0, const cv::Point2f& position1,
                     const int threshold) {
  double norm = cv::norm(cv::Mat(position0), cv::Mat(position1));
  return norm < threshold;
}

void match_stereo_with_opticalflow(
    const KeypointsPositions& kdl, KeypointsPositions& kdr,
    const pangolin::ManagedImage<uint8_t>& img_raw_l,
    const pangolin::ManagedImage<uint8_t>& img_raw_r,
    MatchData& stereo_matches) {
  /// Track from left to right frame
  /// kdl.corners should not change! kdr.corners contains new positions!
  /// md_stereo.matches contains forward and backward tracking inliers!
  ///

  cv::Mat image_l(img_raw_l.h, img_raw_l.w, CV_8U, img_raw_l.ptr);
  cv::Mat image_r(img_raw_r.h, img_raw_r.w, CV_8U, img_raw_r.ptr);
  std::vector<cv::Point2f> left_keypoints_positions;
  for (const auto& left_keypoint_position : kdl) {
    cv::Point2f point;
    point.x = left_keypoint_position.x();
    point.y = left_keypoint_position.y();
    left_keypoints_positions.emplace_back(point);
  }
  std::vector<cv::Point2f> all_right_keypoints_positions;

  std::vector<uchar> status_forward;
  std::vector<float> err_forward;

  calcOpticalFlowPyrLK(image_l, image_r, left_keypoints_positions,
                       all_right_keypoints_positions, status_forward,
                       err_forward);

  std::vector<uchar> status_backward;
  std::vector<float> err_backward;
  std::vector<cv::Point2f> all_new_left_keypoints_positions;
  calcOpticalFlowPyrLK(image_r, image_l, all_right_keypoints_positions,
                       all_new_left_keypoints_positions, status_backward,
                       err_backward);

  for (auto i = 0;
       i < static_cast<int>(all_new_left_keypoints_positions.size()); i++) {
    if (status_forward[i] && status_backward[i] &&
        check_threshold(left_keypoints_positions[i],
                        all_new_left_keypoints_positions[i], 3)) {
      Eigen::Vector2d right_position;
      right_position.x() = all_right_keypoints_positions[i].x;
      right_position.y() = all_right_keypoints_positions[i].y;
      if (img_raw_r.InBounds(static_cast<int>(right_position.x()),
                             static_cast<int>(right_position.y()))) {
        kdr.emplace_back(right_position);
        stereo_matches.matches.emplace_back(i, kdr.size() - 1);
        stereo_matches.inliers.emplace_back(i, kdr.size() - 1);
      }
    }
  }
}

void add_new_landmarks_and_update_trackedPoints(
    const FrameCamId fcidl, const FrameCamId fcidr, const KeypointsData& kdl,
    const KeypointsData& kdr, const Calibration& calib_cam,
    const MatchData& md_stereo, const LandmarkMatchData& md,
    Landmarks& landmarks, TrackId& next_landmark_id,
    TrackedPoints& trackedPoints) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

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
        trackedPoints.emplace(match_feature_id, match_trackID);
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
    const auto& featureID = other_stereo_points[i].first;
    trackedPoints.emplace(featureID, new_trackID);
  }
}
void match_with_opticalflow(
    const KeypointsPositions& kd_current, KeypointsPositions& kd_next,
    const pangolin::ManagedImage<uint8_t>& img_raw_current,
    const pangolin::ManagedImage<uint8_t>& img_raw_next,
    TrackedPoints& trackedPoints, std::vector<TrackId>& trackIDs,
    LandmarkMatchData& md) {
  cv::Mat image_current(img_raw_current.h, img_raw_current.w, CV_8U,
                        img_raw_current.ptr);
  cv::Mat image_next(img_raw_next.h, img_raw_next.w, CV_8U, img_raw_next.ptr);
  std::vector<cv::Point2f> current_keypoints_positions;
  for (const auto& current_keypoint_position : kd_current) {
    cv::Point2f point;
    point.x = current_keypoint_position.x();
    point.y = current_keypoint_position.y();
    current_keypoints_positions.emplace_back(point);
  }
  std::vector<cv::Point2f> all_next_keypoints_positions;

  std::vector<uchar> status_forward;
  std::vector<float> err_forward;

  calcOpticalFlowPyrLK(image_current, image_next, current_keypoints_positions,
                       all_next_keypoints_positions, status_forward,
                       err_forward);

  std::vector<uchar> status_backward;
  std::vector<float> err_backward;
  std::vector<cv::Point2f> all_new_current_keypoints_positions;
  calcOpticalFlowPyrLK(image_next, image_current, all_next_keypoints_positions,
                       all_new_current_keypoints_positions, status_backward,
                       err_backward);
  trackedPoints.clear();

  for (auto i = 0;
       i < static_cast<int>(all_new_current_keypoints_positions.size()); i++) {
    if (status_forward[i] && status_backward[i] &&
        check_threshold(current_keypoints_positions[i],
                        all_new_current_keypoints_positions[i], 3)) {
      Eigen::Vector2d next_position;
      next_position.x() = all_next_keypoints_positions[i].x;
      next_position.y() = all_next_keypoints_positions[i].y;
      if (img_raw_next.InBounds(static_cast<int>(next_position.x()),
                                static_cast<int>(next_position.y()))) {
        kd_next.emplace_back(next_position);
        const auto& trackID = trackIDs[i];
        FeatureId featureID = kd_next.size() - 1;
        trackedPoints.emplace(featureID, trackID);
        md.matches.emplace_back(featureID, trackID);
      }
    }
  }
}

void updateVisualisationTracks(const TrackedPoints& trackedPoints,
                               const size_t tracklength,
                               const KeypointsPositions& kd,
                               VisualisationTracks& visualisationTracks) {
  VisualisationTracks newVisualisationTracks;
  for (const auto& trackedPoint : trackedPoints) {
    PointsOfTrack pointsOfTrack = visualisationTracks[trackedPoint.second];
    pointsOfTrack.insert(pointsOfTrack.begin(), kd[trackedPoint.first]);
    if (pointsOfTrack.size() > tracklength) pointsOfTrack.pop_back();
    newVisualisationTracks[trackedPoint.second] = pointsOfTrack;
  }
  visualisationTracks = newVisualisationTracks;
}

void localize_camera_and_update_trackedPoints(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const KeypointsData& kdl, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    LandmarkMatchData& md, TrackedPoints& trackedPoints) {
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
  trackedPoints.clear();
  md.T_w_c = Sophus::SE3d(refined_rotation, refined_tranlation);
  for (size_t i = 0; i < ransac.inliers_.size(); ++i) {
    md.inliers.push_back(md.matches[ransac.inliers_[i]]);
    trackedPoints.emplace(md.matches[ransac.inliers_[i]]);
  }
}

}  // namespace visnav
