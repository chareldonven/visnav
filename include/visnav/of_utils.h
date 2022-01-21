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

struct OpticalFlowPairs {
  std::vector<cv::Point2f> source_points;
  std::vector<cv::Point2f> target_points;
  std::vector<FeatureId> outliers;
  std::vector<FeatureId> inliers;
};

/// This method saves in kd.corners the keypoints of the left stereo image.
/// It also saves in fpp the pair (featureID, patchID)
/// In trackedPoints saves the keypoints that have been tracked until now. This
/// means that the featureID and PatchID are for this image.
///  At the end of the
/// method, kd contains all points from trackedPoints and the new keypoints in
/// the image. There can be duplicates!

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

/// This method checks if two given points are close enough to each other
bool check_threshold(const cv::Point2f& position0, const cv::Point2f& position1,
                     const int threshold) {
  double norm = cv::norm(cv::Mat(position0), cv::Mat(position1));
  return norm < threshold;
}

bool check_duplicates(const Eigen::Vector2d& point0,
                      const Eigen::Vector2d& point1, const double threshold) {
  double norm = (point0 - point1).norm();
  return norm < threshold;
}

/// This function removes all duplicates from the tracked points. It is called
/// when there are to many points tracked. It searches in the given keypoints
/// for points which are too close to each other and saves the index of that
/// position. Then it removes points at the found index position from kd and
/// also from the tracked_points. Additionally it removes the entry in the fpp.
/// Finally the featureIds are newly distributed for the tracked points and the
/// fpp.
void filter_for_duplicates(TrackedPoints& trackedPoints, KeypointsPositions& kd,
                           FeaturePatchPair& fpp) {
  std::vector<int> indexVector;
  for (int i = kd.size() - 1; i >= 0; i--) {
    for (int j = i - 1; j >= 0; j--) {
      if (check_duplicates(kd.at(i), kd.at(j), 1e-5)) {
        indexVector.push_back(i);
        break;
      }
    }
  }

  for (size_t i = 0; i < indexVector.size(); i++) {
    kd.erase(kd.begin() + indexVector.at(i));
    FeatureId featureId =
        trackedPoints.at(indexVector.at(i)).featureID_current_frame;
    trackedPoints.erase(trackedPoints.begin() + indexVector.at(i));
    fpp.erase(featureId);
  }

  int index = 0;
  FeaturePatchPair new_fpp;
  for (size_t i = 0; i < trackedPoints.size(); i++) {
    new_fpp[index] = trackedPoints.at(i).patchID_current_frame;
    trackedPoints.at(i).featureID_current_frame = index;
  }
  fpp = new_fpp;
}

/// This method is called during stereo matching in order to track the keypoints
/// saved in kdl into the stereo image. kdl stores both the keypoints tracked
/// until now and the new keypoints. The method will not change kdl.This means
/// that all keypoints are saved there, not only the inlier keypoints. kdr is
/// empty at first and then contains the inlier keypoints that could be tracked
/// into it. fpp contains all featurepatchPairs that have been updated by
/// calling divide_image_into_patches At the end of the method, fpp will contain
/// only the inlier pairs of featurepatch.

/// stereo_trackedPoints is empty at first and will contain all inlier matches
/// from tracking with optical flow

void find_opticalflow_stereo_matches(
    FeaturePatchPair& fpp, const KeypointsPositions& kdl,
    KeypointsPositions& kdr, const pangolin::ManagedImage<uint8_t>& img_raw_0,
    const pangolin::ManagedImage<uint8_t>& img_raw_1,
    MatchData& stereo_trackedPoints) {
  cv::Mat image_0(img_raw_0.h, img_raw_0.w, CV_8U, img_raw_0.ptr);
  cv::Mat image_1(img_raw_1.h, img_raw_1.w, CV_8U, img_raw_1.ptr);

  OpticalFlowPairs forward_tracking, backward_tracking;
  // We need cv::Point2f in order to call calcOpticalFlowPyrLk()
  for (const auto& position : kdl) {
    cv::Point2f point;
    point.x = position[0];
    point.y = position[1];
    forward_tracking.source_points.emplace_back(point);
  }
  std::vector<uchar> status;
  std::vector<float> err;
  // forward_tracking.target_points contains the calculated 2d positions in the
  // right image. It has the same dimension as forward_tracking.source_points
  calcOpticalFlowPyrLK(image_0, image_1, forward_tracking.source_points,
                       forward_tracking.target_points, status, err);

  for (size_t i = 0; i < forward_tracking.target_points.size(); i++) {
    // featureID is the index of the keypoint in kdl
    FeatureId featureID = i;
    if (!status[i]) {
      // outliers contains the featureIDs that have not been tracked into the
      // stereo image
      forward_tracking.outliers.emplace_back(featureID);
    } else {
      // inliers contains the featureIDs that have been tracked into the stereo
      // image
      forward_tracking.inliers.emplace_back(featureID);
      // only the points that have a positive status should be tracked back
      backward_tracking.source_points.emplace_back(
          forward_tracking.target_points[i]);
    }
  }

  status.clear();
  err.clear();
  // Track back from the right image to the left image
  calcOpticalFlowPyrLK(image_1, image_0, backward_tracking.source_points,
                       backward_tracking.target_points, status, err);

  size_t last_index = 0;
  for (size_t k = 0; k < backward_tracking.target_points.size(); k++) {
    // Here featureID is again the index of the keypoint in the kdl
    FeatureId featureID = forward_tracking.inliers[k];
    if (!status[k]) {
      // outliers contains the featureIDs that have not been tracked back
      backward_tracking.outliers.emplace_back(featureID);
    } else {
      // the featureIDs that have been tracked back
      backward_tracking.inliers.emplace_back(featureID);
      // backward_tracking.target_points contains only the inlier points from
      // forward and backward tracking
      backward_tracking.target_points[last_index] =
          backward_tracking.target_points[k];
      last_index++;
    }
  }
  backward_tracking.target_points.resize(last_index);

  // Compare tracking_points to tracking_points_backward!
  // threshold 3 pixels
  FeaturePatchPair temp_fpp;

  FeatureId temp_right_featureID = 0;
  for (size_t i = 0; i < backward_tracking.inliers.size(); i++) {
    // backward_tracking.inliers is the smallest set of inliers
    for (size_t j = 0; j < forward_tracking.inliers.size(); j++) {
      const auto& featureID_bti = backward_tracking.inliers[i];
      const auto& featureID_fti = forward_tracking.inliers[j];
      // this set contains also the back tracked points with negative status
      if (featureID_bti == featureID_fti) {
        // i is the right index because target_points and inliers are assumed to
        // have the same dimension featureID_fti is the right index because this
        // is the point with the same featureID as featureID_bti and thus should
        // have the same 3d position as it
        /// TODO: Check whether check_threshhold checks for pixel units
        if (check_threshold(backward_tracking.target_points[i],
                            forward_tracking.source_points[featureID_fti], 3)) {
          // This is an inlier point from the left stereo image
          // Update fpp with this

          temp_fpp.emplace(featureID_fti, fpp.at(featureID_fti));

          Eigen::Vector2d inlier_keypoint_right;
          inlier_keypoint_right[0] =
              forward_tracking.target_points[featureID_fti].x;
          inlier_keypoint_right[1] =
              forward_tracking.target_points[featureID_fti].y;
          // Find the inlier point in the right image and add it to kdr
          kdr.emplace_back(inlier_keypoint_right);
          // Add the pair to matches

          stereo_trackedPoints.inliers.emplace_back(
              std::make_pair(featureID_fti, temp_right_featureID));
          stereo_trackedPoints.matches.emplace_back(
              std::make_pair(featureID_fti, temp_right_featureID));
          temp_right_featureID++;
        }
      }
    }
  }

  fpp.clear();
  for (const auto& fpp_pair : temp_fpp) {
    fpp.emplace(fpp_pair);
  }
}

/// This method is called during frame to frame tracking in order to track the
/// keypoints saved in  kd_current into the next image. kd_current strores the
/// keypoints of the current image, which are either the inlier keypoints in the
/// left frame after stereo-tracking or the tracked points from the previous
/// left frame. At first it contains feature_corners.at(fcid_current); where if
/// the previous step was stereo tracking, it contains all keypoints of the left
/// frame, not only the inlier keypoints. If the previous step was
/// frame_to_frame tracking, it contains only the inlier keypoints.
///
/////////////////////////////////////////////////////////////////////////////////////////////////////
/// Will the method change kd_current? -> I think not
///
/// kd_next is empty at first and at the end of the method it should contain the
/// inlier keypoints that could be tracked into  the next frame
///
/// fpp contains the result of the previous phase.
/// This means, if the previous step was stereo_tracking, it contains the result
/// from stereo_matching, which is all the inlier pairs from the left image. If
/// the previous step was frame to frame tracking, then it contains the result
/// of this method, which is also the inlier keypoints from the current_image
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/// At the end, it should contain the inlier pairs of the next_image.
///  tracked_points contains the tracked points into the current_image and at
///  the end should contain the tracked_points into the next_image
///
void find_opticalflow_matches(FeaturePatchPair& fpp,
                              const KeypointsPositions& kd_current,
                              KeypointsPositions& kd_next,
                              const pangolin::ManagedImage<uint8_t>& img_raw_0,
                              const pangolin::ManagedImage<uint8_t>& img_raw_1,
                              TrackedPoints& trackedPoints) {
  cv::Mat image_0(img_raw_0.h, img_raw_0.w, CV_8U, img_raw_0.ptr);
  cv::Mat image_1(img_raw_1.h, img_raw_1.w, CV_8U, img_raw_1.ptr);

  OpticalFlowPairs forward_tracking, backward_tracking;
  // We need cv::Point2f in order to call calcOpticalFlowPyrLk()
  // In stereo we track all points from kd_current, but what we here actually
  // want to track are only the inlier keypoints The FeatureIDs of these points
  // are saved in tracked_points

  std::map<FeatureId, TrackId> trackIDs;
  for (const auto& tracked_point : trackedPoints) {
    const auto& featureID = tracked_point.featureID_current_frame;
    const auto& trackID = tracked_point.trackID;

    trackIDs[featureID] = trackID;
    cv::Point2f point;

    point.x = kd_current.at(featureID).x();
    point.y = kd_current.at(featureID).y();
    forward_tracking.source_points.emplace_back(point);
  }
  trackedPoints.clear();
  fpp.clear();
  std::vector<uchar> status;
  std::vector<float> err;
  // forward_tracking.target_points contains the calculated 2d positions in the
  // right image. It has the same dimension as forward_tracking.source_points
  calcOpticalFlowPyrLK(image_0, image_1, forward_tracking.source_points,
                       forward_tracking.target_points, status, err);

  for (size_t i = 0; i < forward_tracking.target_points.size(); i++) {
    // new indices
    FeatureId featureID = trackedPoints[i].featureID_current_frame;
    if (!status[i]) {
      // outliers contains the featureIDs that have not been tracked into the
      // stereo image
      forward_tracking.outliers.emplace_back(featureID);
    } else {
      // inliers contains the featureIDs that have been tracked into the stereo
      // image
      forward_tracking.inliers.emplace_back(featureID);
      // only the points that have a positive status should be tracked back
      backward_tracking.source_points.emplace_back(
          forward_tracking.target_points[i]);
    }
  }

  status.clear();
  err.clear();
  // Track back from the right image to the left image
  calcOpticalFlowPyrLK(image_1, image_0, backward_tracking.source_points,
                       backward_tracking.target_points, status, err);

  size_t last_index = 0;
  for (size_t k = 0; k < backward_tracking.target_points.size(); k++) {
    // Here featureID is again the index of the keypoint in the kd_current
    FeatureId featureID = forward_tracking.inliers[k];
    if (!status[k]) {
      // outliers contains the featureIDs that have not been tracked back
      backward_tracking.outliers.emplace_back(featureID);
    } else {
      // the featureIDs that have been tracked back
      backward_tracking.inliers.emplace_back(featureID);
      // backward_tracking.target_points contains only the inlier points from
      // forward and backward tracking
      backward_tracking.target_points[last_index] =
          backward_tracking.target_points[k];
      last_index++;
    }
  }
  backward_tracking.target_points.resize(last_index);

  // Compare tracking_points to tracking_points_backward!
  // threshold 3 pixels

  for (size_t i = 0; i < backward_tracking.inliers.size(); i++) {
    // backward_tracking.inliers is the smallest set of inliers
    for (size_t j = 0; j < forward_tracking.inliers.size(); j++) {
      const auto& featureID_bti = backward_tracking.inliers[i];
      const auto& featureID_fti = forward_tracking.inliers[j];
      // this set contains also the back tracked points with negative status
      if (featureID_bti == featureID_fti) {
        // i is the right index because target_points and inliers are assumed to
        // have the same dimension featureID_fti is the right index because this
        // is the point with the same featureID as featureID_bti and thus should
        // have the same 3d position as it
        /// TODO: Check whether check_threshhold checks for pixel units
        if (check_threshold(backward_tracking.target_points.at(i),
                            forward_tracking.source_points.at(j), 3)) {
          // This is an inlier point from the left stereo image
          // fpp should contain the next pairs. So the featureID is the index in
          // the backward inliers As for the patchID, we should compute it.
          const auto& patchID =
              find_patchID(img_raw_1, forward_tracking.target_points[j]);

          fpp.emplace(i, patchID);

          Eigen::Vector2d inlier_keypoint_next;
          inlier_keypoint_next[0] = forward_tracking.target_points[j].x;
          inlier_keypoint_next[1] = forward_tracking.target_points[j].y;
          // Find the inlier point in the right image and add it to kd_next
          kd_next.emplace_back(inlier_keypoint_next);
          /* Update trackedPoints

           * trackedPoints should contain the inlier points of the next_image
           * featureID is the next featureID it has in the next image -> so the
           index in the backward inliers
           * The patchID we know from fpp.
           * The trackID should remain the same; How do we now which one?
           * */
          OpticalFlowData new_trackedPoint;
          new_trackedPoint.trackID = trackIDs.at(featureID_fti);
          new_trackedPoint.featureID_current_frame = kd_next.size() - 1;
          new_trackedPoint.patchID_current_frame = patchID;

          trackedPoints.emplace_back(new_trackedPoint);
        }
      }
    }
  }
}

/// This method checks if all patch contains enough points
bool not_enough_points_in_patch(const FeaturePatchPair& fpp,
                                unsigned int min_points_per_patch) {
  unsigned int patch_counter[4];
  for (size_t i = 0; i < 4; i++) {
    patch_counter[i] = 0;
  }

  for (const auto& elem : fpp) {
    patch_counter[elem.second - 1] += 1;
  }

  bool res = false;

  for (int i = 0; i < 4; i++) {
    res = res || (patch_counter[i] < min_points_per_patch);
  }

  return !res;
}

/// This checks if sterio matching is nessesary.
/// In the first frame it has to do it in the following enough_points_in_patch
/// is called
bool should_track_into_stereo(int current_frame, const FeaturePatchPair& fpp,
                              int min_points_per_patch) {
  if (current_frame == 0) return true;
  return not_enough_points_in_patch(fpp, min_points_per_patch);
}

/// This method is used by stereo tracking
/// fcidl and fcidr are the corresponding FrameCamIds of the left and right
/// stereo frames. kdl.corners and kdr.corners contain the 2D positions of the
/// keypoints in the left and right stereo frame. landmarks contains the
/// existing landmarks -> Here we should add new observations for the currently
/// tracked points. We add for all keypoints in stereo_trackedPoints AND in
/// trackedPoints only the right observation, because we assume the left one has
/// been already added. Then we create new landmarks from the
/// stereo_trackedPoint that were not used in the first step Right now, we do
/// not check for duplicates! To add new landmarks, we triangulate from the
/// stereo_trackedPoints. We add both, left and right observations. Then, we
/// update trackedPoints by adding the new trackIDs with the corresponding left
/// FeatureId and PatchId. These we have saved in fpp in
/// find_optical_flow_matches for stereo. Until now trackedPoints contained the
/// points tracked into the left frame, patchID is set correctly, without the
/// new found keypoints.
void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       Landmarks& landmarks, MatchData& md_stereo,
                       FeaturePatchPair& fpp, TrackedPoints& trackedPoints,
                       const Calibration& calib_cam,
                       const Sophus::SE3d& current_pose,
                       TrackId& next_landmark_id) {
  // First, add new observations for the tracked points
  std::vector<std::pair<FeatureId, FeatureId>> other_stereo_points;

  for (const auto& stereo_trackedPoint : md_stereo.inliers) {
    bool is_not_used = true;
    const auto& featureID_left = stereo_trackedPoint.first;
    const auto& featureID_right = stereo_trackedPoint.second;
    for (const auto& trackedPoint : trackedPoints) {
      if (featureID_left == trackedPoint.featureID_current_frame) {
        landmarks.at(trackedPoint.trackID).obs.emplace(fcidr, featureID_right);
        is_not_used = false;
      }
    }
    /*
     * Find inlier stereo matches that were not used in the previous stage
     *
     */
    if (is_not_used) {
      other_stereo_points.emplace_back(featureID_left, featureID_right);
    }
  }
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

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
    landmarks[new_trackID].p = current_pose * triangulated_point;
    landmarks.at(new_trackID).obs.emplace(fcidl, other_stereo_points[i].first);
    landmarks.at(new_trackID).obs.emplace(fcidr, other_stereo_points[i].second);

    // Update trackedPoints

    OpticalFlowData new_trackedPoint;
    new_trackedPoint.featureID_current_frame = other_stereo_points[i].first;
    new_trackedPoint.patchID_current_frame =
        fpp.at(new_trackedPoint.featureID_current_frame);
    new_trackedPoint.trackID = new_trackID;
    trackedPoints.emplace_back(new_trackedPoint);
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
  return;

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
