

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
#include <opencv2/opencv.hpp>
#include <math.h>

namespace visnav {

int row_size;
int col_size;
constexpr int rows = 2;
constexpr int cols = 2;

void find_keypoints_in_region(const pangolin::ManagedImage<uint8_t>& img_raw,
                              KeypointsPositions& kd, const PatchID& patchID,
                              const int num_features) {
  cv::Mat image(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);

  //////////////////////////////////////////////////////
  const auto start_row = patchID.x * row_size;
  const auto end_row = (patchID.x + 1) * row_size;
  const auto row_range = cv::Range(start_row, end_row);

  ///////////////////////////////////////////////////////
  const auto start_col = patchID.y * col_size;
  const auto end_col = (patchID.y + 1) * col_size;
  const auto col_range = cv::Range(start_col, end_col);

  /////////////////////////////////////////////////////
  cv::Mat patch(image, row_range, col_range);

  std::vector<cv::Point2f> points;
  goodFeaturesToTrack(patch, points, num_features, 0.01, 8);
  // Find real coordinates of point!!!
  for (size_t i = 0; i < points.size(); i++) {
    const auto& x = points[i].x + start_col;
    const auto& y = points[i].y + start_row;
    /// Should we use edge_threshold?
    if (img_raw.InBounds(x, y, EDGE_THRESHOLD)) {
      kd.emplace_back(x, y);

      // break;
    }
  }
}
void find_keypoints_in_all_regions(
    const pangolin::ManagedImage<uint8_t>& img_raw, KeypointsPositions& kd,
    const int num_features) {
  cv::Mat image(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);

  row_size = image.rows / rows;
  col_size = image.cols / cols;

  for (auto i = 0; i < rows; i++) {
    for (auto j = 0; j < cols; j++) {
      PatchID patchID;
      patchID.x = i;
      patchID.y = j;
      find_keypoints_in_region(img_raw, kd, patchID, num_features);
    }
  }
}

bool should_track_into_stereo(const Patches& patches) {
  bool result = true;
  for (const auto& patch : patches.patchHasKeypoints) {
    result = result && patch;
  }
  return !result;
}
PatchID find_patchID(const pangolin::ManagedImage<uint8_t>& img_raw,
                     const cv::Point2f& point) {
  cv::Mat img(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);

  const auto& x = point.x;
  const auto& y = point.y;

  PatchID patchID;
  patchID.x = std::floor(x / col_size);

  patchID.y = std::floor(y / row_size);
  return patchID;
}
void update_patches(Patches& patches, const TrackedPoints& trackedPoints,
                    const KeypointsPositions& kd,
                    const pangolin::ManagedImage<uint8_t>& img_raw) {
  for (const auto& trackedPoint : trackedPoints) {
    const auto& featureID = trackedPoint.first;
    const auto& position = kd.at(featureID);
    cv::Point2f point;
    point.x = position.x();
    point.y = position.y();
    const auto& patchID = find_patchID(img_raw, point);
    const auto& index = patchID.y * rows + patchID.x;

    patches.patchHasKeypoints.at(index) = true;
  }
}

bool check_threshold(const cv::Point2f& position0, const cv::Point2f& position1,
                     const int threshold) {
  double norm = cv::norm(cv::Mat(position0), cv::Mat(position1));
  return norm < threshold;
}

void match_initial_stereo_with_opticalflow(
    const KeypointsPositions& kdl, KeypointsPositions& kdr,
    const pangolin::ManagedImage<uint8_t>& img_raw_l,
    const pangolin::ManagedImage<uint8_t>& img_raw_r,
    MatchData& stereo_matches) {
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
                        all_new_left_keypoints_positions[i], 2)) {
      Eigen::Vector2d right_position;
      right_position.x() = all_right_keypoints_positions[i].x;
      right_position.y() = all_right_keypoints_positions[i].y;
      if (img_raw_r.InBounds(right_position.x(), right_position.y(),
                             EDGE_THRESHOLD)) {
        kdr.emplace_back(right_position);
        stereo_matches.matches.emplace_back(i, kdr.size() - 1);
        stereo_matches.inliers.emplace_back(i, kdr.size() - 1);
      }
    }
  }
}
void initialize_map(Landmarks& landmarks, TrackedPoints& trackedPoints,
                    TrackId& next_landmark_id, const MatchData& md_stereo,
                    const KeypointsData& kdl, const KeypointsData& kdr,
                    const FrameCamId& fcidl, const FrameCamId& fcidr,
                    const Calibration& calib_cam, const Sophus::SE3d& T_w_c) {
  opengv::bearingVectors_t vectors_0;
  opengv::bearingVectors_t vectors_1;

  for (size_t i = 0; i < md_stereo.inliers.size(); i++) {
    const auto& stereo_l_featureID = md_stereo.inliers[i].first;
    const auto& stereo_r_featureID = md_stereo.inliers[i].second;

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
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();
  adapter.setR12(R_0_1);
  adapter.sett12(t_0_1);
  // Triangulate new landmarks from stereo matches
  for (size_t i = 0; i < md_stereo.inliers.size(); i++) {
    const auto& triangulated_point =
        opengv::triangulation::triangulate(adapter, i);
    next_landmark_id = next_landmark_id + 1;
    const auto& new_trackID = next_landmark_id;
    landmarks[new_trackID].p = T_w_c * triangulated_point;
    landmarks.at(new_trackID).obs.emplace(fcidl, md_stereo.inliers[i].first);
    landmarks.at(new_trackID).obs.emplace(fcidr, md_stereo.inliers[i].second);
    const auto& featureID = md_stereo.inliers[i].first;
    trackedPoints.emplace(featureID, new_trackID);
  }
}

void match_stereo_with_opticalflow(
    const TrackedPoints& trackedPoints, const KeypointsPositions& old_kdl,
    KeypointsPositions& kdr, KeypointsPositions& new_kdl,
    const pangolin::ManagedImage<uint8_t>& img_raw_l,
    const pangolin::ManagedImage<uint8_t>& img_raw_r, MatchData& stereo_matches,
    LandmarkMatchData& md) {
  cv::Mat image_l(img_raw_l.h, img_raw_l.w, CV_8U, img_raw_l.ptr);
  cv::Mat image_r(img_raw_r.h, img_raw_r.w, CV_8U, img_raw_r.ptr);
  std::vector<cv::Point2f> left_keypoints_positions;
  // Save first the new features.
  for (const auto& position : new_kdl) {
    cv::Point2f point;
    point.x = position.x();
    point.y = position.y();
    left_keypoints_positions.emplace_back(point);
  }

  // new_kdl will save later only the good features.
  new_kdl.clear();
  std::vector<TrackId> trackIDs;
  // save the tracked keypoints
  for (const auto& trackedPoint : trackedPoints) {
    const auto& featureID = trackedPoint.first;
    const auto& position = old_kdl.at(featureID);
    cv::Point2f point;
    point.x = position.x();
    point.y = position.y();
    //  starts at new_kdl.size until
    // new_kdl.size + trackedPoints.size -1 as last index
    left_keypoints_positions.emplace_back(point);
    trackIDs.emplace_back(trackedPoint.second);
  }

  std::vector<cv::Point2f> all_right_keypoints_positions;
  all_right_keypoints_positions.reserve(left_keypoints_positions.size());

  std::vector<uchar> status_forward;
  std::vector<float> err_forward;

  calcOpticalFlowPyrLK(image_l, image_r, left_keypoints_positions,
                       all_right_keypoints_positions, status_forward,
                       err_forward);

  std::vector<uchar> status_backward;
  std::vector<float> err_backward;
  std::vector<cv::Point2f> all_new_left_keypoints_positions;
  all_new_left_keypoints_positions.reserve(left_keypoints_positions.size());
  calcOpticalFlowPyrLK(image_r, image_l, all_right_keypoints_positions,
                       all_new_left_keypoints_positions, status_backward,
                       err_backward);
  // for those keypoints that dont have a trackID
  for (auto i = 0;
       i < static_cast<int>(all_new_left_keypoints_positions.size() -
                            trackIDs.size());
       i++) {
    if (status_forward[i] && status_backward[i] &&
        check_threshold(left_keypoints_positions[i],
                        all_new_left_keypoints_positions[i], 2)) {
      Eigen::Vector2d right_position;
      right_position.x() = all_right_keypoints_positions[i].x;
      right_position.y() = all_right_keypoints_positions[i].y;
      if (img_raw_r.InBounds(all_right_keypoints_positions[i].x,
                             all_right_keypoints_positions[i].y,
                             EDGE_THRESHOLD)) {
        Eigen::Vector2d left_position;
        left_position.x() = left_keypoints_positions[i].x;
        left_position.y() = left_keypoints_positions[i].y;
        new_kdl.emplace_back(left_position);
        kdr.emplace_back(right_position);
        stereo_matches.matches.emplace_back(new_kdl.size() - 1, kdr.size() - 1);
        stereo_matches.inliers.emplace_back(new_kdl.size() - 1, kdr.size() - 1);
      }
    }
  }

  for (auto i = static_cast<int>(all_new_left_keypoints_positions.size() -
                                 trackIDs.size());
       i < static_cast<int>(all_new_left_keypoints_positions.size()); i++) {
    if (status_forward[i] && status_backward[i] &&
        check_threshold(left_keypoints_positions[i],
                        all_new_left_keypoints_positions[i], 2)) {
      Eigen::Vector2d right_position;
      right_position.x() = all_right_keypoints_positions[i].x;
      right_position.y() = all_right_keypoints_positions[i].y;
      if (img_raw_r.InBounds(all_right_keypoints_positions[i].x,
                             all_right_keypoints_positions[i].y,
                             EDGE_THRESHOLD)) {
        Eigen::Vector2d left_position;
        left_position.x() = left_keypoints_positions[i].x;
        left_position.y() = left_keypoints_positions[i].y;
        new_kdl.emplace_back(left_position);
        kdr.emplace_back(right_position);
        stereo_matches.matches.emplace_back(new_kdl.size() - 1, kdr.size() - 1);
        stereo_matches.inliers.emplace_back(new_kdl.size() - 1, kdr.size() - 1);
        const auto& index_trackID =
            i - all_new_left_keypoints_positions.size() + trackIDs.size();
        md.matches.emplace_back(new_kdl.size() - 1, trackIDs[index_trackID]);
        md.inliers.emplace_back(new_kdl.size() - 1, trackIDs[index_trackID]);
      }
    }
  }
}
void add_new_landmarks_update_trackedPoints(
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
      ///////////////////////////////////////////////////////////////////

      ///////////////////////////////////////////////////////////////////
      if (match.first == stereo_l_featureID) {
        // Add an observation for the right camera as well
        trackedPoints[match_feature_id] = match_trackID;
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
    //////////////////////////////////////////////////////////////////////////////////////
    trackedPoints[other_stereo_points[i].first] = new_trackID;
  }
}
void match_with_opticalflow(
    const KeypointsPositions& kd_current, KeypointsPositions& kd_next,
    const pangolin::ManagedImage<uint8_t>& img_raw_current,
    const pangolin::ManagedImage<uint8_t>& img_raw_next,
    const TrackedPoints& trackedPoints, std::vector<TrackId>& trackIDs,
    LandmarkMatchData& md) {
  cv::Mat image_current(img_raw_current.h, img_raw_current.w, CV_8U,
                        img_raw_current.ptr);
  cv::Mat image_next(img_raw_next.h, img_raw_next.w, CV_8U, img_raw_next.ptr);
  std::vector<cv::Point2f> current_keypoints_positions;

  for (const auto& trackedPoint : trackedPoints) {
    const auto& featureID = trackedPoint.first;
    const auto& position = kd_current.at(featureID);
    cv::Point2f point;
    point.x = position.x();
    point.y = position.y();
    current_keypoints_positions.emplace_back(point);
    trackIDs.emplace_back(trackedPoint.second);
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

  for (auto i = 0;
       i < static_cast<int>(all_new_current_keypoints_positions.size()); i++) {
    if (status_forward[i] && status_backward[i] &&
        check_threshold(current_keypoints_positions[i],
                        all_new_current_keypoints_positions[i], 2)) {
      Eigen::Vector2d next_position;
      next_position.x() = all_next_keypoints_positions[i].x;
      next_position.y() = all_next_keypoints_positions[i].y;
      if (img_raw_next.InBounds(next_position.x(), next_position.y(),
                                EDGE_THRESHOLD)) {
        kd_next.emplace_back(next_position);
        const auto& trackID = trackIDs[i];
        FeatureId featureID = kd_next.size() - 1;

        md.matches.emplace_back(featureID, trackID);
      }
    }
  }
}

}  // namespace visnav
