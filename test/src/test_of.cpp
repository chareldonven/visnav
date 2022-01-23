
/*
#include <gtest/gtest.h>

#include <fstream>
#include <random>
#include <chrono>

#include "visnav/keypoints.h"
#include "visnav/of_utils.h"
#include "visnav/common_types.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/video/tracking.hpp>

#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
using namespace visnav;
using namespace cv;
const int NUM_FEATURES = 1500;
const int MATCH_THRESHOLD = 70;
const double DIST_2_BEST = 1.2;

const std::string img0_path = "../../visnav/test/ex3_test_data/0_0.jpg";
const std::string img1_path = "../../visnav/test/ex3_test_data/0_1.jpg";

const std::string kd0_path = "../../visnav/test/ex3_test_data/kd0.json";
const std::string kd1_path = "../../visnav/test/ex3_test_data/kd1.json";

const std::string matches_stereo_path =
    "../../visnav/test/ex3_test_data/matches_stereo.json";
const std::string matches_path = "../../visnav/test/ex3_test_data/matches.json";

const std::string calib_path = "../../visnav/test/ex3_test_data/calib.json";

TEST(OpticalFlowTestSuite, not_enough_points_in_patch) {
  FeaturePatchPair fpp;
  fpp.emplace(std::make_pair(0, 1));
  fpp.emplace(std::make_pair(1, 1));

  fpp.emplace(std::make_pair(2, 2));

  fpp.emplace(std::make_pair(3, 2));

  fpp.emplace(std::make_pair(4, 2));

  fpp.emplace(std::make_pair(5, 3));

  fpp.emplace(std::make_pair(6, 3));

  auto res = not_enough_points_in_patch(fpp, 1);

  // ASSERT_EQ();

  EXPECT_FALSE(res) << "res: " << res;
}

TEST(OpticalFlowTestSuite, find_patchID) {
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(img1_path);

  FrameCamId fcidl(0, 0);

  KeypointsData kd0, kd1;

  FeaturePatchPair fpp;
  TrackedPoints trackedPoints;
  Corners corners;

  divide_image_into_patches(img0, kd0.corners, NUM_FEATURES, fpp, trackedPoints,
                            corners, fcidl);

  for (const auto& pair : fpp) {
    cv::Point2f point;
    point.x = kd0.corners[pair.first].x();
    point.y = kd0.corners[pair.first].y();
    ASSERT_EQ(find_patchID(img0, point), pair.second);
  }
}

TEST(OpticalFlowTestSuite, add_new_landmarks_none) {
  /// Assumptions:
  /// After the call for the tracked points we add the right observations
  /// The left observation is already there
  /// We have both observations for the new landmarks
  ///
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  const FrameCamId fcidl(0, 0);
  const FrameCamId fcidr(0, 1);
  KeypointsData kdl;
  KeypointsData kdr;
  Landmarks landmarks;
  MatchData md_stereo;
  FeaturePatchPair fpp;
  TrackedPoints trackedPoints;
  Calibration calib;
  const auto& current_pose = Sophus::SE3d();

  TrackId next_landmark_id = 0;

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kdl);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kdr);
  }
  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_stereo);
  }
  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
    } else {
      ASSERT_TRUE(false) << "could not load camera ";
    }
  }
  TrackedPoints old_trackedPoints;
  FeaturePatchPair old_fpp;
  Landmarks old_landmarks;

  for (const auto& match : md_stereo.inliers) {
    OpticalFlowData track_point;
    auto featureID = match.first;
    track_point.featureID_current_frame = featureID;
    Point2f point;
    point.x = kdl.corners.at(featureID).x();
    point.y = kdl.corners.at(featureID).y();
    track_point.patchID_current_frame = find_patchID(img0, point);
    track_point.trackID = next_landmark_id;

    trackedPoints.emplace_back(track_point);
    old_trackedPoints.emplace_back(track_point);
    old_fpp.emplace(
        std::make_pair(featureID, track_point.patchID_current_frame));
    fpp.emplace(std::make_pair(featureID, track_point.patchID_current_frame));
    Landmark landmark;
    landmarks.emplace(track_point.trackID, landmark);
    old_landmarks.emplace(track_point.trackID, landmark);
    next_landmark_id++;
  }

  add_new_landmarks(fcidl, fcidr, kdl, kdr, landmarks, md_stereo, fpp,
                    trackedPoints, calib, current_pose, next_landmark_id);
  // No new landmarks to add;

  for (const auto& landmark : landmarks) {
    const auto& trackID = landmark.first;
    const auto& observations_size = landmark.second.obs.size();

    const auto& old_landmark = old_landmarks.at(trackID);

    const auto& observations_size_old = old_landmark.obs.size();

    ASSERT_EQ(observations_size, observations_size_old + 1);
  }
}

TEST(OpticalFlowTestSuite, add_new_landmarks_all) {
  /// Assumptions:
  /// After the call for the tracked points we add the right observations
  /// The left observation is already there
  /// We have both observations for the new landmarks
  ///
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  const FrameCamId fcidl(0, 0);
  const FrameCamId fcidr(0, 1);
  KeypointsData kdl;
  KeypointsData kdr;
  Landmarks landmarks;
  MatchData md_stereo;
  FeaturePatchPair fpp;
  TrackedPoints trackedPoints;
  Calibration calib;
  const auto& current_pose = Sophus::SE3d();

  TrackId next_landmark_id = 0;

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kdl);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kdr);
  }
  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_stereo);
  }
  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
    } else {
      ASSERT_TRUE(false) << "could not load camera ";
    }
  }

  for (const auto& match : md_stereo.inliers) {
    const auto& featureID = match.first;
    Point2f point;
    point.x = kdl.corners.at(featureID).x();
    point.y = kdl.corners.at(featureID).y();
    const auto& patchID = find_patchID(img0, point);
    fpp.emplace(std::make_pair(featureID, patchID));
  }
  add_new_landmarks(fcidl, fcidr, kdl, kdr, landmarks, md_stereo, fpp,
                    trackedPoints, calib, current_pose, next_landmark_id);
  std::cout << "Size: " << landmarks.size() << std::endl;
  // All new landmarks to add;
  ASSERT_EQ(landmarks.size(), md_stereo.inliers.size());
  ASSERT_EQ(trackedPoints.size(), fpp.size());
  ASSERT_EQ(trackedPoints.size(), next_landmark_id);

  for (const auto& landmark : landmarks) {
    const auto& obs = landmark.second.obs;
    ASSERT_EQ(obs.size(), 2);
  }
}

TEST(OpticalFlowTestSuite, test_all) {
  TrackedPoints trackedPoints;
  FeaturePatchPair fpp;
  Landmarks landmarks;
  Sophus::SE3d current_pose;
  Corners feature_corners;
  Matches feature_matches;
  Cameras cameras;
}
*/
