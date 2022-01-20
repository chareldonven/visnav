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

TEST(OpticalFlowTestSuite, enough_points_in_patch) {
  FeaturePatchPair fpp;
  fpp.emplace(std::make_pair(0, 1));
  fpp.emplace(std::make_pair(1, 1));

  fpp.emplace(std::make_pair(2, 2));

  fpp.emplace(std::make_pair(3, 2));

  fpp.emplace(std::make_pair(4, 2));

  fpp.emplace(std::make_pair(5, 3));

  fpp.emplace(std::make_pair(6, 3));

  auto res = enough_points_in_patch(fpp, 1);

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
