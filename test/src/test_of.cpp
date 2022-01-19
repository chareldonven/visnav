#include <gtest/gtest.h>

#include <fstream>

#include "visnav/keypoints.h"
#include "visnav/of_utils.h"
#include "visnav/common_types.h"

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/tracking.hpp>
using namespace visnav;
using namespace cv;
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

/*
 * PatchID find_patchID(const pangolin::ManagedImage<uint8_t>& img_raw,
                     cv::Point2f point)
 *
 * */

TEST(OpticalFlowTestSuite, find_patchID) {
  // load_data("../visnav/data/V1_01_easy/mav0/");
  FrameCamId fcidl(0, 0);

  Mat image = imread(
      "home/ajvi/Documents/WI2122/VisNav/TEAM1/visnav/data/V1_01_easy/mav0/"
      "cam0/data/1403715273262142976.png");
  if (image.empty()) {
    std::cout << "Could not open or find the image" << std::endl;
  }

  imshow("Bild", image);
  waitKey(0);
  /*
   KeypointsPositions kd;
   int num_features = 10;
   FeaturePatchPair fpp;
   TrackedPoints trackedPoints;
   Corners corners;

   divide_image_into_patches(img, kd, num_features, fpp, trackedPoints,
   corners, fcidl);

     for (const auto& pair : fpp) {
       cv::Point2f point;
       point.x = kd[pair.first].x();
       point.y = kd[pair.first].y();
       ASSERT_EQ(find_patchID(img, point), pair.second);
     }*/
}
