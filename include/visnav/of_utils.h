

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
constexpr int rows = 16;
constexpr int cols = 16;
constexpr int nr_it = 16;
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
    const pangolin::ManagedImage<uint8_t>& img_raw, KeypointsData& kd,
    const int num_features) {
  cv::Mat image(img_raw.h, img_raw.w, CV_8U, img_raw.ptr);

  row_size = image.rows / nr_it;
  col_size = image.cols / nr_it;
  for (auto i = 0; i < nr_it; i++) {
    for (auto j = 0; j < nr_it; j++) {
      PatchID patchID;
      patchID.x = i;
      patchID.y = j;
      find_keypoints_in_region(img_raw, kd.corners, patchID, num_features);
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
  patchID.x = std::floor(x / row_size);
  patchID.y = std::floor(y / col_size);
  return patchID;
}
void update_patches(Patches& patches, const TrackedPoints& trackedPoints,
                    const KeypointsPositions& kd,
                    const pangolin::ManagedImage<uint8_t>& img_raw) {
  for (const auto& trackedPoint : trackedPoints) {
    const auto& featureID = trackedPoint.first;
    const auto& position = kd.at(featureID);
    cv::Point2d point;
    point.x = position.x();
    point.y = position.y();
    const auto& patchID = find_patchID(img_raw, point);
    const auto& index = patchID.y * rows + patchID.x;
    patches.patchHasKeypoints.at(index) = true;
  }
}
}  // namespace visnav
