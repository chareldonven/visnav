#pragma once

#include <iostream>
#include <sstream>

#include <fstream>
#include <sophus/se3.hpp>
#include <iomanip>

namespace visnav {
// used for evaluation
void save_trajectory(const std::vector<Sophus::SE3d>& poses,
                     const std::vector<int64_t>& timestamps,
                     const std::string& filename) {
  std::ofstream os(filename);

  os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

  for (size_t i = 0; i < poses.size(); i++) {
    const Sophus::SE3d& pose = poses[i];

    os << std::scientific << std::setprecision(18) << timestamps[i] << " "
       << pose.translation().x() << " " << pose.translation().y() << " "
       << pose.translation().z() << " " << pose.unit_quaternion().x() << " "
       << pose.unit_quaternion().y() << " " << pose.unit_quaternion().z() << " "
       << pose.unit_quaternion().w() << std::endl;
  }

  std::cout << "Size of poses: " << poses.size() << " -> Saved trajectory!"
            << std::endl;
  os.close();
}

}  // namespace visnav
