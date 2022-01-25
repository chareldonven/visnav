

#include <CLI/CLI.hpp>

#include <sophus/se3.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <visnav/evaluate_utils.h>

template <class Archive, class T, class C, class A>
inline void load(Archive& ar, std::map<std::string, T, C, A>& map) {
  map.clear();

  auto hint = map.begin();
  while (true) {
    const auto namePtr = ar.getNodeName();

    if (!namePtr) break;

    std::string key = namePtr;
    T value;
    ar(value);
    hint = map.emplace_hint(hint, std::move(key), std::move(value));
  }
}

void save_trajectory(
    const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>&
        result_poses) {
  std::ofstream os("trajectory_evaluate.txt");

  os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

  for (size_t i = 1; i < result_poses.size(); i++) {
    const Sophus::SE3d& pose = result_poses[i];
    const auto& timestamp = i;
    os << std::scientific << std::setprecision(18) << timestamp << " "
       << pose.translation().x() << " " << pose.translation().y() << " "
       << pose.translation().z() << " " << pose.unit_quaternion().x() << " "
       << pose.unit_quaternion().y() << " " << pose.unit_quaternion().z()
       << pose.unit_quaternion().w() << std::endl;
  }

  std::cout << "Saved trajectory!" << std::endl;
  os.close();
}

std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> load_poses(
    const std::string& path, std::vector<double>& t_ns) {
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> res;

  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line[0] == '#') continue;

    for (char& c : line) {
      if (c == ',') c = ' ';
    }
    std::stringstream ss(line);

    Eigen::Quaternion<double> q;
    Eigen::Vector3d p;
    double timestamp;
    ss >> timestamp >> p.x() >> p.y() >> p.z() >> q.w() >> q.x() >> q.y() >>
        q.z();
    Sophus::SE3d pose(q, p);

    res.emplace_back(pose);
    t_ns.emplace_back(timestamp);
  }

  return res;
}

double alignSVD(
    const std::vector<double>& filter_t_ns,
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& filter_t_w_i,
    const std::vector<double>& gt_t_ns,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>&
        gt_t_w_i,
    Sophus::SE3d& result) {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      est_associations;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      gt_associations;

  for (size_t i = 0; i < filter_t_w_i.size(); i++) {
    double t_ns = filter_t_ns[i];

    size_t j;
    for (j = 0; j < gt_t_ns.size(); j++) {
      if (gt_t_ns.at(j) > t_ns) break;
    }
    j--;

    if (j >= gt_t_ns.size() - 1) {
      continue;
    }

    double dt_ns = t_ns - gt_t_ns.at(j);
    double int_t_ns = gt_t_ns.at(j + 1) - gt_t_ns.at(j);

    BASALT_ASSERT_STREAM(dt_ns >= 0, "dt_ns " << dt_ns);
    BASALT_ASSERT_STREAM(int_t_ns > 0, "int_t_ns " << int_t_ns);

    // Skip if the interval between gt larger than 100ms
    if (int_t_ns > 1.1e8) continue;

    double ratio = dt_ns / int_t_ns;

    BASALT_ASSERT(ratio >= 0);
    BASALT_ASSERT(ratio < 1);

    Eigen::Vector3d gt = (1 - ratio) * gt_t_w_i[j] + ratio * gt_t_w_i[j + 1];

    gt_associations.emplace_back(gt);
    est_associations.emplace_back(filter_t_w_i[i]);
  }

  int num_kfs = est_associations.size();

  Eigen::Matrix<double, 3, Eigen::Dynamic> gt, est;
  gt.setZero(3, num_kfs);
  est.setZero(3, num_kfs);

  for (size_t i = 0; i < est_associations.size(); i++) {
    gt.col(i) = gt_associations[i];
    est.col(i) = est_associations[i];
  }

  Eigen::Vector3d mean_gt = gt.rowwise().mean();
  Eigen::Vector3d mean_est = est.rowwise().mean();

  gt.colwise() -= mean_gt;
  est.colwise() -= mean_est;

  Eigen::Matrix3d cov = gt * est.transpose();

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3d S;
  S.setIdentity();

  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0)
    S(2, 2) = -1;

  Eigen::Matrix3d rot_gt_est = svd.matrixU() * S * svd.matrixV().transpose();
  Eigen::Vector3d trans = mean_gt - rot_gt_est * mean_est;

  Sophus::SE3d T_gt_est(rot_gt_est, trans);
  Sophus::SE3d T_est_gt = T_gt_est.inverse();

  for (size_t i = 0; i < gt_t_w_i.size(); i++) {
    gt_t_w_i[i] = T_est_gt * gt_t_w_i[i];
  }

  double error = 0;
  for (size_t i = 0; i < est_associations.size(); i++) {
    est_associations[i] = T_gt_est * est_associations[i];
    Eigen::Vector3d res = est_associations[i] - gt_associations[i];

    error += res.transpose() * res;
  }

  error /= est_associations.size();
  error = std::sqrt(error);
  result = T_gt_est;
  std::cout << "T_align\n" << T_gt_est.matrix() << std::endl;
  std::cout << "error " << error << std::endl;
  std::cout << "number of associations " << num_kfs << std::endl;

  return error;
}

void eval_kitti(
    const std::vector<double>& lengths,
    const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>&
        poses_gt,
    const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>&
        poses_result,
    std::map<std::string, std::map<std::string, double>>& res) {
  auto lastFrameFromSegmentLength = [](std::vector<float>& dist,
                                       int first_frame, float len) {
    for (int i = first_frame; i < (int)dist.size(); i++)
      if (dist[i] > dist[first_frame] + len) return i;
    return -1;
  };

  std::cout << "poses_gt.size() " << poses_gt.size() << std::endl;
  std::cout << "poses_result.size() " << poses_result.size() << std::endl;

  // pre-compute distances (from ground truth as reference)
  std::vector<float> dist_gt;
  dist_gt.emplace_back(0);
  for (size_t i = 1; i < poses_gt.size(); i++) {
    const auto& p1 = poses_gt[i - 1];
    const auto& p2 = poses_gt[i];

    dist_gt.emplace_back(dist_gt.back() +
                         (p2.translation() - p1.translation()).norm());
  }

  const size_t step_size = 10;

  for (size_t i = 0; i < lengths.size(); i++) {
    // current length
    float len = lengths[i];

    double t_error_sum = 0;
    double r_error_sum = 0;
    int num_meas = 0;

    for (size_t first_frame = 0; first_frame < poses_gt.size();
         first_frame += step_size) {
      // for all segment lengths do

      // compute last frame
      int32_t last_frame =
          lastFrameFromSegmentLength(dist_gt, first_frame, len);

      // continue, if sequence not long enough
      if (last_frame == -1) continue;

      // compute rotational and translational errors
      Sophus::SE3d pose_delta_gt =
          poses_gt[first_frame].inverse() * poses_gt[last_frame];
      Sophus::SE3d pose_delta_result =
          poses_result[first_frame].inverse() * poses_result[last_frame];
      // Sophus::SE3d pose_error = pose_delta_result.inverse() * pose_delta_gt;
      double r_err = pose_delta_result.unit_quaternion().angularDistance(
                         pose_delta_gt.unit_quaternion()) *
                     180.0 / M_PI;
      double t_err =
          (pose_delta_result.translation() - pose_delta_gt.translation())
              .norm();

      t_error_sum += t_err / len;
      r_error_sum += r_err / len;
      num_meas++;
    }

    std::string len_str = std::to_string((int)len);
    res[len_str]["trans_error"] = 100.0 * t_error_sum / num_meas;
    res[len_str]["rot_error"] = r_error_sum / num_meas;
    res[len_str]["num_meas"] = num_meas;
  }
}

int main(int argc, char** argv) {
  std::vector<double> lengths = {100, 200, 300, 400, 500, 600, 700, 800};
  std::string result_path;
  std::string traj_path;
  std::string gt_path;

  CLI::App app{"KITTI evaluation"};

  app.add_option("--traj-path", traj_path,
                 "Path to the file with computed trajectory.")
      ->required();
  app.add_option("--gt-path", gt_path,
                 "Path to the file with ground truth trajectory.")
      ->required();
  app.add_option("--result-path", result_path, "Path to store the result file.")
      ->required();

  app.add_option("--eval-lengths", lengths, "Trajectory length to evaluate.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }
  std::vector<double> gt_t_ns;
  std::vector<double> traj_t_ns;
  const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>
      poses_gt = load_poses(gt_path, gt_t_ns);
  const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>
      poses_estimated = load_poses(traj_path, traj_t_ns);

  if (poses_gt.empty() || poses_gt.size() != poses_estimated.size()) {
    std::cerr << "Wrong number of poses: poses_gt " << poses_gt.size()
              << " poses_result " << poses_estimated.size() << std::endl;
    std::abort();
  }
  /*
    std::map<std::string, std::map<std::string, double>> res_map;
    eval_kitti(lengths, poses_gt, poses_result, res_map);

    {
      cereal::JSONOutputArchive ar(std::cout);
      ar(cereal::make_nvp("results", res_map));
      std::cout << std::endl;
    }

    if (!result_path.empty()) {
      std::ofstream os(result_path);
      {
        cereal::JSONOutputArchive ar(os);
        ar(cereal::make_nvp("results", res_map));
      }
      os.close();
    } */

  // basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i)
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      gt_translations;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      traj_translations;
  for (const auto& pose : poses_estimated) {
    traj_translations.emplace_back(pose.translation());
  }
  for (const auto& pose : poses_gt) {
    gt_translations.emplace_back(pose.translation());
  }
  Sophus::SE3d result;
  alignSVD(traj_t_ns, traj_translations, gt_t_ns, gt_translations, result);

  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>
      result_poses;

  for (auto& pose : poses_estimated) {
    result_poses.emplace_back(result * pose);
  }
  save_trajectory(result_poses);
}
