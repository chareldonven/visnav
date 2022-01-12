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

#include <fstream>
#include <thread>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

#include <visnav/reprojection.h>
#include <visnav/local_parameterization_se3.hpp>

#include <visnav/tracks.h>

namespace visnav {

// save map with all features and matches
void save_map_file(const std::string& map_path, const Corners& feature_corners,
                   const Matches& feature_matches,
                   const FeatureTracks& feature_tracks,
                   const FeatureTracks& outlier_tracks, const Cameras& cameras,
                   const Landmarks& landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string& map_path, Corners& feature_corners,
                   Matches& feature_matches, FeatureTracks& feature_tracks,
                   FeatureTracks& outlier_tracks, Cameras& cameras,
                   Landmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const FrameCamId& fcid0,
                                   const FrameCamId& fcid1,
                                   const Calibration& calib_cam,
                                   const Corners& feature_corners,
                                   const FeatureTracks& feature_tracks,
                                   const Cameras& cameras,
                                   Landmarks& landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<FrameCamId> fcids = {fcid0, fcid1};
  if (!GetTracksInImages(fcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // TODO SHEET 4: Triangulate all new features and add to the map
  // my code:
  // Later I need the trackIDs of shared tracks not in the map
  // So I use new_track_ids

  // Bearing vectors
  opengv::bearingVectors_t vectors_0;
  opengv::bearingVectors_t vectors_1;
  // From the shared features, find the features which are not in the map
  // The features not in the map dont have their track id in landmarks
  for (const auto& shared_track_id : shared_track_ids) {
    if (landmarks.find(shared_track_id) == landmarks.end()) {
      new_track_ids.emplace_back(shared_track_id);
      // For shared features not in the map, find the corresponding point in
      // each of the two frames

      // Find feature id in each frame
      const auto& featureID_0 = feature_tracks.at(shared_track_id).at(fcid0);

      const auto& featureID_1 = feature_tracks.at(shared_track_id).at(fcid1);

      // Find 2d point to this feature id

      const auto& point2D_0 = feature_corners.at(fcid0).corners[featureID_0];
      const auto& point2D_1 = feature_corners.at(fcid1).corners[featureID_1];

      // Bearing vectors need 3D
      auto point3D_0 =
          calib_cam.intrinsics.at(fcid0.cam_id)->unproject(point2D_0);
      auto point3D_1 =
          calib_cam.intrinsics.at(fcid1.cam_id)->unproject(point2D_1);

      vectors_0.emplace_back(point3D_0.normalized());
      vectors_1.emplace_back(point3D_1.normalized());
    }

    // I need an adapter to use opengv's functions
    opengv::relative_pose::CentralRelativeAdapter adapter(vectors_0, vectors_1);

    const auto& transformation_01 =
        cameras.at(fcid0).T_w_c.inverse() * cameras.at(fcid1).T_w_c;
    adapter.setR12(transformation_01.rotationMatrix());
    adapter.sett12(transformation_01.translation());
    // triangulate needs also the index, so I iterate over the size of the
    // bearing_vectors

    for (size_t i = 0; i < new_track_ids.size(); ++i) {
      const auto& triangulated_point =
          opengv::triangulation::triangulate(adapter, i);
      // Output of the method is the 3D point expressed in the first viewpoint.
      // I need it expressed in world coordinates
      // Get transformation from first view to world
      const auto& T_w_c_0 = cameras.at(fcid0).T_w_c;
      const auto& triangulated_point_w = T_w_c_0 * triangulated_point;
      // Save this into landmarks

      const auto& new_trackID = new_track_ids[i];

      landmarks[new_trackID].p = triangulated_point_w;

      // Find the frames that have observed the new landmark

      for (const auto& current_feature_track : feature_tracks.at(new_trackID)) {
        auto frame_id = current_feature_track.first;
        if (cameras.find(frame_id) != cameras.end()) {
          landmarks[new_trackID].obs.emplace(current_feature_track);
        }
      }
    }
  }

  // my code!

  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const FrameCamId& fcid0,
                                       const FrameCamId& fcid1,
                                       const Calibration& calib_cam,
                                       const Corners& feature_corners,
                                       const FeatureTracks& feature_tracks,
                                       Cameras& cameras, Landmarks& landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(fcid0.frame_id == fcid1.frame_id && fcid0.cam_id != fcid1.cam_id)) {
    std::cerr << "Images " << fcid0 << " and " << fcid1
              << " don't form a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // TODO SHEET 4: Initialize scene (add initial cameras and landmarks)

  // Camera camera0{Sophus::SE3d{Eigen::Matrix4d{}}};
  Camera camera0{calib_cam.T_i_c[0]};
  Camera camera1{calib_cam.T_i_c[1]};
  cameras[fcid0] = camera0;
  cameras[fcid1] = camera1;

  int number_new_landmarks =
      add_new_landmarks_between_cams(fcid0, fcid1, calib_cam, feature_corners,
                                     feature_tracks, cameras, landmarks);

  return (number_new_landmarks > 0);
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const FrameCamId& fcid, const std::vector<TrackId>& shared_track_ids,
    const Calibration& calib_cam, const Corners& feature_corners,
    const FeatureTracks& feature_tracks, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& T_w_c, std::vector<TrackId>& inlier_track_ids) {
  inlier_track_ids.clear();

  // TODO SHEET 4: Localize a new image in a given map

  opengv::bearingVectors_t bearing_vectors;
  opengv::points_t points;
  for (const auto& shared_track_id : shared_track_ids) {
    // find corresponding 3D coordinates

    auto landmark = landmarks.at(shared_track_id);
    auto point3d = landmark.p;
    points.push_back(point3d);

    // find corresponding projected 2D coordinates
    auto camID = calib_cam.intrinsics[fcid.cam_id];

    // find corresponding corner 2D coordinates
    auto keypoints = feature_corners.at(fcid);
    auto corners = keypoints.corners;

    auto feature_track = feature_tracks.at(shared_track_id);
    auto featureID = feature_track.at(fcid);
    auto corner_point2d = corners[featureID];
    auto corner_point3d = camID->unproject(corner_point2d);
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
  // Get refined pose
  const auto& refined_tranlation =
      nonlinear_transformation.topRightCorner(3, 1);
  const auto& refined_rotation = nonlinear_transformation.topLeftCorner(3, 3);

  T_w_c = Sophus::SE3d(refined_rotation, refined_tranlation);

  for (size_t i = 0; i < ransac.inliers_.size(); ++i) {
    inlier_track_ids.push_back(shared_track_ids[ransac.inliers_[i]]);
  }
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment(const Corners& feature_corners,
                       const BundleAdjustmentOptions& options,
                       const std::set<FrameCamId>& fixed_cameras,
                       Calibration& calib_cam, Cameras& cameras,
                       Landmarks& landmarks) {
  ceres::Problem problem;
  const std::string cam_model = "ds";
  // TODO SHEET 4: Setup optimization problem

  for (auto& current_element_in_map : landmarks) {
    auto& current_landmark = current_element_in_map.second;
    // Find 3D point
    auto& point_3D_w = current_landmark.p;

    for (const auto& current_observed_feature_track : current_landmark.obs) {
      // Find point 2d and tranformation
      const auto& frameCamID = current_observed_feature_track.first;
      const auto& featureID = current_observed_feature_track.second;
      auto& T_w_c = cameras.at(frameCamID).T_w_c;
      const auto& point_2D = feature_corners.at(frameCamID).corners[featureID];

      auto& intrinsics = calib_cam.intrinsics.at(frameCamID.cam_id);

      problem.AddParameterBlock(T_w_c.data(), Sophus::SE3d::num_parameters,
                                new Sophus::test::LocalParameterizationSE3);

      problem.AddParameterBlock(point_3D_w.data(), 3);

      problem.AddParameterBlock(intrinsics->data(), 8);

      BundleAdjustmentReprojectionCostFunctor* cost_functor =
          new BundleAdjustmentReprojectionCostFunctor(point_2D, cam_model);
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
          BundleAdjustmentReprojectionCostFunctor, 2,
          Sophus::SE3d::num_parameters, 3, 8>(cost_functor);

      if (fixed_cameras.find(frameCamID) != fixed_cameras.end()) {
        problem.SetParameterBlockConstant(T_w_c.data());
      }
      if (!options.optimize_intrinsics) {
        problem.SetParameterBlockConstant(intrinsics->data());
      }

      if (options.use_huber) {
        problem.AddResidualBlock(
            cost_function, new ceres::HuberLoss(options.huber_parameter),
            T_w_c.data(), point_3D_w.data(), intrinsics->data());
      } else {
        problem.AddResidualBlock(cost_function, nullptr, T_w_c.data(),
                                 point_3D_w.data(), intrinsics->data());
      }
    }
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

}  // namespace visnav
