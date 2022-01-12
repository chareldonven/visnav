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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 3: compute essential matrix

  // Normalize the translation vector
  const Eigen::Vector3d t_0_1_normalized = t_0_1.normalized();
  // See PDF file for the derivation of the essential matrix: E = t^ * R
  Eigen::Matrix3d skew_matrix_t;
  skew_matrix_t << 0, -t_0_1_normalized[2], t_0_1_normalized[1],
      t_0_1_normalized[2], 0, -t_0_1_normalized[0], -t_0_1_normalized[1],
      t_0_1_normalized[0], 0;
  E = skew_matrix_t * R_0_1;
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // TODO SHEET 3: determine inliers and store in md.inliers
    const auto& p0_3d = cam1->unproject(p0_2d);
    const auto& p1_3d = cam2->unproject(p1_2d);
    // Compute the constraint
    auto result = p0_3d.transpose() * E * p1_3d;
    // For inliers the constraint should be fulfilled up to the given threshold.
    if (std::abs(result) < epipolar_error_threshold) {
      md.inliers.emplace_back(md.matches[j].first, md.matches[j].second);
    }
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();
  md.T_i_j = Sophus::SE3d();

  // TODO SHEET 3:
  // Run RANSAC with using opengv's CentralRelativePose
  // and
  // store
  // the final inlier indices in md.inliers
  // and the final relative pose in md.T_i_j (normalize translation).

  // If the number of inliers is smaller than ransac_min_inliers,
  // leave md.inliers empty.
  // Note that if the initial RANSAC
  // was successful, you should do non-linear refinement of the model parameters
  // using all inliers,
  // and then re-estimate the inlier set with the refined model parameters.
  opengv::bearingVectors_t vectors_P;
  opengv::bearingVectors_t vectors_Q;
  // Iterate over all matches to initialize the bearing vectors for the adapter
  for (size_t i = 0; i < md.matches.size(); i++) {
    // Get the indices of the current match
    auto index_P = md.matches[i].first;
    auto index_Q = md.matches[i].second;
    // Get the respective 2D corner points
    auto p_2d_P = kd1.corners[index_P];
    auto p_2d_Q = kd2.corners[index_Q];
    // Find the respective 3D point to the corner point
    auto p_3d_P = cam1->unproject(p_2d_P);
    auto p_3d_Q = cam2->unproject(p_2d_Q);

    vectors_P.emplace_back(p_3d_P);
    vectors_Q.emplace_back(p_3d_Q);
  }
  // Define the central relative adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(vectors_P, vectors_Q);
  // Create a Ransac object
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  // Create a CentralRelativePoseSacProblem with STEWENIUS 5 point algorithm
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::STEWENIUS));
  // Run Ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  // What is a good number of iterations?
  ransac.max_iterations_ = 10;
  ransac.computeModel();
  // Get the results:
  const auto& translation = ransac.model_coefficients_.topRightCorner(3, 1);
  const auto& rotation = ransac.model_coefficients_.topLeftCorner(3, 3);

  adapter.sett12(translation);
  adapter.setR12(rotation);
  // Nonlinear optimization to refine the model parameters using ALL inliers
  const opengv::transformation_t nonlinear_transformation =
      opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);
  // Update the set of inliers using the refined relative pose
  // Select the inliers that are within threshold from the model
  // sac_model_->selectWithinDistance( model_coefficients, threshold_, inliers
  // );
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac_thresh, ransac.inliers_);
  // Get refined pose
  // Normalize translation vector
  const auto& refined_tranlation =
      nonlinear_transformation.topRightCorner(3, 1).normalized();
  const auto& refined_rotation = nonlinear_transformation.topLeftCorner(3, 3);

  // Store the final refined relative pose.
  md.T_i_j = Sophus::SE3d(refined_rotation, refined_tranlation);

  // Store the refined set of inliers
  // Why do I need ransac_min_inliers?

  if (static_cast<long>(ransac.inliers_.size()) >= ransac_min_inliers) {
    for (size_t i = 0; i < ransac.inliers_.size(); i++) {
      md.inliers.emplace_back(md.matches[ransac.inliers_[i]]);
    }
  }
}
}  // namespace visnav
