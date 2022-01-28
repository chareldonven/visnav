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

#include <sophus/se3.hpp>

#include <visnav/common_types.h>
namespace ex1_intern {

template <class T>
auto epsilon = std::numeric_limits<T>::min();

template <class T>
Eigen::Matrix<T, 3, 3> hat_operator(const Eigen::Matrix<T, 3, 1>& xi) {
  Eigen::Matrix<T, 3, 3> xi_hat{};
  xi_hat << 0, -xi(2), xi(1), xi(2), 0, -xi(0), -xi(1), xi(0), 0;
  return xi_hat;
}

}  // namespace ex1_intern
namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  auto norm = xi.norm();

  auto sin_of_norm = std::sin(norm);
  auto cos_of_norm = std::cos(norm);

  auto xi_hat = ex1_intern::hat_operator(xi);

  return Eigen::Matrix<T, 3, 3>::Identity() +
         sin_of_norm / (norm + ex1_intern::epsilon<T>)*xi_hat +
         (1 - cos_of_norm) / (norm * norm + ex1_intern::epsilon<T>)*xi_hat *
             xi_hat;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  auto trace = mat.trace();
  auto norm = std::acos((trace - 1) / 2);

  Eigen::Matrix<T, 3, 1> r{(mat(2, 1) - mat(1, 2)), (mat(0, 2) - mat(2, 0)),
                           (mat(1, 0) - mat(0, 1))};

  auto constant_factor = (norm + ex1_intern::epsilon<T>) /
                         (2 * std::sin(norm) + 2 * ex1_intern::epsilon<T>);

  return Eigen::Matrix<T, 3, 1>(constant_factor * r);
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  Eigen::Matrix<T, 3, 1> w = xi.tail(3);

  auto w_hat = ex1_intern::hat_operator(w);
  auto norm_of_w = w.norm();

  auto cos_term = (1 - std::cos(norm_of_w) + ex1_intern::epsilon<T>) /
                  (norm_of_w * norm_of_w + 2 * ex1_intern::epsilon<T>);

  auto sin_term =
      (norm_of_w - std::sin(norm_of_w) + ex1_intern::epsilon<T>) /
      (norm_of_w * norm_of_w * norm_of_w + 6 * ex1_intern::epsilon<T>);

  auto jacobian_matrix = Eigen::Matrix<T, 3, 3>::Identity() + cos_term * w_hat +
                         sin_term * w_hat * w_hat;

  Eigen::Matrix<T, 3, 1> v = xi.head(3);
  Eigen::Matrix<T, 3, 1> translation_term = jacobian_matrix * v;

  Eigen::Matrix<T, 4, 4> result{};
  result.topLeftCorner(3, 3) = user_implemented_expmap(w);
  result.topRightCorner(3, 1) = translation_term;
  result.bottomRows(1).setZero();
  result(3, 3) = 1;

  return result;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  Eigen::Matrix<T, 3, 3> rotation_term = mat.topLeftCorner(3, 3);

  auto w = user_implemented_logmap(rotation_term);
  auto w_hat = ex1_intern::hat_operator(w);

  auto norm_of_w = w.norm();

  auto constant_term = (1 + ex1_intern::epsilon<T>) /
                       (norm_of_w * norm_of_w + 2 * ex1_intern::epsilon<T>);
  constant_term -=
      (1 + std::cos(norm_of_w) - ex1_intern::epsilon<T>) /
      (2 * norm_of_w * std::sin(norm_of_w) + 4 * ex1_intern::epsilon<T>);

  auto inverseJ = Eigen::Matrix<T, 3, 3>::Identity() - 0.5 * w_hat +
                  constant_term * w_hat * w_hat;

  Eigen::Matrix<T, 3, 1> translation_term = mat.topRightCorner(3, 1);

  Eigen::Matrix<T, 3, 1> v = inverseJ * translation_term;

  Eigen::Matrix<T, 6, 1> result{};
  result.head(3) = v;
  result.tail(3) = w;
  return result;
}

}  // namespace visnav
