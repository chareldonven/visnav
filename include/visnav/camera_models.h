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

#include <memory>
#include <ceres/jet.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

template <typename Scalar>
class AbstractCamera;

template <typename Scalar>
class PinholeCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  PinholeCamera() = default;
  PinholeCamera(const VecN& p) : param(p) {}

  static PinholeCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0, 0, 0, 0;
    PinholeCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "pinhole"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;
    res.x() = fx * x / z + cx;
    res.y() = fy * y / z + cy;
    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    Vec3 res;

    auto mx = (p.x() - cx) / fx;
    auto my = (p.y() - cy) / fy;
    auto d = ceres::sqrt(mx * mx + my * my + Scalar(1));
    res.x() = mx;
    res.y() = my;
    res.z() = Scalar(1);
    return res / d;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar = double>
class ExtendedUnifiedCamera : public AbstractCamera<Scalar> {
 public:
  // NOTE: For convenience for serialization and handling different camera
  // models in ceres functors, we use the same parameter vector size for all of
  // them, even if that means that for some certain entries are unused /
  // constant 0.
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  ExtendedUnifiedCamera() = default;
  ExtendedUnifiedCamera(const VecN& p) : param(p) {}

  static ExtendedUnifiedCamera getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 500, 0.5 * 500, 319.5, 239.5, 0.51231234, 0.9, 0, 0;
    ExtendedUnifiedCamera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static const std::string getName() { return "eucm"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];
    const Scalar& beta = param[5];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    auto d = ceres::sqrt(beta * (x * x + y * y) + z * z);
    auto denominator = alpha * d + (Scalar(1) - alpha) * z;
    res.x() = fx * x / denominator + cx;
    res.y() = fy * y / denominator + cy;

    return res;
  }

  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];
    const Scalar& beta = param[5];

    Vec3 res;

    auto mx = (p.x() - cx) / fx;
    auto my = (p.y() - cy) / fy;
    auto r_squared = mx * mx + my * my;
    auto mz_numerator = Scalar(1) - beta * alpha * alpha * r_squared;
    auto mz_denominator =
        alpha * ceres::sqrt(Scalar(1) - (Scalar(2) * alpha - Scalar(1)) * beta *
                                            r_squared) +
        (Scalar(1) - alpha);
    auto mz = mz_numerator / mz_denominator;

    auto d = ceres::sqrt(r_squared + mz * mz);
    res.x() = mx;
    res.y() = my;
    res.z() = mz;
    return res / d;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar>
class DoubleSphereCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  DoubleSphereCamera() = default;
  DoubleSphereCamera(const VecN& p) : param(p) {}

  static DoubleSphereCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * 1.48785, 0,
        0;
    DoubleSphereCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "ds"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;
    auto d1 = ceres::sqrt((x * x + y * y + z * z));
    //
    auto factor = xi * d1 + z;
    auto d2 = ceres::sqrt(x * x + y * y + factor * factor);

    auto denominator = alpha * d2 + (Scalar(1) - alpha) * factor;
    res.x() = fx * x / denominator + cx;
    res.y() = fy * y / denominator + cy;

    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    Vec3 res;

    auto mx = (p.x() - cx) / fx;
    auto my = (p.y() - cy) / fy;
    auto r_squared = mx * mx + my * my;
    auto mz_numerator = Scalar(1) - alpha * alpha * r_squared;
    auto mz_denominator =
        alpha * ceres::sqrt(Scalar(1) -
                            (Scalar(2) * alpha - Scalar(1)) * r_squared) +
        (Scalar(1) - alpha);
    auto mz = mz_numerator / mz_denominator;

    auto d_numerator =
        mz * xi + ceres::sqrt(mz * mz + (Scalar(1) - xi * xi) * r_squared);
    auto d_denominator = mz * mz + r_squared;
    auto d = d_numerator / d_denominator;

    res.x() = mx;
    res.y() = my;
    res.z() = mz;
    res *= d;

    res.z() -= xi;
    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar = double>
class KannalaBrandt4Camera : public AbstractCamera<Scalar> {
 public:
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  KannalaBrandt4Camera() = default;
  KannalaBrandt4Camera(const VecN& p) : param(p) {}

  static KannalaBrandt4Camera getTestProjections() {
    VecN vec1;
    vec1 << 379.045, 379.008, 505.512, 509.969, 0.00693023, -0.0013828,
        -0.000272596, -0.000452646;
    KannalaBrandt4Camera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "kb4"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;
    auto r = ceres::sqrt(x * x + y * y);
    auto theta = ceres::atan2(r, z);

    // See description of calculate_polynom()
    const std::array<Scalar, 6> polynom_coefficients{Scalar(0), Scalar(1), k1,
                                                     k2,        k3,        k4};
    auto last_index_of_coefficients = polynom_coefficients.size() - 1;
    auto d = calculate_polynom(theta, Scalar(0), last_index_of_coefficients,
                               polynom_coefficients[last_index_of_coefficients],
                               polynom_coefficients);

    if (r == Scalar(0)) {
      // to avoid division by 0
      res.x() = cx;
      res.y() = cy;
    } else {
      res.x() = fx * d * x / r + cx;
      res.y() = fy * d * y / r + cy;
    }

    return res;
  }
  /*
   * This method calculates for the value of the polynom for KannalaBrandt4
   * x is the point for which the polynom should be evaluated
   * r is used for the unprojection method, when we search for x* = d^(-1)(r)
   * iteration indicates the current recursion step
   * current_term is needed for end recursion and contains the factor with which
   x^2 should be multiplied in each step for Horner's method
   *    0 + x(1 + x^2 (k1 + x^2(...)))
   *    The first coefficient is only a place holder in order for coefficient[1]
   to be 1
   * coefficients are the coefficients of the KannalaBrandt4 polynom: 1 * x + k1
   x^3 + ...


  */
  Scalar calculate_polynom(Scalar x, Scalar r, int iteration,
                           Scalar current_term,
                           const std::array<Scalar, 6>& coefficients) const {
    auto x_squared = x * x;
    // unless iteration step == 1 then evaluate k(i - 1) + x^2 k(i)
    auto next_term = current_term;
    if (iteration > 1) {
      next_term = next_term * x_squared + coefficients[iteration - 1];
    }
    // if iteration step == 1 then evaluate x * current_term - r
    if (iteration == 1) {
      next_term = next_term * x - r;
      return next_term;
    }
    return calculate_polynom(x, r, (iteration - 1), next_term, coefficients);
  }
  /*
   * Similar to calculate_polynom
   * Uses Horner's method to evaluate the derivative of the KannalaBrandt
   polynom
   Different from the calculate_polynom method, the first argument is x_squared
   *

*/
  Scalar calculate_derivative(Scalar x_squared, int iteration,
                              Scalar current_term,
                              const std::array<Scalar, 5>& coefficients) const {
    auto next_term = coefficients[iteration - 1] + x_squared * current_term;

    // For the derivative the Horner's method looks like this: 1 + x^2(k1 +
    // x^2(...))
    if (iteration == 1) {
      return next_term;
    }
    return calculate_derivative(x_squared, (iteration - 1), next_term,
                                coefficients);
  }

  /*  This method uses Newtow's method to find the root of the KannalaBradt4
     polynom - r, where r is x* = d^(-1)(r)
     */
  Scalar find_root(Scalar start_x, Scalar r, int iterations_number,
                   const std::array<Scalar, 6>& polynom_coefficients,
                   const std::array<Scalar, 5>& derivative_coefficients) const {
    auto last_index_polynom_coefficients = polynom_coefficients.size() - 1;
    auto last_index_derivative_coefficients =
        derivative_coefficients.size() - 1;
    auto next_x =
        start_x -
        calculate_polynom(start_x, r, last_index_polynom_coefficients,
                          polynom_coefficients[last_index_polynom_coefficients],
                          polynom_coefficients) /
            calculate_derivative(
                start_x * start_x, last_index_derivative_coefficients,
                derivative_coefficients[last_index_derivative_coefficients],
                derivative_coefficients);
    if (iterations_number == 0) {
      return next_x;
    }
    return find_root(next_x, r, iterations_number - 1, polynom_coefficients,
                     derivative_coefficients);
  }
  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    Vec3 res;

    auto mx = (p.x() - cx) / fx;
    auto my = (p.y() - cy) / fy;
    auto r = ceres::sqrt(mx * mx + my * my);
    if (r == Scalar(0)) {
      res.x() = Scalar(0);
      res.y() = Scalar(0);
      res.z() = Scalar(1);
    } else {
      const Scalar x_0(1);
      const std::array<Scalar, 6> polynom_coefficients{
          Scalar(0), Scalar(1), k1, k2, k3, k4};

      const std::array<Scalar, 5> derivative_coefficients{
          Scalar(1), Scalar(3) * k1, Scalar(5) * k2, Scalar(7) * k3,
          Scalar(9) * k4};

      auto theta =
          find_root(x_0, r, 6, polynom_coefficients, derivative_coefficients);

      auto sin_theta = ceres::sin(theta);
      auto cos_theta = ceres::cos(theta);

      res.x() = sin_theta * mx / r;
      res.y() = sin_theta * my / r;
      res.z() = cos_theta;
    }
    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar>
class AbstractCamera {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  virtual ~AbstractCamera() = default;

  virtual Scalar* data() = 0;

  virtual const Scalar* data() const = 0;

  virtual Vec2 project(const Vec3& p) const = 0;

  virtual Vec3 unproject(const Vec2& p) const = 0;

  virtual std::string name() const = 0;

  virtual const VecN& getParam() const = 0;

  inline int width() const { return width_; }
  inline int& width() { return width_; }
  inline int height() const { return height_; }
  inline int& height() { return height_; }

  static std::shared_ptr<AbstractCamera> from_data(const std::string& name,
                                                   const Scalar* sIntr) {
    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(new PinholeCamera<Scalar>(intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

  // Loading from double sphere initialization
  static std::shared_ptr<AbstractCamera> initialize(const std::string& name,
                                                    const Scalar* sIntr) {
    Eigen::Matrix<Scalar, 8, 1> init_intr;

    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;

      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(init_intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new PinholeCamera<Scalar>(init_intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(init_intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();
      init_intr[4] = 0.5;
      init_intr[5] = 1;

      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(init_intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

 private:
  // image dimensions
  int width_ = 0;
  int height_ = 0;
};

}  // namespace visnav
