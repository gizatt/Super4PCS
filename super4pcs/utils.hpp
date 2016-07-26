#ifndef UTILS_HPP
#define UTILS_HPP

#ifndef UTILS_H
#include "utils.h"
#endif

namespace FastRegistration {
namespace Utils {

template <typename Point3D, typename Scalar>
Scalar distSegmentToSegment(const Point3D& p1, const Point3D& p2,
                           const Point3D& q1, const Point3D& q2,
                           Scalar& invariant1, Scalar& invariant2) {
  static const Scalar kSmallNumber = 0.0001;
  Point3D u = p2 - p1;
  Point3D v = q2 - q1;
  Point3D w = p1 - q1;
  Scalar a = u.dot(u);
  Scalar b = u.dot(v);
  Scalar c = v.dot(v);
  Scalar d = u.dot(w);
  Scalar e = v.dot(w);
  Scalar f = a * c - b * b;
  // s1,s2 and t1,t2 are the parametric representation of the intersection.
  // they will be the invariants at the end of this simple computation.
  Scalar s1 = 0.0;
  Scalar s2 = f;
  Scalar t1 = 0.0;
  Scalar t2 = f;

  if (f < kSmallNumber) {
    s1 = 0.0;
    s2 = 1.0;
    t1 = e;
    t2 = c;
  } else {
    s1 = (b * e - c * d);
    t1 = (a * e - b * d);
    if (s1 < 0.0) {
      s1 = 0.0;
      t1 = e;
      t2 = c;
    } else if (s1 > s2) {
      s1 = s2;
      t1 = e + b;
      t2 = c;
    }
  }

  if (t1 < 0.0) {
    t1 = 0.0;
    if (-d < 0.0)
      s1 = 0.0;
    else if (-d > a)
      s1 = s2;
    else {
      s1 = -d;
      s2 = a;
    }
  } else if (t1 > t2) {
    t1 = t2;
    if ((-d + b) < 0.0)
      s1 = 0;
    else if ((-d + b) > a)
      s1 = s2;
    else {
      s1 = (-d + b);
      s2 = a;
    }
  }

  invariant1 = ((std::abs(s1) < kSmallNumber) ? Scalar(0) : (s1 / s2));
  invariant2 = ((std::abs(t1) < kSmallNumber) ? Scalar(0) : (t1 / t2));

  return cv::norm(w + (invariant1 * u) - (invariant2 * v));
}


// Selects a random triangle in the set P (then we add another point to keep the
// base as planar as possible). We apply a simple heuristic that works in most
// practical cases. The idea is to accept maximum distance, computed by the
// estimated overlap, multiplied by the diameter of P, and try to have
// a triangle with all three edges close to this distance. Wide triangles helps
// to make the transformation robust while too large triangles makes the
// probability of having all points in the inliers small so we try to trade-off.
template <typename PointContainer, typename Scalar>
bool SelectRandomTriangle(int& base1, int& base2, int& base3,
                          const PointContainer& cloud,
                          Scalar maxBaseDiameter,
                          int nbTries) {

  int number_of_points = cloud.size();
  base1 = base2 = base3 = -1;

  // Pick the first point at random.
  int first_point = rand() % number_of_points;

  // Try fixed number of times retaining the best other two.
  Scalar best_wide = 0.0;
  for (int i = 0; i < nbTries; ++i) {
    // Pick and compute
    int second_point = rand() % number_of_points;
    int third_point = rand() % number_of_points;
    cv::Point3f u = cloud[second_point] - cloud[first_point];
    cv::Point3f w = cloud[third_point] - cloud[first_point];
    // We try to have wide triangles but still not too large.
    Scalar how_wide = cv::norm(u.cross(w));
    if (how_wide > best_wide && cv::norm(u) < maxBaseDiameter &&
        cv::norm(w) < maxBaseDiameter) {
      best_wide = how_wide;
      base1 = first_point;
      base2 = second_point;
      base3 = third_point;
    }
  }
  if (base1 == -1 || base2 == -1 || base3 == -1)
    return false;
  else
    return true;
}


// Selects a good base from P and computes its invariants. Returns false if
// a good planar base cannot can be found.
template <typename BaseContainer, typename PointContainer, typename Scalar>
bool SelectQuadrilateral(Scalar &invariant1, Scalar &invariant2,
                         int &base1, int &base2, int &base3, int &base4,
                         BaseContainer& base,
                         const PointContainer& cloud,
                         Scalar maxBaseDiameter,
                         int nbTries) {

  static const Scalar kBaseTooSmall = 0.2;
  int current_trial = 0;

  // Try fix number of times.
  while (current_trial < nbTries) {
    // Select a triangle if possible. otherwise fail.
    if (!SelectRandomTriangle(base1, base2, base3,
                              cloud, maxBaseDiameter,
                              nbTries)){
      return false;
    }

    base[0] = cloud[base1];
    base[1] = cloud[base2];
    base[2] = cloud[base3];

    // The 4th point will be a one that is close to be planar to the other 3
    // while still not too close to them.
    const auto& x1 = base[0].x;
    const auto& y1 = base[0].y;
    const auto& z1 = base[0].z;
    const auto& x2 = base[1].x;
    const auto& y2 = base[1].y;
    const auto& z2 = base[1].z;
    const auto& x3 = base[2].x;
    const auto& y3 = base[2].y;
    const auto& z3 = base[2].z;

    // Fit a plan.
    Scalar denom = (-x3 * y2 * z1 + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 -
                    x2 * y1 * z3 + x1 * y2 * z3);

    if (denom != 0) {
      Scalar A =
          (-y2 * z1 + y3 * z1 + y1 * z2 - y3 * z2 - y1 * z3 + y2 * z3) / denom;
      Scalar B =
          (x2 * z1 - x3 * z1 - x1 * z2 + x3 * z2 + x1 * z3 - x2 * z3) / denom;
      Scalar C =
          (-x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3) / denom;
      base4 = -1;
      Scalar best_distance = std::numeric_limits<Scalar>::max();
      // Go over all points in P.
      for (unsigned int i = 0; i < cloud.size(); ++i) {
        Scalar d1 = cv::norm(cloud[i] - cloud[base1]);
        Scalar d2 = cv::norm(cloud[i] - cloud[base2]);
        Scalar d3 = cv::norm(cloud[i] - cloud[base3]);
        Scalar too_small = maxBaseDiameter * kBaseTooSmall;
        if (d1 >= too_small && d2 >= too_small && d3 >= too_small) {
          // Not too close to any of the first 3.
          Scalar distance =
              std::abs(A * cloud[i].x + B * cloud[i].y + C * cloud[i].z - 1.0);
          // Search for the most planar.
          if (distance < best_distance) {
            best_distance = distance;
            base4 = int(i);
          }
        }
      }
      // If we have a good one we can quit.
      if (base4 != -1) {
        base[3] = cloud[base4];
        TryQuadrilateral(invariant1, invariant2, base1, base2, base3, base4, base);
        return true;
      }
    }
    current_trial++;
  }

  // We failed to find good enough base..
  return false;
}


// Try the current base in P and obtain the best pairing, i.e. the one that
// gives the smaller distance between the two closest points. The invariants
// corresponding the the base pairing are computed.
template <typename BaseContainer, typename Scalar>
bool TryQuadrilateral(Scalar &invariant1, Scalar &invariant2,
                      int &base1, int &base2, int &base3, int &base4,
                      BaseContainer& base) {
  float min_distance = FLT_MAX;
  int best1, best2, best3, best4;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (i == j) continue;
      int k = 0;
      while (k == i || k == j) k++;
      int l = 0;
      while (l == i || l == j || l == k) l++;
      Scalar local_invariant1;
      Scalar local_invariant2;
      // Compute the closest points on both segments, the corresponding
      // invariants and the distance between the closest points.
      Scalar segment_distance = Utils::distSegmentToSegment(
                  base[i], base[j], base[k], base[l],
                  local_invariant1, local_invariant2);
      // Retail the smallest distance and the best order so far.
      std::cout << "segment_distance: " << segment_distance<<std::endl;
      std::cout << "min_distance:     " << min_distance<<std::endl;
      if (segment_distance < min_distance) {
        min_distance = segment_distance;
        best1 = i;
        best2 = j;
        best3 = k;
        best4 = l;
        invariant1 = local_invariant1;
        invariant2 = local_invariant2;

        std::cout << "Swap: "
                  << best1 << " "
                  << best2 << " "
                  << best3 << " "
                  << best4 << " "
                  << invariant1 << " "
                  << invariant2 << std::endl;
      }
    }
  }
  BaseContainer tmp = base;
  base[0] = tmp[best1];
  base[1] = tmp[best2];
  base[2] = tmp[best3];
  base[3] = tmp[best4];

  std::array<int, 4> tmpId = {base1, base2, base3, base4};
  base1 = tmpId[best1];
  base2 = tmpId[best2];
  base3 = tmpId[best3];
  base4 = tmpId[best4];

  std::cout << "Basis: "
            << base1 << " "
            << base2 << " "
            << base3 << " "
            << base4 << std::endl;

  return true;
}



} // namespace Utils
} // namespace FastRegistration

#endif // UTILS_HPP
