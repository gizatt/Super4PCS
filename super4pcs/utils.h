#ifndef UTILS_H
#define UTILS_H

#include <limits>
#include <cv.h>


namespace FastRegistration {
namespace Utils {

//! \brief Compile time pow
template<typename baseT, typename expoT>
constexpr baseT Pow(baseT base, expoT expo)
{
    return (expo != 0 )? base * Pow(base, expo -1) : 1;
}

template <typename Scalar>
static constexpr Scalar square(const Scalar& x) { return x * x; }


// Compute the closest points between two 3D line segments and obtain the two
// invariants corresponding to the closet points. This is the "intersection"
// point that determines the invariants. Since the 4 points are not exactly
// planar, we use the center of the line segment connecting the two closest
// points as the "intersection".
template <typename Point3D, typename Scalar>
inline Scalar distSegmentToSegment(const Point3D& p1, const Point3D& p2,
                                   const Point3D& q1, const Point3D& q2,
                                   Scalar& invariant1, Scalar& invariant2);


// Select random triangle in P such that its diameter is close to
// max_base_diameter_. This enables to increase the probability of having
// all three points in the inlier set. Return true on success, false if such a
// triangle cannot be found.
template <typename PointContainer, typename Scalar>
inline bool SelectRandomTriangle(int& base1, int& base2, int& base3,
                                 const PointContainer& cloud,
                                 Scalar maxBaseDiameter,
                                 int nbTries);


// Selects a quadrilateral from P and returns the corresponding invariants
// and point indices. Returns true if a quadrilateral has been found, false
// otherwise.
template <typename BaseContainer, typename PointContainer, typename Scalar>
inline bool SelectQuadrilateral(Scalar& invariant1, Scalar& invariant2,
                                int& base1, int& base2, int& base3, int& base4,
                                BaseContainer& base,
                                const PointContainer& cloud,
                                Scalar maxBaseDiameter,
                                int nbTries);


// Takes quadrilateral as a base, computes robust intersection point
// (approximate as the lines might not intersect) and returns the invariants
// corresponding to the two selected lines. The method also updates the order
// of the base base_3D_.
template <typename BaseContainer, typename Scalar>
inline bool TryQuadrilateral(Scalar& invariant1, Scalar& invariant2,
                             int &base1, int &base2, int &base3, int &base4,
                             BaseContainer& base);

} // namespace Utils
} // namespace fastregistration

#include "utils.hpp"
#endif // UTILS_H
