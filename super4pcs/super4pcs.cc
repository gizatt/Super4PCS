// Copyright 2014 Nicolas Mellado
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -------------------------------------------------------------------------- //
//
// Authors: Nicolas Mellado, Dror Aiger
//
// An implementation of the Super 4-points Congruent Sets (Super 4PCS) 
// algorithm presented in:
//
// Super 4PCS: Fast Global Pointcloud Registration via Smart Indexing
// Nicolas Mellado, Dror Aiger, Niloy J. Mitra
// Symposium on Geometry Processing 2014.
//
// Data acquisition in large-scale scenes regularly involves accumulating 
// information across multiple scans. A common approach is to locally align scan 
// pairs using Iterative Closest Point (ICP) algorithm (or its variants), but 
// requires static scenes and small motion between scan pairs. This prevents 
// accumulating data across multiple scan sessions and/or different acquisition
// modalities (e.g., stereo, depth scans). Alternatively, one can use a global
// registration algorithm allowing scans to be in arbitrary initial poses. The 
// state-of-the-art global registration algorithm, 4PCS, however has a quadratic
// time complexity in the number of data points. This vastly limits its 
// applicability to acquisition of large environments. We present Super 4PCS for
// global pointcloud registration that is optimal, i.e., runs in linear time (in 
// the number of data points) and is also output sensitive in the complexity of 
// the alignment problem based on the (unknown) overlap across scan pairs. 
// Technically, we map the algorithm as an 'instance problem' and solve it 
// efficiently using a smart indexing data organization. The algorithm is 
// simple, memory-efficient, and fast. We demonstrate that Super 4PCS results in
// significant speedup over alternative approaches and allows unstructured 
// efficient acquisition of scenes at scales previously not possible. Complete 
// source code and datasets are available for research use at 
// http://geometry.cs.ucl.ac.uk/projects/2014/super4PCS/.

#include "4pcs.h"

#include "Eigen/Core"
#include "Eigen/Geometry"                 // MatrixBase.homogeneous()
#include "Eigen/SVD"                      // Transform.computeRotationScaling()

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include "accelerators/pairExtraction/bruteForceFunctor.h"
#include "accelerators/pairExtraction/intersectionFunctor.h"
#include "accelerators/pairExtraction/intersectionPrimitive.h"
#include "accelerators/normalset.h"
#include "accelerators/normalHealSet.h"
#include "accelerators/kdtree.h"
#include "accelerators/bbox.h"
#include "sampling.h"
#include "utils.h"

#include "pairCreationFunctor.h"

#include <fstream>
#include <array>
#include <time.h>

#define sqr(x) ((x)*(x))
#define norm2(p) (sqr(p.x)+sqr(p.y)+sqr(p.z))

#ifdef TEST_GLOBAL_TIMINGS
#   include "utils/timer.h"
#endif

//#define MULTISCALE

namespace FastRegistration {

using namespace std;

class MatchSuper4PCSImpl {
public:
    typedef double Scalar;

private:
    typedef PairCreationFunctor<Scalar>::PairsVector PairsVector;

#ifdef TEST_GLOBAL_TIMINGS

    Scalar totalTime;
    Scalar kdTreeTime;
    Scalar verifyTime;

    using Timer = Super4PCS::Utils::Timer;

#endif
 public:
  explicit MatchSuper4PCSImpl(const Match4PCSOptions& options)
      : number_of_trials_(0),
        max_base_diameter_(-1),
        P_mean_distance_(1.0),
        best_LCP_(0.0F),
        options_(options),
        pcfunctor_(options_, sampled_Q_3D_) {
    base_3D_.resize(4);
  }

  ~MatchSuper4PCSImpl() {
    Clear();
  }

  void Clear() {
  }
  // Computes an approximation of the best LCP (directional) from Q to P
  // and the rigid transformation that realizes it. The input sets may or may
  // not contain normal information for any point.
  // @param [in] P The first input set.
  // @param [in] Q The second input set.
  // as a fraction of the size of P ([0..1]).
  // @param [out] transformation Rigid transformation matrix (4x4) that brings
  // Q to the (approximate) optimal LCP.
  // @return the computed LCP measure.
  // The method updates the coordinates of the second set, Q, applying
  // the found transformation.
  Scalar ComputeTransformation(const std::vector<Point3D>& P,
                              std::vector<Point3D>* Q, cv::Mat* transformation);

 private:
  // Private data contains parameters and internal variables that are computed
  // and change during the match computation. All parameters have default
  // values.

  // Internal data members.

  // Number of trials. Every trial picks random base from P.
  int number_of_trials_;
  // Maximum base diameter. It is computed automatically from the diameter of
  // P and the estimated overlap and used to limit the distance between the
  // points in the base in P so that the probability to have all points in
  // the base as inliers is increased.
  Scalar max_base_diameter_;
  // The diameter of P.
  Scalar P_diameter_;
  // KdTree used to compute the LCP
  Super4PCS::KdTree<Scalar> kd_tree_;
  // Mean distance between points and their nearest neighbor in the set P.
  // Used to normalize the "delta" which is given in terms of this distance.
  Scalar P_mean_distance_;
  // The transformation matrix by wich we transform Q to P
  Eigen::Matrix<Scalar, 4, 4> transform_;
  // Quad centroids in first and second clouds
  // They are used temporarily and makes the transformations more robust to
  // noise. At the end, the direct transformation applied as a 4x4 matrix on
  // every points in Q is computed and returned.
  Eigen::Matrix<Scalar, 3, 1> qcentroid1_, qcentroid2_;
  // The points in the base (indices to P). It is being updated in every
  // RANSAC iteration.
  int base_[4];
  // The current congruent 4 points from Q. Every RANSAC iteration the
  // algorithm examines a set of such congruent 4-points from Q and retains
  // the best from them (the one that realizes the best LCP).
  int current_congruent_[4];
  // Sampled P (3D coordinates).
  std::vector<Point3D> sampled_P_3D_;
  // Sampled Q (3D coordinates).
  std::vector<Point3D> sampled_Q_3D_;
  // The 3D points of the base.
  std::vector<Point3D> base_3D_;
  // The copy of the input Q. We transform Q to match P and returned the
  // transformed version.
  std::vector<Point3D> Q_copy_;
  // The centroid of P.
  cv::Point3f centroid_P_;
  // The centroid of Q.
  cv::Point3f centroid_Q_;
  // The best LCP (Largest Common Point) fraction so far.
  Scalar best_LCP_;
  // Current trial.
  int current_trial_;
  // Parameters.
  Match4PCSOptions options_;

  PairCreationFunctor<Scalar> pcfunctor_;

  // Private member functions.

  // Tries one base and finds the best transformation for this base.
  // Returns true if the achieved LCP is greater than terminate_threshold_,
  // else otherwise.
  bool TryOneBase();

  // Constructs pairs of points in Q, corresponding to a single pair in the
  // in basein P.
  // @param [in] pair_distance The distance between the pairs in P that we have
  // to match in the pairs we select from Q.
  // @param [in] pair_normal_distance The angle between the normals of the pair
  // in P.
  // @param [in] pair_distance_epsilon Tolerance on the pair distance. We allow
  // candidate pair in Q to have distance of
  // pair_distance+-pair_distance_epsilon.
  // @param [in] base_point1 The index of the first point in P.
  // @param [in] base_point2 The index of the second point in P.
  // @param [out] pairs A set of pairs in Q that match the pair in P with
  // respect to distance and normals, up to the given tolerance.
  void
  ExtractPairs(Scalar pair_distance, Scalar pair_normals_angle,
                       Scalar pair_distance_epsilon, int base_point1,
                       int base_point2,
                       PairsVector* pairs);

  // For each randomly picked base, verifies the computed transformation by
  // computing the number of points that this transformation brings near points
  // in Q. Returns the current LCP. R is the rotation matrix, (tx,ty,tz) is
  // the translation vector and (cx,cy,cz) is the center of transformation.template <class MatrixDerived>
  Scalar Verify(const Eigen::Matrix<Scalar, 4, 4>& mat);

  // Computes the mean distance between point in Q and its nearest neighbor.
  Scalar MeanDistance();

  // Finds congruent candidates in the set Q, given the invariants and threshold
  // distances. Returns true if a non empty set can be found, false otherwise.
  // @param invariant1 [in] The first invariant corresponding to the set P_pairs
  // of pairs, previously extracted from Q.
  // @param invariant2 [in] The second invariant corresponding to the set
  // Q_pairs of pairs, previously extracted from Q.
  // @param [in] distance_threshold1 The distance for verification.
  // @param [in] distance_threshold2 The distance for matching middle points due
  // to the invariants (See the paper for e1, e2).
  // @param [in] P_pairs The first set of pairs.
  // @param [in] Q_pairs The second set of pairs.
  // @param [in] Q_pointsFirst Point coordinates for the pairs first id
  // @param [in] Q_pointsSecond Point coordinates for the pairs second id
  // @param [out] quadrilaterals The set of congruent quadrilateral. In fact,
  // it's a super set from which we extract the real congruent set.
  bool FindCongruentQuadrilateralsFast(Scalar invariant1, Scalar invariant2,
                                       Scalar distance_threshold1,
                                       Scalar distance_threshold2,
                                       const PairsVector& P_pairs,
                                       const PairsVector& Q_pairs,
                                       const std::vector<Point3D>& Q_points,
                                       std::vector<Quadrilateral>* quadrilaterals);

  // Computes the best rigid transformation between three corresponding pairs.
  // The transformation is characterized by rotation matrix, translation vector
  // and a center about which we rotate. The set of pairs is potentially being
  // updated by the best permutation of the second set. Returns the RMS of the
  // fit. The method is being called with 4 points but it applies the fit for
  // only 3 after the best permutation is selected in the second set (see
  // bellow). This is done because the solution for planar points is much
  // simpler.
  // The method is the closed-form solution by Horn:
  // people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf
  bool ComputeRigidTransformation(const vector< pair<Point3D, Point3D> >& pairs,
                                  const Eigen::Matrix<Scalar, 3, 1>& centroid1,
                                  Eigen::Matrix<Scalar, 3, 1> centroid2,
                                  Scalar max_angle,
                                  Eigen::Matrix<Scalar, 4, 4> &transform,
                                  Scalar& rms_,
                                  bool computeScale );

  // Initializes the data structures and needed values before the match
  // computation.
  // @param [in] point_P First input set.
  // @param [in] point_Q Second input set.
  // expected to be in the inliers.
  void Initialize(const std::vector<Point3D>& P, const std::vector<Point3D>& Q);

  // Performs n RANSAC iterations, each one of them containing base selection,
  // finding congruent sets and verification. Returns true if the process can be
  // terminated (the target LCP was obtained or the maximum number of trials has
  // been reached), false otherwise.
  bool Perform_N_steps(int n, cv::Mat* transformation, std::vector<Point3D>* Q);
};

// Finds congruent candidates in the set Q, given the invariants and threshold
// distances.
bool MatchSuper4PCSImpl::FindCongruentQuadrilateralsFast(
    Scalar invariant1, Scalar invariant2, Scalar distance_threshold1,
    Scalar distance_threshold2, const std::vector<std::pair<int, int>>& P_pairs,
    const std::vector<std::pair<int, int>>& Q_pairs,
    const std::vector<Point3D>& Q_points,
    std::vector<Quadrilateral>* quadrilaterals) {

  // Compute the angle formed by the two vectors of the basis
  Point3D b1 = base_3D_[1] - base_3D_[0];  b1.normalize();
  Point3D b2 = base_3D_[3] - base_3D_[2];  b2.normalize();
  Scalar alpha = /*std::abs(*/b1.dot(b2)/*)*/;

  // 1. Datastructure construction
  typedef PairCreationFunctor<Scalar>::Point Point;
  const Scalar eps = pcfunctor_.getNormalizedEpsilon(
    std::min(distance_threshold1, distance_threshold2));
  typedef Super4PCS::IndexedNormalHealSet IndexedNormalSet3D;
  
  // Use the following definition to get ride of Healpix
//  typedef  IndexedNormalSet
//                  < Point,   //! \brief Point type used internally
//                    3,       //! \brief Nb dimension
//                    7,       //! \brief Nb cells/dim normal
//                    Scalar>  //! \brief Scalar type
//  IndexedNormalSet3D;


  IndexedNormalSet3D nset (eps, 2.);

  for (unsigned int i = 0; i < P_pairs.size(); ++i) {
    const Point& p1 = pcfunctor_.points[P_pairs[i].first];
    const Point& p2 = pcfunctor_.points[P_pairs[i].second];
    Point  n  = (p2 - p1);

//    cout << "new entry: " << endl
//         << p1.transpose() << "(" << P_pairs[i].first  << ")" << endl
//         << p2.transpose() << "(" << P_pairs[i].second << ")" << endl
//         << (p1+ invariant1       * (p2 - p1)).transpose() << endl
//         << n.transpose() << endl;

    nset.addElement( p1 + invariant1 * n,  n.normalized(), i);
  }


  std::set< std::pair<unsigned int, unsigned int > > comb;

  unsigned int j = 0;
  std::vector<unsigned int> nei;
  // 2. Query time
  for (unsigned int i = 0; i < Q_pairs.size(); ++i) {
    const Point& p1 = pcfunctor_.points[Q_pairs[i].first];
    const Point& p2 = pcfunctor_.points[Q_pairs[i].second];

    const Point3D& pq1 = Q_points[Q_pairs[i].first];
    const Point3D& pq2 = Q_points[Q_pairs[i].second];
    Point n = p2 - p1;

    nei.clear();

//    cout << "query: " << endl
//         << p1.transpose() << "(" << Q_pairs[i].first  << ")" << endl
//         << p2.transpose() << "(" << Q_pairs[i].second << ")" << endl
//         << query.transpose() << endl
//         << queryn.transpose() << endl;

    nset.getNeighbors( p1 + invariant2 * n,  n.normalized(), alpha, nei);

    if (! nei.empty()){
      Point3D queryQ = pq1 + invariant2 * (pq2 - pq1);

      Point3D invPoint;
      //const Scalar distance_threshold2s = distance_threshold2 * distance_threshold2;
      for (unsigned int k = 0; k != nei.size(); k++){
        int id = nei[k];

        const Point3D& pp1 = Q_points[P_pairs[id].first];
        const Point3D& pp2 = Q_points[P_pairs[id].second];

        invPoint = pp1 + (pp2 - pp1) * invariant1;

         // use also distance_threshold2 for inv 1 and 2 in 4PCS
        if (cv::norm(queryQ-invPoint) <= distance_threshold2){
            comb.insert(std::pair<unsigned int, unsigned int>(id, i));
        }
      }
    }
  }

  for (std::set< std::pair<unsigned int, unsigned int > >::const_iterator it =
             comb.cbegin();
       it != comb.cend(); it++){
    const unsigned int & id = (*it).first;
    const unsigned int & i  = (*it).second;

    quadrilaterals->push_back(
                Quadrilateral(P_pairs[id].first, P_pairs[id].second,
                              Q_pairs[i].first,  Q_pairs[i].second));
  }

  return quadrilaterals->size() != 0;
}


// Computes the mean distance between points in Q and their nearest neighbor.
// We need this for normalization of the user delta (See the paper) to the
// "scale" of the set.
MatchSuper4PCSImpl::Scalar MatchSuper4PCSImpl::MeanDistance() {
  const Scalar kDiameterFraction = 0.2;
  Super4PCS::KdTree<Scalar>::VectorType query_point;

  int number_of_samples = 0;
  Scalar distance = 0.0;

  for (int i = 0; i < sampled_P_3D_.size(); ++i) {
    query_point << sampled_P_3D_[i].x, sampled_P_3D_[i].y, sampled_P_3D_[i].z;

    Super4PCS::KdTree<Scalar>::Index resId =
    kd_tree_.doQueryRestrictedClosest(query_point, P_diameter_ * kDiameterFraction, i);

    if (resId != Super4PCS::KdTree<Scalar>::invalidIndex()) {
      distance += cv::norm(sampled_P_3D_[i] - sampled_P_3D_[resId]);
      number_of_samples++;
    }
  }

  return distance / number_of_samples;
}

// Verify a given transformation by computing the number of points in P at
// distance at most (normalized) delta from some point in Q. In the paper
// we describe randomized verification. We apply deterministic one here with
// early termination. It was found to be fast in practice.
MatchSuper4PCSImpl::Scalar
MatchSuper4PCSImpl::Verify(const Eigen::Matrix<Scalar, 4, 4>& mat) {

#ifdef TEST_GLOBAL_TIMINGS
    Timer t_verify (true);
#endif

  // We allow factor 2 scaling in the normalization.
  Scalar epsilon = options_.delta;
  int good_points = 0;
  int number_of_points = pcfunctor_.points.size();
  int terminate_value = best_LCP_ * number_of_points;

  Scalar sq_eps = epsilon*epsilon;

  for (int i = 0; i < number_of_points; ++i) {

    // Use the kdtree to get the nearest neighbor
#ifdef TEST_GLOBAL_TIMINGS
    Timer t (true);
#endif
    Super4PCS::KdTree<Scalar>::Index resId =
    kd_tree_.doQueryRestrictedClosest(
                (mat * pcfunctor_.getPointInWorldCoord( i ).homogeneous()).head<3>(),
                sq_eps);


#ifdef TEST_GLOBAL_TIMINGS
    kdTreeTime += Scalar(t.elapsed().count()) / Scalar(CLOCKS_PER_SEC);
#endif

    if ( resId != Super4PCS::KdTree<Scalar>::invalidIndex() ) {
//      Point3D& q = sampled_P_3D_[near_neighbor_index[0]];
//      bool rgb_good =
//          (p.rgb()[0] >= 0 && q.rgb()[0] >= 0)
//              ? cv::norm(p.rgb() - q.rgb()) < options_.max_color_distance
//              : true;
//      bool norm_good = norm(p.normal()) > 0 && norm(q.normal()) > 0
//                           ? fabs(p.normal().ddot(q.normal())) >= cos_dist
//                           : true;
//      if (rgb_good && norm_good) {
        good_points++;
//      }
    }

    // We can terminate if there is no longer chance to get better than the
    // current best LCP.
    if (number_of_points - i + good_points < terminate_value) {
      break;
    }
  }

#ifdef TEST_GLOBAL_TIMINGS
  verifyTime += Scalar(t_verify.elapsed().count()) / Scalar(CLOCKS_PER_SEC);
#endif
  return static_cast<Scalar>(good_points) / number_of_points;
}

// Constructs two sets of pairs in Q, each corresponds to one pair in the base
// in P, by having the same distance (up to some tolerantz) and optionally the
// same angle between normals and same color.
void
MatchSuper4PCSImpl::ExtractPairs(Scalar pair_distance,
                                    Scalar pair_normals_angle,
                                    Scalar pair_distance_epsilon,
                                    int base_point1, int base_point2,
                                    PairsVector* pairs) {

  using namespace Super4PCS::Accelerators::PairExtraction;

  pcfunctor_.pairs = pairs;

  pairs->clear();
  pairs->reserve(2 * pcfunctor_.points.size());

  pcfunctor_.pair_distance         = pair_distance;
  pcfunctor_.pair_distance_epsilon = pair_distance_epsilon;
  pcfunctor_.pair_normals_angle    = pair_normals_angle;
  pcfunctor_.norm_threshold =
      0.5 * options_.max_normal_difference * M_PI / 180.0;

  pcfunctor_.setRadius(pair_distance);
  pcfunctor_.setBase(base_point1, base_point2, base_3D_);


#ifdef MULTISCALE
  BruteForceFunctor
  <PairCreationFunctor<Scalar>::Point, 3, Scalar> interFunctor;
#else
  IntersectionFunctor
          <PairCreationFunctor<Scalar>::Primitive,
          PairCreationFunctor<Scalar>::Point, 3, Scalar> interFunctor;
#endif

  Scalar eps = pcfunctor_.getNormalizedEpsilon(pair_distance_epsilon);

  interFunctor.process(pcfunctor_.primitives,
                       pcfunctor_.points,
                       eps,
                       50,
                       pcfunctor_);
}

// Pick one base, finds congruent 4-points in Q, verifies for all
// transformations, and retains the best transformation and LCP. This is
// a complete RANSAC iteration.
bool MatchSuper4PCSImpl::TryOneBase() {
  vector<pair<Point3D, Point3D>> congruent_points(4);
  Scalar invariant1, invariant2;
  int base_id1, base_id2, base_id3, base_id4;
  Scalar distance_factor = 2.0;

//#define STATIC_BASE

#ifdef STATIC_BASE
  static bool first_time = true;

  if (first_time){
      base_id1 = 0;
      base_id2 = 3;
      base_id3 = 1;
      base_id4 = 4;

      base_3D_[0] = sampled_P_3D_ [base_id1];
      base_3D_[1] = sampled_P_3D_ [base_id2];
      base_3D_[2] = sampled_P_3D_ [base_id3];
      base_3D_[3] = sampled_P_3D_ [base_id4];

      Utils::TryQuadrilateral(invariant1, invariant2, base_id1, base_id2, base_id3, base_id4, base_3D_);

      first_time = false;
  }
  else
      return false;

#else

  if (!Utils::SelectQuadrilateral(
              invariant1, invariant2,
              base_id1, base_id2, base_id3, base_id4,
              base_3D_, sampled_P_3D_, max_base_diameter_,
              kNumberOfDiameterTrials)) {
    return false;
  }
#endif

  // Computes distance between pairs.
  Scalar distance1 = cv::norm(base_3D_[0] - base_3D_[1]);
  Scalar distance2 = cv::norm(base_3D_[2] - base_3D_[3]);

  vector<pair<int, int>> pairs1, pairs2;
  vector<Quadrilateral> congruent_quads;

  // Compute normal angles.
  double normal_angle1 = cv::norm(base_3D_[0].normal() - base_3D_[1].normal());
  double normal_angle2 = cv::norm(base_3D_[2].normal() - base_3D_[3].normal());

  ExtractPairs(distance1, normal_angle1, distance_factor * options_.delta, 0,
                  1, &pairs1);
  ExtractPairs(distance2, normal_angle2, distance_factor * options_.delta, 2,
                  3, &pairs2);

  if (pairs1.size() == 0 || pairs2.size() == 0) {
    return false;
  }


  if (!FindCongruentQuadrilateralsFast(invariant1, invariant2,
                                   distance_factor /** factor*/ * options_.delta,
                                   distance_factor /** factor*/ * options_.delta,
                                   pairs1,
                                   pairs2,
                                   sampled_Q_3D_,
                                   &congruent_quads)) {
    return false;
  }

  // get references to the basis coordinates
  const Point3D& b1 = sampled_P_3D_[base_id1];
  const Point3D& b2 = sampled_P_3D_[base_id2];
  const Point3D& b3 = sampled_P_3D_[base_id3];
  const Point3D& b4 = sampled_P_3D_[base_id4];


  // Centroid of the basis, computed once and using only the three first points
  Eigen::Matrix<Scalar, 3, 1> centroid1;
  // Must be improved when running without opencv
  centroid1 << (b1.x + b2.x + b3.x) / Scalar(3.),
               (b1.y + b2.y + b3.y) / Scalar(3.),
               (b1.z + b2.z + b3.z) / Scalar(3.);

  // Centroid of the sets, computed in the loop using only the three first points
  Eigen::Matrix<Scalar, 3, 1> centroid2;

  // set the basis coordinates in the congruent quad array
  congruent_points.resize(4);
  congruent_points[0].first = b1;
  congruent_points[1].first = b2;
  congruent_points[2].first = b3;
  congruent_points[3].first = b4;

  Eigen::Matrix<Scalar, 4, 4> transform;
  for (int i = 0; i < congruent_quads.size(); ++i) {
    int a = congruent_quads[i].vertices[0];
    int b = congruent_quads[i].vertices[1];
    int c = congruent_quads[i].vertices[2];
    int d = congruent_quads[i].vertices[3];
    congruent_points[0].second = sampled_Q_3D_[a];
    congruent_points[1].second = sampled_Q_3D_[b];
    congruent_points[2].second = sampled_Q_3D_[c];
    congruent_points[3].second = sampled_Q_3D_[d];

#ifdef STATIC_BASE
    std::cout << "Ids:" << std::endl;
    std::cout << base_id1 << "\t"
              << base_id2 << "\t"
              << base_id3 << "\t"
              << base_id4 << std::endl;
    std::cout << a << "\t"
              << b << "\t"
              << c << "\t"
              << d << std::endl;
#endif

    centroid2 << (congruent_points[0].second.x + congruent_points[1].second.x + congruent_points[2].second.x) / Scalar(3.),
                 (congruent_points[0].second.y + congruent_points[1].second.y + congruent_points[2].second.y) / Scalar(3.),
                 (congruent_points[0].second.z + congruent_points[1].second.z + congruent_points[2].second.z) / Scalar(3.);

    Scalar rms = -1;

    bool ok =
    ComputeRigidTransformation(congruent_points,   // input congruent quads
                               centroid1,          // input: basis centroid
                               centroid2,          // input: candidate quad centroid
                               options_.max_angle * M_PI / 180.0, // maximum per-dimension angle, check return value to detect invalid cases
                               transform,          // output: transformation
                               rms,                // output: rms error of the transformation between the basis and the congruent quad
                           #ifdef MULTISCALE
                               true
                           #else
                               false
                           #endif
                               );             // state: compute scale ratio ?

    if (ok && rms >= Scalar(0.)) {

      // We give more tolerantz in computing the best rigid transformation.
      if (rms < distance_factor * options_.delta) {
        // The transformation is computed from the point-clouds centered inn [0,0,0]

        // Verify the rest of the points in Q against P.
        Scalar lcp = Verify(transform);
        if (lcp > best_LCP_) {
          // Retain the best LCP and transformation.
          base_[0] = base_id1;
          base_[1] = base_id2;
          base_[2] = base_id3;
          base_[3] = base_id4;

          current_congruent_[0] = a;
          current_congruent_[1] = b;
          current_congruent_[2] = c;
          current_congruent_[3] = d;

          best_LCP_    = lcp;
          transform_   = transform;
          qcentroid1_  = centroid1;
          qcentroid2_  = centroid2;
        }
        // Terminate if we have the desired LCP already.
        if (best_LCP_ > options_.terminate_threshold){
          return true;
        }
      }
    }
  }
  
  // If we reached here we do not have yet the desired LCP.
  return false;
}

struct eqstr {
  bool operator()(const char* s1, const char* s2) const {
    return strcmp(s1, s2) == 0;
  }
};

// Initialize all internal data structures and data members.
void MatchSuper4PCSImpl::Initialize(const std::vector<Point3D>& P,
                               const std::vector<Point3D>& Q) {

  const Scalar kSmallError = 0.00001;
  const int kMinNumberOfTrials = 4;
  const Scalar kDiameterFraction = 0.3;

  centroid_P_.x = 0;
  centroid_P_.y = 0;
  centroid_P_.z = 0;
  centroid_Q_.x = 0;
  centroid_Q_.y = 0;
  centroid_Q_.z = 0;

  sampled_P_3D_.clear();
  sampled_Q_3D_.clear();

#ifdef TEST_GLOBAL_TIMINGS
    kdTreeTime = 0;
    totalTime  = 0;
    verifyTime = 0;
#endif


  // prepare P
  if (P.size() > options_.sample_size){
      std::vector<Point3D> uniform_P;
      Sampling::DistUniformSampling(P, options_.delta, &uniform_P);

      int sample_fraction_P = 1;  // We prefer not to sample P but any number can be
                                  // placed here.

      // Sample the sets P and Q uniformly.
      for (int i = 0; i < uniform_P.size(); ++i) {
        if (rand() % sample_fraction_P == 0) {
          sampled_P_3D_.push_back(uniform_P[i]);
        }
      }
  }
  else
  {
      cout << "(P) More samples requested than available: use whole cloud" << endl;
      sampled_P_3D_ = P;
  }



  // prepare Q
  if (Q.size() > options_.sample_size){
      std::vector<Point3D> uniform_Q;
      Sampling::DistUniformSampling(Q, options_.delta, &uniform_Q);
      int sample_fraction_Q =
          max(1, static_cast<int>(uniform_Q.size() / options_.sample_size));

      for (int i = 0; i < uniform_Q.size(); ++i) {
        if (rand() % sample_fraction_Q == 0) {
          sampled_Q_3D_.push_back(uniform_Q[i]);
        }
      }
  }
  else
  {
      cout << "(Q) More samples requested than available: use whole cloud" << endl;
      sampled_Q_3D_ = Q;
  }

  // Compute the centroids.
  for (int i = 0; i < sampled_P_3D_.size(); ++i) {
    centroid_P_.x += sampled_P_3D_[i].x;
    centroid_P_.y += sampled_P_3D_[i].y;
    centroid_P_.z += sampled_P_3D_[i].z;
  }

  centroid_P_.x /= sampled_P_3D_.size();
  centroid_P_.y /= sampled_P_3D_.size();
  centroid_P_.z /= sampled_P_3D_.size();

  for (int i = 0; i < sampled_Q_3D_.size(); ++i) {
    centroid_Q_.x += sampled_Q_3D_[i].x;
    centroid_Q_.y += sampled_Q_3D_[i].y;
    centroid_Q_.z += sampled_Q_3D_[i].z;
  }

  centroid_Q_.x /= sampled_Q_3D_.size();
  centroid_Q_.y /= sampled_Q_3D_.size();
  centroid_Q_.z /= sampled_Q_3D_.size();

  // Move the samples to the centroids to allow robustness in rotation.
  for (int i = 0; i < sampled_P_3D_.size(); ++i) {
    sampled_P_3D_[i].x -= centroid_P_.x;
    sampled_P_3D_[i].y -= centroid_P_.y;
    sampled_P_3D_[i].z -= centroid_P_.z;
  }
  for (int i = 0; i < sampled_Q_3D_.size(); ++i) {
    sampled_Q_3D_[i].x -= centroid_Q_.x;
    sampled_Q_3D_[i].y -= centroid_Q_.y;
    sampled_Q_3D_[i].z -= centroid_Q_.z;
  }

  int number_of_points = sampled_P_3D_.size();

  // Build the kdtree.
  kd_tree_ = Super4PCS::KdTree<Scalar>(number_of_points);

  Super4PCS::KdTree<Scalar>::VectorType p;
  for (int i = 0; i < number_of_points; ++i) {
      p << sampled_P_3D_[i].x,
           sampled_P_3D_[i].y,
           sampled_P_3D_[i].z;

      kd_tree_.add(p);
  }
  kd_tree_.finalize();

  pcfunctor_.synch3DContent();

  // Compute the diameter of P approximately (randomly). This is far from being
  // Guaranteed close to the diameter but gives good results for most common
  // objects if they are densely sampled.
  P_diameter_ = 0.0;
  int nbPoints = pcfunctor_.points.size();
  for (int i = 0; i < kNumberOfDiameterTrials; ++i) {
    int at = rand() % nbPoints;
    int bt = rand() % nbPoints;
    Scalar l = (pcfunctor_.points[bt] - pcfunctor_.points[at]).norm();
    if (l > P_diameter_) {
      P_diameter_ = l;
    }
  }
  //switch from unit space to original space
  P_diameter_ = pcfunctor_.unitToWorld(P_diameter_);

  // Mean distance and a bit more... We increase the estimation to allow for
  // noise, wrong estimation and non-uniform sampling.
  // can be used to automatically select delta
  //P_mean_distance_ = MeanDistance();

  // Normalize the delta (See the paper) and the maximum base distance.
  // delta = P_mean_distance_ * delta;
  max_base_diameter_ = P_diameter_;  // * estimated_overlap_;

  // RANSAC probability and number of needed trials.
  Scalar first_estimation =
      log(kSmallError) / log(1.0 - pow(options_.overlap_estimation,
                                       static_cast<Scalar>(kMinNumberOfTrials)));
  // We use a simple heuristic to elevate the probability to a reasonable value
  // given that we don't simply sample from P, but instead, we bound the
  // distance between the points in the base as a fraction of the diameter.
  number_of_trials_ =
      static_cast<int>(first_estimation * (P_diameter_ / kDiameterFraction) /
                       max_base_diameter_);
  if (options_.terminate_threshold < 0)
    options_.terminate_threshold = options_.overlap_estimation;
  if (number_of_trials_ < kMinNumberOfTrials)
    number_of_trials_ = kMinNumberOfTrials;

  printf("norm_max_dist: %f\n", options_.delta);
  current_trial_ = 0;
  transform_ = Eigen::Matrix<Scalar, 4, 4>::Identity();
  best_LCP_ = Verify(transform_);
  printf("Initial LCP: %f\n", best_LCP_);

  Q_copy_ = Q;
  for (int i = 0; i < 4; ++i) {
    base_[i] = 0;
    current_congruent_[i] = 0;
  }
}

bool MatchSuper4PCSImpl::ComputeRigidTransformation(const vector< pair<Point3D, Point3D> >& pairs,
                                                    const Eigen::Matrix<Scalar, 3, 1>& centroid1,
                                                    Eigen::Matrix<Scalar, 3, 1> centroid2,
                                                    Scalar max_angle,
                                                    Eigen::Matrix<Scalar, 4, 4> &transform,
                                                    Scalar& rms_,
                                                    bool computeScale ) {

  rms_ = std::numeric_limits<Scalar>::max();

  if (pairs.size() == 0 || pairs.size() % 2 != 0)
      return false;


  Scalar kSmallNumber = 1e-6;
  cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);

  // We only use the first 3 pairs. This simplifies the process considerably
  // because it is the planar case.

  const cv::Point3f& p0 = pairs[0].first;
  const cv::Point3f& p1 = pairs[1].first;
  const cv::Point3f& p2 = pairs[2].first;
        cv::Point3f  q0 = pairs[0].second;
        cv::Point3f  q1 = pairs[1].second;
        cv::Point3f  q2 = pairs[2].second;

  Scalar scaleEst (1.);

  // Compute scale factor if needed
  if (computeScale){
      const cv::Point3f& p3 = pairs[3].first;
      const cv::Point3f& q3 = pairs[3].second;

      Scalar ratio1 = cv::norm(p1 - p0) / cv::norm(q1 - q0);
      Scalar ratio2 = cv::norm(p3 - p2) / cv::norm(q3 - q2);

      Scalar ratioDev  = std::abs(ratio1/ratio2 - Scalar(1.));  // deviation between the two
      Scalar ratioMean = (ratio1+ratio2)/Scalar(2.);            // mean of the two

      if ( ratioDev > Scalar(0.1) )
          return false;

//      std::cout << ratio1 << " "
//                << ratio2 << " "
//                << ratioDev << " "
//                << ratioMean << std::endl;

      scaleEst = ratioMean;

      // apply scale factor to q
      q0 = q0*scaleEst;
      q1 = q1*scaleEst;
      q2 = q2*scaleEst;
      centroid2 *= scaleEst;
  }

  cv::Point3f vector_p1 = p1 - p0;
  if (cv::norm(vector_p1) == 0) return false;
  vector_p1 = vector_p1 * (1.0 / cv::norm(vector_p1));
  cv::Point3f vector_p2 = (p2 - p0) - ((p2 - p0).dot(vector_p1)) * vector_p1;
  if (cv::norm(vector_p2) == 0) return false;
  vector_p2 = vector_p2 * (1.0 / cv::norm(vector_p2));
  cv::Point3f vector_p3 = vector_p1.cross(vector_p2);

  cv::Point3f vector_q1 = q1 - q0;
  if (cv::norm(vector_q1) == 0) return false;
  vector_q1 = vector_q1 * (1.0 / cv::norm(vector_q1));
  cv::Point3f vector_q2 = (q2 - q0) - ((q2 - q0).dot(vector_q1)) * vector_q1;
  if (cv::norm(vector_q2) == 0) return false;
  vector_q2 = vector_q2 * (1.0 / cv::norm(vector_q2));
  cv::Point3f vector_q3 = vector_q1.cross(vector_q2);

  cv::Mat rotate_p(3, 3, CV_64F);
  rotate_p.at<Scalar>(0, 0) = vector_p1.x;
  rotate_p.at<Scalar>(0, 1) = vector_p1.y;
  rotate_p.at<Scalar>(0, 2) = vector_p1.z;
  rotate_p.at<Scalar>(1, 0) = vector_p2.x;
  rotate_p.at<Scalar>(1, 1) = vector_p2.y;
  rotate_p.at<Scalar>(1, 2) = vector_p2.z;
  rotate_p.at<Scalar>(2, 0) = vector_p3.x;
  rotate_p.at<Scalar>(2, 1) = vector_p3.y;
  rotate_p.at<Scalar>(2, 2) = vector_p3.z;

  cv::Mat rotate_q(3, 3, CV_64F);
  rotate_q.at<Scalar>(0, 0) = vector_q1.x;
  rotate_q.at<Scalar>(0, 1) = vector_q1.y;
  rotate_q.at<Scalar>(0, 2) = vector_q1.z;
  rotate_q.at<Scalar>(1, 0) = vector_q2.x;
  rotate_q.at<Scalar>(1, 1) = vector_q2.y;
  rotate_q.at<Scalar>(1, 2) = vector_q2.z;
  rotate_q.at<Scalar>(2, 0) = vector_q3.x;
  rotate_q.at<Scalar>(2, 1) = vector_q3.y;
  rotate_q.at<Scalar>(2, 2) = vector_q3.z;

  rotation = rotate_p.t() * rotate_q;

  // Discard singular solutions. The rotation should be orthogonal.
  cv::Mat unit = rotation * rotation.t();
  if (std::abs(unit.at<Scalar>(0, 0) - 1.0) > kSmallNumber ||
      std::abs(unit.at<Scalar>(1, 1) - 1.0) > kSmallNumber ||
      std::abs(unit.at<Scalar>(2, 2) - 1.0) > kSmallNumber){
      return false;
  }

  // Discard too large solutions (todo: lazy evaluation during boolean computation
  Scalar theta_x = std::abs(atan2(rotation.at<Scalar>(2, 1), rotation.at<Scalar>(2, 2)));
  Scalar theta_y = std::abs(atan2(-rotation.at<Scalar>(2, 0),
                             sqrt(Utils::square(rotation.at<Scalar>(2, 1)) +
                                  Utils::square(rotation.at<Scalar>(2, 2)))));
  Scalar theta_z = std::abs(atan2(rotation.at<Scalar>(1, 0), rotation.at<Scalar>(0, 0)));
  if (theta_x > max_angle ||
      theta_y > max_angle ||
      theta_z > max_angle)
      return false;


  // Compute rms and return it.
  rms_ = Scalar(0.0);
  {
      cv::Mat first(3, 1, CV_64F), transformed;
      for (int i = 0; i < 3; ++i) {
          first.at<Scalar>(0, 0) = scaleEst*pairs[i].second.x - centroid2(0);
          first.at<Scalar>(1, 0) = scaleEst*pairs[i].second.y - centroid2(1);
          first.at<Scalar>(2, 0) = scaleEst*pairs[i].second.z - centroid2(2);
          transformed = rotation * first;
          rms_ += sqrt(Utils::square(transformed.at<Scalar>(0, 0) -
                              (pairs[i].first.x - centroid1(0))) +
                       Utils::square(transformed.at<Scalar>(1, 0) -
                              (pairs[i].first.y - centroid1(1))) +
                       Utils::square(transformed.at<Scalar>(2, 0) -
                              (pairs[i].first.z - centroid1(2))));
      }
  }

  rms_ /= Scalar(pairs.size());

  Eigen::Transform<Scalar, 3, Eigen::Affine> etrans (Eigen::Transform<Scalar, 3, Eigen::Affine>::Identity());

  // compute rotation and translation
  {
      Eigen::Matrix<Scalar, 3, 3> rot;
      cv::cv2eigen(rotation, rot);

      //std::cout << scaleEst << endl;

      etrans.scale(scaleEst);       // apply scale factor
      etrans.translate(centroid1);  // translation between quads
      etrans.rotate(rot);           // rotate to align frames
      etrans.translate(-centroid2); // move to congruent quad frame

      transform = etrans.matrix();
  }

  return true;
}


// Performs N RANSAC iterations and compute the best transformation. Also,
// transforms the set Q by this optimal transformation.
bool MatchSuper4PCSImpl::Perform_N_steps(int n, cv::Mat* transformation,
                                    std::vector<Point3D>* Q) {
  if (transformation == NULL || Q == NULL) return false;

#ifdef TEST_GLOBAL_TIMINGS
    Timer t (true);
#endif

  Scalar last_best_LCP = best_LCP_;
  bool ok;
  int64 t0 = clock();
  for (int i = current_trial_; i < current_trial_ + n; ++i) {
    ok = TryOneBase();

    Scalar fraction_try =
        static_cast<Scalar>(i) / static_cast<Scalar>(number_of_trials_);
    Scalar fraction_time = static_cast<Scalar>(clock() - t0) / CLOCKS_PER_SEC /
                          options_.max_time_seconds;
    Scalar fraction = max(fraction_time, fraction_try);
    printf("done: %d%c best: %f                  \r",
           static_cast<int>(fraction * 100), '%', best_LCP_);
    fflush(stdout);
    // ok means that we already have the desired LCP.
    if (ok || i > number_of_trials_ || fraction >= 0.99 || best_LCP_ == 1.0) break;
  }

  current_trial_ += n;
  if (best_LCP_ > last_best_LCP) {
    *Q = Q_copy_;

      // The transformation has been computed between the two point clouds centered
    // at the origin, we need to recompute the translation to apply it to the original clouds
    {
        Eigen::Matrix<Scalar, 3,1> centroid_P,centroid_Q;
        cv::Mat first(3, 1, CV_64F);
        first.at<Scalar>(0, 0) = centroid_P_.x;
        first.at<Scalar>(1, 0) = centroid_P_.y;
        first.at<Scalar>(2, 0) = centroid_P_.z;
        cv::cv2eigen(first, centroid_P);
        first.at<Scalar>(0, 0) = centroid_Q_.x;
        first.at<Scalar>(1, 0) = centroid_Q_.y;
        first.at<Scalar>(2, 0) = centroid_Q_.z;
        cv::cv2eigen(first, centroid_Q);

        Eigen::Matrix<Scalar, 3, 3> rot, scale;
        Eigen::Transform<Scalar, 3, Eigen::Affine> (transform_).computeRotationScaling(&rot, &scale);
        transform_.col(3) = (qcentroid1_ + centroid_P - ( rot * scale * (qcentroid2_ + centroid_Q))).homogeneous();
    }

    cv::eigen2cv( transform_, *transformation );

    // Transforms Q by the new transformation.
    for (unsigned int i = 0; i < Q->size(); ++i) {
      cv::Mat first(4, 1, CV_64F), transformed;
      first.at<Scalar>(0, 0) = (*Q)[i].x;
      first.at<Scalar>(1, 0) = (*Q)[i].y;
      first.at<Scalar>(2, 0) = (*Q)[i].z;
      first.at<Scalar>(3, 0) = 1;
      transformed = *transformation * first;
      (*Q)[i].x = transformed.at<Scalar>(0, 0);
      (*Q)[i].y = transformed.at<Scalar>(1, 0);
      (*Q)[i].z = transformed.at<Scalar>(2, 0);
    }
  }
#ifdef TEST_GLOBAL_TIMINGS
    totalTime += Scalar(t.elapsed().count()) / Scalar(CLOCKS_PER_SEC);
#endif

  return ok || current_trial_ >= number_of_trials_;
}

// The main 4PCS function. Computes the best rigid transformation and transfoms
// Q toward P by this transformation.
MatchSuper4PCSImpl::Scalar MatchSuper4PCSImpl::ComputeTransformation(const std::vector<Point3D>& P,
                                           std::vector<Point3D>* Q,
                                           cv::Mat* transformation) {

  if (Q == nullptr || transformation == nullptr)
      return std::numeric_limits<Scalar>::max();
  Initialize(P, *Q);

  *transformation = cv::Mat(4, 4, CV_64F, cv::Scalar(0.0));
  for (int i = 0; i < 4; ++i) transformation->at<Scalar>(i, i) = 1.0;
  Perform_N_steps(number_of_trials_, transformation, Q);

#ifdef TEST_GLOBAL_TIMINGS
  cout << "----------- Timings (msec) -------------"           << endl;
  cout << " Total computation time  : " << totalTime           << endl;
  cout << " Total verify time       : " << verifyTime          << endl;
  cout << "    Kdtree query         : " << kdTreeTime          << endl;
#endif

  return best_LCP_;
}

MatchSuper4PCS::MatchSuper4PCS(const Match4PCSOptions& options)
    : pimpl_{new MatchSuper4PCSImpl{options}} {}

MatchSuper4PCS::~MatchSuper4PCS() {}

double
MatchSuper4PCS::ComputeTransformation(const std::vector<Point3D>& P,
                                       std::vector<Point3D>* Q,
                                       cv::Mat* transformation) {
  return pimpl_->ComputeTransformation(P, Q, transformation);
}
}

