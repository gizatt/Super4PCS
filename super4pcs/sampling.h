#ifndef SAMPLING_H
#define SAMPLING_H

#include <vector>
#include <array>

namespace FastRegistration {
namespace Sampling {

namespace internal {

/*!
 * \brief Class used to subsample a point cloud using hashing functions
 */
class HashTable {
 private:
  using triplet =  std::array<int,3>;
  const uint64 MAGIC1 = 100000007;
  const uint64 MAGIC2 = 161803409;
  const uint64 MAGIC3 = 423606823;
  const uint64 NO_DATA = 0xffffffffu;
  float voxel_;
  float scale_;
  std::vector<triplet> voxels_;
  std::vector<uint64> data_;

 public:
  HashTable(int maxpoints, float voxel) : voxel_(voxel), scale_(1.0f / voxel) {
    uint64 n = maxpoints;
    voxels_.resize(n);
    data_.resize(n, NO_DATA);
  }

  template <typename Point3D>
  uint64& operator[](const Point3D& p) {
    std::array<int,3> c;
    c[0] = static_cast<int>(floor(p.x * scale_));
    c[1] = static_cast<int>(floor(p.y * scale_));
    c[2] = static_cast<int>(floor(p.z * scale_));
    uint64 key = (MAGIC1 * c[0] + MAGIC2 * c[1] + MAGIC3 * c[2]) % data_.size();
    while (1) {
      if (data_[key] == NO_DATA) {
        voxels_[key] = c;
        break;
      } else if (voxels_[key] == c) {
        break;
      }
      key++;
      if (key == data_.size()) key = 0;
    }
    return data_[key];
  }
};

} // namespace internal


template <typename Point3D>
static inline void DistUniformSampling(
        const std::vector<Point3D>& set,
        float delta,
        std::vector<Point3D>* sample) {
  int num_input = set.size();
  sample->clear();
  internal::HashTable hash(num_input, delta);
  for (int i = 0; i < num_input; i++) {
    uint64& ind = hash[set[i]];
    if (ind >= num_input) {
      sample->push_back(set[i]);
      ind = sample->size();
    }
  }
}


} // namespace sampling
} // namespace fastregistration

#endif // SAMPLING_H
