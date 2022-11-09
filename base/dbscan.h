#ifndef DBSCAN_H
#define DBSCAN_H
#include <cmath>
#include <vector>

namespace apollo {
namespace perception {
namespace base {

#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3

struct DPoint {
  DPoint(float x_, float y_, float z_, float vx_, float vy_)
      : x(x_), y(y_), z(z_), vx(vx_), vy(vy_) {}
  DPoint() {}
  float x, y, z, vx, vy;  // X, Y, Z position, vx, vy
  int clusterID = -1;     // clustered ID
};

class Dbscan {
 public:
  Dbscan(unsigned int minPts, float eps, std::vector<DPoint> points) {
    m_minPoints_ = minPts;
    m_epsilon_ = eps;
    m_points_ = points;
    m_pointSize_ = points.size();
  }
  ~Dbscan() {}

  int run();
  std::vector<int> calculateCluster(DPoint point);
  int expandCluster(DPoint point, int clusterID);
  inline double calculateDistance(const DPoint& pointCore,
                                  const DPoint& pointTarget);

  int getTotalPointSize() { return m_pointSize_; }
  int getMinimumClusterSize() { return m_minPoints_; }
  int getEpsilonSize() { return m_epsilon_; }

 public:
  std::vector<DPoint> m_points_;

 private:
  unsigned int m_pointSize_;
  unsigned int m_minPoints_;
  float m_epsilon_;
};

}  // namespace base
}  // namespace perception
}  // namespace apollo
#endif  // DBSCAN_H
