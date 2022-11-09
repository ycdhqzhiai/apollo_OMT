#include "dbscan.h"
namespace apollo {
namespace perception {
namespace base {

using namespace std;
int Dbscan::run() {
  int clusterID = 1;
  vector<DPoint>::iterator iter;
  for (iter = m_points_.begin(); iter != m_points_.end(); ++iter) {
    if (iter->clusterID == UNCLASSIFIED) {
      if (expandCluster(*iter, clusterID) != FAILURE) {
        clusterID += 1;
      }
    }
  }

  return 0;
}

int Dbscan::expandCluster(DPoint point, int clusterID) {
  vector<int> clusterSeeds = calculateCluster(point);

  if (clusterSeeds.size() < m_minPoints_) {
    point.clusterID = NOISE;
    return FAILURE;
  } else {
    int index = 0, indexCorePoint = 0;
    vector<int>::iterator iterSeeds;
    for (iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end();
         ++iterSeeds) {
      m_points_.at(*iterSeeds).clusterID = clusterID;
      if (m_points_.at(*iterSeeds).x == point.x &&
          m_points_.at(*iterSeeds).y == point.y &&
          m_points_.at(*iterSeeds).z == point.z) {
        indexCorePoint = index;
      }
      ++index;
    }
    clusterSeeds.erase(clusterSeeds.begin() + indexCorePoint);

    for (vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i) {
      vector<int> clusterNeighors =
          calculateCluster(m_points_.at(clusterSeeds[i]));

      if (clusterNeighors.size() >= m_minPoints_) {
        vector<int>::iterator iterNeighors;
        for (iterNeighors = clusterNeighors.begin();
             iterNeighors != clusterNeighors.end(); ++iterNeighors) {
          if (m_points_.at(*iterNeighors).clusterID == UNCLASSIFIED ||
              m_points_.at(*iterNeighors).clusterID == NOISE) {
            if (m_points_.at(*iterNeighors).clusterID == UNCLASSIFIED) {
              clusterSeeds.push_back(*iterNeighors);
              n = clusterSeeds.size();
            }
            m_points_.at(*iterNeighors).clusterID = clusterID;
          }
        }
      }
    }

    return SUCCESS;
  }
}

vector<int> Dbscan::calculateCluster(DPoint point) {
  int index = 0;
  vector<DPoint>::iterator iter;
  vector<int> clusterIndex;
  for (iter = m_points_.begin(); iter != m_points_.end(); ++iter) {
    if (calculateDistance(point, *iter) <= m_epsilon_) {
      clusterIndex.push_back(index);
    }
    index++;
  }
  return clusterIndex;
}

inline double Dbscan::calculateDistance(const DPoint& pointCore,
                                        const DPoint& pointTarget) {
  return sqrt(pow(pointCore.x - pointTarget.x, 2) +
         pow(pointCore.y - pointTarget.y, 2)); 
}

}  // namespace base
}  // namespace perception
}  // namespace apollo
