/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include "base/object_pool_types.h"
#include "camera/common/data_provider.h"
#include <Eigen/StdVector>

namespace apollo {
namespace perception {
namespace camera {

class BaseCalibrationService;

struct CameraFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // timestamp
  double timestamp = 0.0;
  // frame sequence id
  int frame_id = 0;
  // data provider
  DataProvider *data_provider = nullptr;
  // calibration service
  // hdmap struct
  // tracker proposed objects
  std::vector<base::ObjectPtr> proposed_objects;
  // segmented objects
  std::vector<base::ObjectPtr> detected_objects;
  // tracked objects
  std::vector<base::ObjectPtr> tracked_objects;
  // feature of all detected object ( num x dim)
  // detect lane mark info
  std::vector<float> pred_vpt;
  std::shared_ptr<base::Blob<float>> track_feature_blob = nullptr;
  // detected traffic lights
  // camera intrinsics
  Eigen::Matrix3f camera_k_matrix = Eigen::Matrix3f::Identity();
  // narrow to obstacle projected_matrix
  Eigen::Matrix3d project_matrix = Eigen::Matrix3d::Identity();
  // camera to world pose
  Eigen::Affine3d camera2world_pose = Eigen::Affine3d::Identity();
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;  // struct CameraFrame

}  // namespace camera
}  // namespace perception
}  // namespace apollo
