
#include "yaml-cpp/yaml.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "cyber/common/file.h"
#include "cyber/common/log.h"
#include "gflags/gflags.h"
#include "gtest/gtest.h"

#include "base/object.h"
#include "base/object_types.h"
#include "camera/common/object_template_manager.h"
#include "camera/common/camera_frame.h"
#include "camera/lib/interface/base_obstacle_tracker.h"
#include "inference/utils/cuda_util.h"
#include "common/common.h"
#include "camera/lib/obstacle/tracker/omt/omt_obstacle_tracker.h"
#include "common/util/eigen_defs.h"
DEFINE_string(data_root,
              "/home/yc-mc/workspace/MC_CV/self_driving_env/perception/"
              "production/data",
              "root dir of images");
DEFINE_int32(max_img_num, 10, "max length of test images");
DEFINE_string(narrow_name, "narrow", "type of camera for projecting");
DEFINE_string(obstacle_name, "obstacle", "type of camera to be peojected");
DEFINE_string(image_root,
              "../testdata/camera/lib/obstacle/tracker/omt/test_fusion_data/images",
              "root dir of images");
DEFINE_int32(feature_length, 576, "size of feature");
DEFINE_string(base_camera_name, "onsemi_obstacle", "camera to be peojected");
DEFINE_string(sensor_name, "onsemi_obstacle,onsemi_narrow", "camera to use");
DEFINE_string(test_list, "test.txt", "exe image list");
DEFINE_double(camera_fps, 15, "camera_fps");
DEFINE_string(image_ext, ".jpg", "extension of image name");
DEFINE_string(image_color, "bgr", "color space of image");

namespace apollo {
namespace perception {
namespace camera {
int read_detections(const std::string &path, const int &feature_dim,
                    const std::string &camera_name, apollo::perception::camera::CameraFrame *frame) {
  const TemplateMap &kMinTemplateHWL =
      ObjectTemplateManager::Instance()->MinTemplateHWL();

  std::ifstream fin(path);
  if (!fin.is_open()) {
    AERROR << "Cannot open : " << path;
    return -1;
  }
  int frame_num = -1;
  int det_count = 0;

  int feature_size = feature_dim;
  fin >> frame_num >> det_count;
  frame->frame_id = frame_num;
  frame->track_feature_blob.reset(new base::Blob<float>);
  (frame->track_feature_blob)->Reshape({det_count, feature_size});
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float feature = 0.0f;
  float score = 0.0f;
  frame->detected_objects.clear();
  for (int i = 0; i < det_count; ++i) {
    fin >> x >> y >> width >> height >> score;
    base::BBox2DF bbox(x, y, x + width, y + height);
    base::ObjectPtr object(new base::Object);
    object->camera_supplement.box = bbox;
    object->camera_supplement.sensor_name = camera_name;
    object->sub_type = base::ObjectSubType::CAR;
    object->type = base::ObjectType::VEHICLE;
    object->size(0) = kMinTemplateHWL.at(base::ObjectSubType::CAR).at(2);
    object->size(1) = kMinTemplateHWL.at(base::ObjectSubType::CAR).at(1);
    object->size(2) = kMinTemplateHWL.at(base::ObjectSubType::CAR).at(0);
    float *data =
        frame->track_feature_blob->mutable_cpu_data() + i * feature_size;
    for (int j = 0; j < feature_size; j++) {
      fin >> feature;
      *data = feature;
      ++data;
    }
    frame->detected_objects.push_back(object);
  }
  return 0;
}


// @description: load camera extrinsics from yaml file
bool LoadExtrinsics(const std::string &yaml_file,
                    Eigen::Matrix4d *camera_extrinsic) {
  if (!apollo::cyber::common::PathExists(yaml_file)) {
    AINFO << yaml_file << " not exist!";
    return false;
  }
  YAML::Node node = YAML::LoadFile(yaml_file);
  double qw = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;
  try {
    if (node.IsNull()) {
      AINFO << "Load " << yaml_file << " failed! please check!";
      return false;
    }
    qw = node["transform"]["rotation"]["w"].as<double>();
    qx = node["transform"]["rotation"]["x"].as<double>();
    qy = node["transform"]["rotation"]["y"].as<double>();
    qz = node["transform"]["rotation"]["z"].as<double>();
    tx = node["transform"]["translation"]["x"].as<double>();
    ty = node["transform"]["translation"]["y"].as<double>();
    tz = node["transform"]["translation"]["z"].as<double>();
  } catch (YAML::InvalidNode &in) {
    AERROR << "load camera extrisic file " << yaml_file
           << " with error, YAML::InvalidNode exception";
    return false;
  } catch (YAML::TypedBadConversion<double> &bc) {
    AERROR << "load camera extrisic file " << yaml_file
           << " with error, YAML::TypedBadConversion exception";
    return false;
  } catch (YAML::Exception &e) {
    AERROR << "load camera extrisic file " << yaml_file
           << " with error, YAML exception:" << e.what();
    return false;
  }
  camera_extrinsic->setConstant(0);
  Eigen::Quaterniond q;
  q.x() = qx;
  q.y() = qy;
  q.z() = qz;
  q.w() = qw;
  (*camera_extrinsic).block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
  (*camera_extrinsic)(0, 3) = tx;
  (*camera_extrinsic)(1, 3) = ty;
  (*camera_extrinsic)(2, 3) = tz;
  (*camera_extrinsic)(3, 3) = 1;
  return true;
}

// @description: modified from common::LoadBrownCameraIntrinsic
// the original funciton return by BrownCameraDistortionModel class, but
// intrinsic_parm_ is a protected filed, cannot get directly
bool LoadCameraIntrinsics(const std::string &yaml_file,
                          Eigen::Matrix3f *camera_intrinsic) {
  if (!apollo::cyber::common::PathExists(yaml_file)) {
    AINFO << yaml_file << " not exist!";
    return false;
  }
  YAML::Node node = YAML::LoadFile(yaml_file);
  if (node.IsNull()) {
    AINFO << "Load " << yaml_file << " failed! please check!";
    return false;
  }
  try {
    for (int i = 0; i < static_cast<int>(node["K"].size()); ++i) {
      (*camera_intrinsic)(i / 3, i % 3) = node["K"][i].as<float>();
    }
  } catch (YAML::Exception &e) {
    AERROR << "load camera extrisic file " << yaml_file
           << " with error, YAML exception: " << e.what();
    return false;
  }
  return true;
}

int work() {
  // Init object template
  ObjectTemplateManagerInitOptions object_template_init_options;
  object_template_init_options.root_dir =
      "/home/yc-mc/workspace/MC_CV/self_driving_env/perception/"
      "production/data/perception/camera/common/object_template";
  object_template_init_options.conf_file = "object_template.pt";
  ACHECK(ObjectTemplateManager::Instance()->Init(object_template_init_options));

  // Init camera list
  std::string camera_name = "front_6mm";
  // Init data provider
  DataProvider::InitOptions data_options;
  data_options.image_height = 1080;
  data_options.image_width = 1920;
  data_options.do_undistortion = false;
  data_options.device_id = 0;

  DataProvider data_provider;
  ACHECK(data_provider.Init(data_options));

  // Init extrinsic/intrinsic
  Eigen::Matrix4d extrinsic;
  Eigen::Matrix3f intrinsic;

  ACHECK(LoadCameraIntrinsics(
      FLAGS_data_root + "/params/" + camera_name + "_intrinsics.yaml",
      &intrinsic));
    
  ACHECK(LoadExtrinsics(
      FLAGS_data_root + "/params/" + camera_name + "_extrinsics.yaml",
      &extrinsic));
  inference::CudaUtil::set_device_id(0);
  // init tracker
  ObstacleTrackerInitOptions init_options;
  init_options.root_dir =
      "/home/yc-mc/workspace/MC_CV/self_driving_env/perception/production/"
      "data/perception/camera/models/omt_obstacle_tracker";
  init_options.conf_file = "config.pt";
  init_options.image_height = 1080;
  init_options.image_width = 1920;
  std::unique_ptr<BaseObstacleTracker> camera_tracker(
      BaseObstacleTrackerRegisterer::GetInstanceByName("OMTObstacleTracker"));
  ACHECK(camera_tracker->Init(init_options));
  EXPECT_EQ("OMTObstacleTracker", camera_tracker->Name());

  // read gt
  std::string filename = "../testdata/camera/lib/obstacle/tracker/omt/test_fusion_data/det_gt.txt";
  std::ifstream fin_gt(filename);
  ACHECK(fin_gt.is_open());
  std::vector<std::vector<base::CameraObjectSupplement>> det_gts;
  std::string image_name;
  int det_count = 0;
  while (fin_gt >> image_name >> det_count) {
    std::vector<base::CameraObjectSupplement> bboxs;
    base::CameraObjectSupplement bbox;
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    int id = 0;
    for (int i = 0; i < det_count; ++i) {
      fin_gt >> x >> y >> width >> height >> id;
      base::BBox2DF box(x, y, x + width, y + height);
      bbox.box = box;
      bbox.local_track_id = id;
      bboxs.push_back(bbox);
    }
    det_gts.push_back(bboxs);
  }
  AINFO << det_gts.size();
  std::vector<std::vector<int>> tracked_index(det_gts.size());
  std::vector<std::vector<base::CameraObjectSupplement>> tracked_results(
      det_gts.size());
  EigenVector<CameraFrame> frames(det_gts.size());
  std::string line;
  int frame_num = 0;
  AINFO << det_gts.size();
  // Init input list
  std::ifstream fin;
  filename = "../testdata/camera/lib/obstacle/tracker/omt/test_fusion_data/test.txt";
  fin.open(filename, std::ifstream::in);
  ACHECK(fin.is_open());
  while (fin >> line) {
    const std::vector<std::string> temp_strs = absl::StrSplit(line, '/');
    if (temp_strs.size() != 2) {
      AERROR << "invaid format in " << FLAGS_test_list;
    }
    image_name = temp_strs[1];

    AINFO << "image: " << image_name << " camera_name:" << camera_name;
    std::string image_path = FLAGS_image_root + "/obstacle" + "/" +
                             image_name + FLAGS_image_ext;
    AINFO << image_path;
    CameraFrame &frame = frames[frame_num];

    // read detections from txt
    const std::string filename =
        absl::StrCat("../testdata/camera/lib/obstacle/tracker/omt/test_fusion_data/detection_feature/", frame_num, ".txt");
    read_detections(filename, FLAGS_feature_length, camera_name, &frame);
    AINFO << "Frame " << frame_num << " has " << frame.detected_objects.size()
          << " detection objects";
    frame.frame_id = frame_num;
    std::stringstream ss(image_name);
    frame.timestamp = 0.0;
    ss >> frame.timestamp;
    frame.timestamp *= 1e-9;
    AINFO << "Timestamp: " << frame.timestamp;
    Eigen::Matrix3d project_matrix = Eigen::Matrix3d::Identity();
    double pitch_diff = 0.0;
    frame.project_matrix = project_matrix;
    frame.data_provider = &data_provider;
    AINFO << "Project Matrix: \n" << frame.project_matrix;
    ACHECK(camera_tracker->Predict(ObstacleTrackerOptions(), &frame));
    ACHECK(camera_tracker->Associate2D(ObstacleTrackerOptions(), &frame));
    ACHECK(camera_tracker->Associate3D(ObstacleTrackerOptions(), &frame));
    ACHECK(camera_tracker->Track(ObstacleTrackerOptions(), &frame));
    AINFO << "Frame " << frame_num
          << " tracked object size: " << frame.tracked_objects.size();
    for (auto &bbox_gt : det_gts[frame_num]) {
      int id = -1;
      float max_iou = 0.0f;
      Eigen::Matrix<double, 3, 1> center0, size0;
      center0[0] = bbox_gt.box.Center().x;
      size0[0] = bbox_gt.box.xmax - bbox_gt.box.xmin;
      center0[1] = bbox_gt.box.Center().y;
      size0[1] = bbox_gt.box.ymax - bbox_gt.box.ymin;
      base::CameraObjectSupplement bbox;
      for (int i = 0; i < frames[frame_num].tracked_objects.size(); i++) {
        Eigen::Matrix<double, 3, 1> center1, size1;
        base::BBox2DF temp_box;
        temp_box = frames[frame_num].tracked_objects[i]->camera_supplement.box;
        center1[0] = temp_box.Center().x;
        size1[0] = temp_box.xmax - temp_box.xmin;
        center1[1] = temp_box.Center().y;
        size1[1] = temp_box.ymax - temp_box.ymin;
        float iou = common::CalculateIou2DXY(center0, size0, center1, size1);
        AINFO << "IOU is :" << iou;
        if (iou > max_iou) {
          max_iou = iou;
          id = i;
          bbox.local_track_id = frames[frame_num].tracked_objects[i]->track_id;
        }
      }
      if (frame_num >= dynamic_cast<OMTObstacleTracker *>(camera_tracker.get())
                           ->omt_param_.target_param()
                           .tracked_life()) {
        EXPECT_GE(max_iou, 0.5);
      }
      tracked_index[frame_num].push_back(id);
      tracked_results[frame_num].push_back(bbox);
    }
    ++frame_num;
  }
  std::vector<int> ids(2, -100);
  for (frame_num = dynamic_cast<OMTObstacleTracker *>(camera_tracker.get())
                       ->omt_param_.target_param()
                       .tracked_life();
       frame_num < det_gts.size(); ++frame_num) {
    EXPECT_GE(tracked_results[frame_num].size(), 1);
    for (int i = 0; i < det_gts[frame_num].size(); ++i) {
      if (ids[i] < -50) {
        ids[i] = det_gts[frame_num][i].local_track_id -
                 tracked_results[frame_num][i].local_track_id;
      } else {
        EXPECT_EQ(ids[i], det_gts[frame_num][i].local_track_id -
                              tracked_results[frame_num][i].local_track_id);
      }
    }
  }
  return 0;
}

}  // namespace camera
}  // namespace perception
}  // namespace apollo
int main(int argc, char* argv[]) {

  google::ParseCommandLineFlags(&argc, &argv, true);
  google::SetUsageMessage(
      "command line brew\n"
      "Usage: camera_benchmark <args>\n");

  return apollo::perception::camera::work();
}
