#include "base/frame.h"
#include <iomanip>
namespace apollo {
namespace perception {
namespace base {

void LogFrame(std::ofstream &file, FrameConstPtr frame, size_t count) {
  file << "frame " << count << " timestamp: " << std::fixed
       << std::setprecision(3) << frame->timestamp
       << " objects size = " << frame->objects.size() << " "
       << frame->sensor_info.name << " "
       << static_cast<int>(frame->sensor_info.type) << "\n";
  for (base::ObjectPtr a_obj : frame->objects) {
    file << a_obj->ToString();
    file << " xmin= " << a_obj->camera_supplement.box.xmin
         << " ymin= " << a_obj->camera_supplement.box.ymin
         << " xmax= " << a_obj->camera_supplement.box.xmax
         << " ymax= " << a_obj->camera_supplement.box.ymax << "\n";
  }
}

} // namespace base
} // namespace perception
} // namespace apollo