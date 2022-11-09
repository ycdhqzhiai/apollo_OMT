// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: sensor_image.proto

#include "sensor_image.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
namespace apollo {
namespace drivers {
class ImageDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<Image> _instance;
} _Image_default_instance_;
}  // namespace drivers
}  // namespace apollo
static void InitDefaultsscc_info_Image_sensor_5fimage_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::apollo::drivers::_Image_default_instance_;
    new (ptr) ::apollo::drivers::Image();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_Image_sensor_5fimage_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_Image_sensor_5fimage_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_sensor_5fimage_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_sensor_5fimage_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_sensor_5fimage_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_sensor_5fimage_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::apollo::drivers::Image, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::apollo::drivers::Image, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::apollo::drivers::Image, frame_id_),
  PROTOBUF_FIELD_OFFSET(::apollo::drivers::Image, measurement_time_),
  PROTOBUF_FIELD_OFFSET(::apollo::drivers::Image, height_),
  PROTOBUF_FIELD_OFFSET(::apollo::drivers::Image, width_),
  PROTOBUF_FIELD_OFFSET(::apollo::drivers::Image, encoding_),
  PROTOBUF_FIELD_OFFSET(::apollo::drivers::Image, step_),
  PROTOBUF_FIELD_OFFSET(::apollo::drivers::Image, data_),
  0,
  3,
  4,
  5,
  1,
  6,
  2,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 12, sizeof(::apollo::drivers::Image)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::apollo::drivers::_Image_default_instance_),
};

const char descriptor_table_protodef_sensor_5fimage_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\022sensor_image.proto\022\016apollo.drivers\"\200\001\n"
  "\005Image\022\020\n\010frame_id\030\001 \001(\t\022\030\n\020measurement_"
  "time\030\002 \001(\001\022\016\n\006height\030\003 \001(\r\022\r\n\005width\030\004 \001("
  "\r\022\020\n\010encoding\030\005 \001(\t\022\014\n\004step\030\006 \001(\r\022\014\n\004dat"
  "a\030\007 \001(\014*\373\005\n\013PixelFormat\022\t\n\004RGB8\020\351\007\022\n\n\005RG"
  "BA8\020\352\007\022\n\n\005RGB16\020\353\007\022\013\n\006RGBA16\020\354\007\022\t\n\004BGR8\020"
  "\355\007\022\n\n\005BGRA8\020\356\007\022\n\n\005BGR16\020\357\007\022\013\n\006BGRA16\020\360\007\022"
  "\n\n\005MONO8\020\361\007\022\013\n\006MONO16\020\362\007\022\016\n\tTYPE_8UC1\020\321\017"
  "\022\016\n\tTYPE_8UC2\020\322\017\022\016\n\tTYPE_8UC3\020\323\017\022\016\n\tTYPE"
  "_8UC4\020\324\017\022\016\n\tTYPE_8SC1\020\325\017\022\016\n\tTYPE_8SC2\020\326\017"
  "\022\016\n\tTYPE_8SC3\020\327\017\022\016\n\tTYPE_8SC4\020\330\017\022\017\n\nTYPE"
  "_16UC1\020\331\017\022\017\n\nTYPE_16UC2\020\332\017\022\017\n\nTYPE_16UC3"
  "\020\333\017\022\017\n\nTYPE_16UC4\020\334\017\022\017\n\nTYPE_16SC1\020\335\017\022\017\n"
  "\nTYPE_16SC2\020\336\017\022\017\n\nTYPE_16SC3\020\337\017\022\017\n\nTYPE_"
  "16SC4\020\340\017\022\017\n\nTYPE_32SC1\020\341\017\022\017\n\nTYPE_32SC2\020"
  "\342\017\022\017\n\nTYPE_32SC3\020\343\017\022\017\n\nTYPE_32SC4\020\344\017\022\017\n\n"
  "TYPE_32FC1\020\345\017\022\017\n\nTYPE_32FC2\020\346\017\022\017\n\nTYPE_3"
  "2FC3\020\347\017\022\017\n\nTYPE_32FC4\020\350\017\022\017\n\nTYPE_64FC1\020\351"
  "\017\022\017\n\nTYPE_64FC2\020\352\017\022\017\n\nTYPE_64FC3\020\353\017\022\017\n\nT"
  "YPE_64FC4\020\354\017\022\020\n\013BAYER_RGGB8\020\271\027\022\020\n\013BAYER_"
  "BGGR8\020\272\027\022\020\n\013BAYER_GBRG8\020\273\027\022\020\n\013BAYER_GRBG"
  "8\020\274\027\022\021\n\014BAYER_RGGB16\020\275\027\022\021\n\014BAYER_BGGR16\020"
  "\276\027\022\021\n\014BAYER_GBRG16\020\277\027\022\021\n\014BAYER_GRBG16\020\300\027"
  "\022\013\n\006YUV422\020\241\037"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_sensor_5fimage_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_sensor_5fimage_2eproto_sccs[1] = {
  &scc_info_Image_sensor_5fimage_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_sensor_5fimage_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_sensor_5fimage_2eproto = {
  false, false, descriptor_table_protodef_sensor_5fimage_2eproto, "sensor_image.proto", 933,
  &descriptor_table_sensor_5fimage_2eproto_once, descriptor_table_sensor_5fimage_2eproto_sccs, descriptor_table_sensor_5fimage_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_sensor_5fimage_2eproto::offsets,
  file_level_metadata_sensor_5fimage_2eproto, 1, file_level_enum_descriptors_sensor_5fimage_2eproto, file_level_service_descriptors_sensor_5fimage_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_sensor_5fimage_2eproto = (static_cast<void>(::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_sensor_5fimage_2eproto)), true);
namespace apollo {
namespace drivers {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* PixelFormat_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_sensor_5fimage_2eproto);
  return file_level_enum_descriptors_sensor_5fimage_2eproto[0];
}
bool PixelFormat_IsValid(int value) {
  switch (value) {
    case 1001:
    case 1002:
    case 1003:
    case 1004:
    case 1005:
    case 1006:
    case 1007:
    case 1008:
    case 1009:
    case 1010:
    case 2001:
    case 2002:
    case 2003:
    case 2004:
    case 2005:
    case 2006:
    case 2007:
    case 2008:
    case 2009:
    case 2010:
    case 2011:
    case 2012:
    case 2013:
    case 2014:
    case 2015:
    case 2016:
    case 2017:
    case 2018:
    case 2019:
    case 2020:
    case 2021:
    case 2022:
    case 2023:
    case 2024:
    case 2025:
    case 2026:
    case 2027:
    case 2028:
    case 3001:
    case 3002:
    case 3003:
    case 3004:
    case 3005:
    case 3006:
    case 3007:
    case 3008:
    case 4001:
      return true;
    default:
      return false;
  }
}


// ===================================================================

class Image::_Internal {
 public:
  using HasBits = decltype(std::declval<Image>()._has_bits_);
  static void set_has_frame_id(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_measurement_time(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_height(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static void set_has_width(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
  static void set_has_encoding(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_step(HasBits* has_bits) {
    (*has_bits)[0] |= 64u;
  }
  static void set_has_data(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
};

Image::Image(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:apollo.drivers.Image)
}
Image::Image(const Image& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  frame_id_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from._internal_has_frame_id()) {
    frame_id_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_frame_id(), 
      GetArena());
  }
  encoding_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from._internal_has_encoding()) {
    encoding_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_encoding(), 
      GetArena());
  }
  data_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from._internal_has_data()) {
    data_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_data(), 
      GetArena());
  }
  ::memcpy(&measurement_time_, &from.measurement_time_,
    static_cast<size_t>(reinterpret_cast<char*>(&step_) -
    reinterpret_cast<char*>(&measurement_time_)) + sizeof(step_));
  // @@protoc_insertion_point(copy_constructor:apollo.drivers.Image)
}

void Image::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_Image_sensor_5fimage_2eproto.base);
  frame_id_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  encoding_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  data_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  ::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
      reinterpret_cast<char*>(&measurement_time_) - reinterpret_cast<char*>(this)),
      0, static_cast<size_t>(reinterpret_cast<char*>(&step_) -
      reinterpret_cast<char*>(&measurement_time_)) + sizeof(step_));
}

Image::~Image() {
  // @@protoc_insertion_point(destructor:apollo.drivers.Image)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void Image::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  frame_id_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  encoding_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  data_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void Image::ArenaDtor(void* object) {
  Image* _this = reinterpret_cast< Image* >(object);
  (void)_this;
}
void Image::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void Image::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const Image& Image::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_Image_sensor_5fimage_2eproto.base);
  return *internal_default_instance();
}


void Image::Clear() {
// @@protoc_insertion_point(message_clear_start:apollo.drivers.Image)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    if (cached_has_bits & 0x00000001u) {
      frame_id_.ClearNonDefaultToEmpty();
    }
    if (cached_has_bits & 0x00000002u) {
      encoding_.ClearNonDefaultToEmpty();
    }
    if (cached_has_bits & 0x00000004u) {
      data_.ClearNonDefaultToEmpty();
    }
  }
  if (cached_has_bits & 0x00000078u) {
    ::memset(&measurement_time_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&step_) -
        reinterpret_cast<char*>(&measurement_time_)) + sizeof(step_));
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Image::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // optional string frame_id = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_frame_id();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "apollo.drivers.Image.frame_id");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional double measurement_time = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 17)) {
          _Internal::set_has_measurement_time(&has_bits);
          measurement_time_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // optional uint32 height = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_height(&has_bits);
          height_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional uint32 width = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32)) {
          _Internal::set_has_width(&has_bits);
          width_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional string encoding = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 42)) {
          auto str = _internal_mutable_encoding();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "apollo.drivers.Image.encoding");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional uint32 step = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 48)) {
          _Internal::set_has_step(&has_bits);
          step_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bytes data = 7;
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 58)) {
          auto str = _internal_mutable_data();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* Image::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:apollo.drivers.Image)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional string frame_id = 1;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_frame_id().data(), static_cast<int>(this->_internal_frame_id().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "apollo.drivers.Image.frame_id");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_frame_id(), target);
  }

  // optional double measurement_time = 2;
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(2, this->_internal_measurement_time(), target);
  }

  // optional uint32 height = 3;
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt32ToArray(3, this->_internal_height(), target);
  }

  // optional uint32 width = 4;
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt32ToArray(4, this->_internal_width(), target);
  }

  // optional string encoding = 5;
  if (cached_has_bits & 0x00000002u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_encoding().data(), static_cast<int>(this->_internal_encoding().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "apollo.drivers.Image.encoding");
    target = stream->WriteStringMaybeAliased(
        5, this->_internal_encoding(), target);
  }

  // optional uint32 step = 6;
  if (cached_has_bits & 0x00000040u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt32ToArray(6, this->_internal_step(), target);
  }

  // optional bytes data = 7;
  if (cached_has_bits & 0x00000004u) {
    target = stream->WriteBytesMaybeAliased(
        7, this->_internal_data(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:apollo.drivers.Image)
  return target;
}

size_t Image::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:apollo.drivers.Image)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000007fu) {
    // optional string frame_id = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_frame_id());
    }

    // optional string encoding = 5;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_encoding());
    }

    // optional bytes data = 7;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::BytesSize(
          this->_internal_data());
    }

    // optional double measurement_time = 2;
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 + 8;
    }

    // optional uint32 height = 3;
    if (cached_has_bits & 0x00000010u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt32Size(
          this->_internal_height());
    }

    // optional uint32 width = 4;
    if (cached_has_bits & 0x00000020u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt32Size(
          this->_internal_width());
    }

    // optional uint32 step = 6;
    if (cached_has_bits & 0x00000040u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt32Size(
          this->_internal_step());
    }

  }
  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Image::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:apollo.drivers.Image)
  GOOGLE_DCHECK_NE(&from, this);
  const Image* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<Image>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:apollo.drivers.Image)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:apollo.drivers.Image)
    MergeFrom(*source);
  }
}

void Image::MergeFrom(const Image& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:apollo.drivers.Image)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x0000007fu) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_frame_id(from._internal_frame_id());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_set_encoding(from._internal_encoding());
    }
    if (cached_has_bits & 0x00000004u) {
      _internal_set_data(from._internal_data());
    }
    if (cached_has_bits & 0x00000008u) {
      measurement_time_ = from.measurement_time_;
    }
    if (cached_has_bits & 0x00000010u) {
      height_ = from.height_;
    }
    if (cached_has_bits & 0x00000020u) {
      width_ = from.width_;
    }
    if (cached_has_bits & 0x00000040u) {
      step_ = from.step_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void Image::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:apollo.drivers.Image)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Image::CopyFrom(const Image& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:apollo.drivers.Image)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Image::IsInitialized() const {
  return true;
}

void Image::InternalSwap(Image* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  frame_id_.Swap(&other->frame_id_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  encoding_.Swap(&other->encoding_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  data_.Swap(&other->data_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(Image, step_)
      + sizeof(Image::step_)
      - PROTOBUF_FIELD_OFFSET(Image, measurement_time_)>(
          reinterpret_cast<char*>(&measurement_time_),
          reinterpret_cast<char*>(&other->measurement_time_));
}

::PROTOBUF_NAMESPACE_ID::Metadata Image::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace drivers
}  // namespace apollo
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::apollo::drivers::Image* Arena::CreateMaybeMessage< ::apollo::drivers::Image >(Arena* arena) {
  return Arena::CreateMessageInternal< ::apollo::drivers::Image >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
