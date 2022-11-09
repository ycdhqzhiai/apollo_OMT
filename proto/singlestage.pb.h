// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: singlestage.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_singlestage_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_singlestage_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3014000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3014000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_singlestage_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_singlestage_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_singlestage_2eproto;
namespace apollo {
namespace perception {
namespace camera {
namespace singlestage {
class SinglestageParam;
class SinglestageParamDefaultTypeInternal;
extern SinglestageParamDefaultTypeInternal _SinglestageParam_default_instance_;
}  // namespace singlestage
}  // namespace camera
}  // namespace perception
}  // namespace apollo
PROTOBUF_NAMESPACE_OPEN
template<> ::apollo::perception::camera::singlestage::SinglestageParam* Arena::CreateMaybeMessage<::apollo::perception::camera::singlestage::SinglestageParam>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace apollo {
namespace perception {
namespace camera {
namespace singlestage {

// ===================================================================

class SinglestageParam PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:apollo.perception.camera.singlestage.SinglestageParam) */ {
 public:
  inline SinglestageParam() : SinglestageParam(nullptr) {}
  virtual ~SinglestageParam();

  SinglestageParam(const SinglestageParam& from);
  SinglestageParam(SinglestageParam&& from) noexcept
    : SinglestageParam() {
    *this = ::std::move(from);
  }

  inline SinglestageParam& operator=(const SinglestageParam& from) {
    CopyFrom(from);
    return *this;
  }
  inline SinglestageParam& operator=(SinglestageParam&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const SinglestageParam& default_instance();

  static inline const SinglestageParam* internal_default_instance() {
    return reinterpret_cast<const SinglestageParam*>(
               &_SinglestageParam_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(SinglestageParam& a, SinglestageParam& b) {
    a.Swap(&b);
  }
  inline void Swap(SinglestageParam* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(SinglestageParam* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline SinglestageParam* New() const final {
    return CreateMaybeMessage<SinglestageParam>(nullptr);
  }

  SinglestageParam* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<SinglestageParam>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const SinglestageParam& from);
  void MergeFrom(const SinglestageParam& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(SinglestageParam* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "apollo.perception.camera.singlestage.SinglestageParam";
  }
  protected:
  explicit SinglestageParam(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_singlestage_2eproto);
    return ::descriptor_table_singlestage_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kMinDimensionValFieldNumber = 1,
    kCheckDimensionFieldNumber = 2,
  };
  // optional float min_dimension_val = 1 [default = 0.2];
  bool has_min_dimension_val() const;
  private:
  bool _internal_has_min_dimension_val() const;
  public:
  void clear_min_dimension_val();
  float min_dimension_val() const;
  void set_min_dimension_val(float value);
  private:
  float _internal_min_dimension_val() const;
  void _internal_set_min_dimension_val(float value);
  public:

  // optional bool check_dimension = 2 [default = true];
  bool has_check_dimension() const;
  private:
  bool _internal_has_check_dimension() const;
  public:
  void clear_check_dimension();
  bool check_dimension() const;
  void set_check_dimension(bool value);
  private:
  bool _internal_check_dimension() const;
  void _internal_set_check_dimension(bool value);
  public:

  // @@protoc_insertion_point(class_scope:apollo.perception.camera.singlestage.SinglestageParam)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  float min_dimension_val_;
  bool check_dimension_;
  friend struct ::TableStruct_singlestage_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// SinglestageParam

// optional float min_dimension_val = 1 [default = 0.2];
inline bool SinglestageParam::_internal_has_min_dimension_val() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool SinglestageParam::has_min_dimension_val() const {
  return _internal_has_min_dimension_val();
}
inline void SinglestageParam::clear_min_dimension_val() {
  min_dimension_val_ = 0.2f;
  _has_bits_[0] &= ~0x00000001u;
}
inline float SinglestageParam::_internal_min_dimension_val() const {
  return min_dimension_val_;
}
inline float SinglestageParam::min_dimension_val() const {
  // @@protoc_insertion_point(field_get:apollo.perception.camera.singlestage.SinglestageParam.min_dimension_val)
  return _internal_min_dimension_val();
}
inline void SinglestageParam::_internal_set_min_dimension_val(float value) {
  _has_bits_[0] |= 0x00000001u;
  min_dimension_val_ = value;
}
inline void SinglestageParam::set_min_dimension_val(float value) {
  _internal_set_min_dimension_val(value);
  // @@protoc_insertion_point(field_set:apollo.perception.camera.singlestage.SinglestageParam.min_dimension_val)
}

// optional bool check_dimension = 2 [default = true];
inline bool SinglestageParam::_internal_has_check_dimension() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool SinglestageParam::has_check_dimension() const {
  return _internal_has_check_dimension();
}
inline void SinglestageParam::clear_check_dimension() {
  check_dimension_ = true;
  _has_bits_[0] &= ~0x00000002u;
}
inline bool SinglestageParam::_internal_check_dimension() const {
  return check_dimension_;
}
inline bool SinglestageParam::check_dimension() const {
  // @@protoc_insertion_point(field_get:apollo.perception.camera.singlestage.SinglestageParam.check_dimension)
  return _internal_check_dimension();
}
inline void SinglestageParam::_internal_set_check_dimension(bool value) {
  _has_bits_[0] |= 0x00000002u;
  check_dimension_ = value;
}
inline void SinglestageParam::set_check_dimension(bool value) {
  _internal_set_check_dimension(value);
  // @@protoc_insertion_point(field_set:apollo.perception.camera.singlestage.SinglestageParam.check_dimension)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace singlestage
}  // namespace camera
}  // namespace perception
}  // namespace apollo

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_singlestage_2eproto
