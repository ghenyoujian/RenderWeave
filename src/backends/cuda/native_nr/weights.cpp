#include "native_nr.hpp"
#include "rw/rw_api.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace rw::native_nr {
namespace {
using Json = nlohmann::json;
uint64_t little_integer(const uint8_t* bytes, unsigned count) {
    uint64_t value = 0;
    for (unsigned i = 0; i < count; ++i) value |= uint64_t(bytes[i]) << (8 * i);
    return value;
}
size_t nonnegative(const Json& value, const char* what) {
    if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < 0))
        throw std::invalid_argument(std::string("invalid safetensors ") + what);
    const auto result = value.get<uint64_t>();
    if (result > std::numeric_limits<size_t>::max()) throw std::overflow_error("safetensors integer overflows host size");
    return static_cast<size_t>(result);
}
float f16_decode(uint16_t bits) {
    const unsigned exponent = (bits >> 10) & 31u, fraction = bits & 1023u;
    if (exponent == 31u) {
        const uint32_t word = (uint32_t(bits & 0x8000u) << 16) | 0x7f800000u | (fraction << 13);
        return std::bit_cast<float>(word);
    }
    const float absolute = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24)
        : std::ldexp(static_cast<float>(1024 + fraction), static_cast<int>(exponent) - 25);
    return std::copysign(absolute, (bits & 0x8000u) ? -1.0f : 1.0f);
}
std::string parameter_key(int block, int layer, const std::string& name) {
    if (block < 0 || layer < 0 || name.empty()) throw std::invalid_argument("invalid native NR parameter key");
    return "block" + std::to_string(block) + ".layer" + std::to_string(layer) + "." + name;
}
std::string cache_key(Context& context, const std::string& name, const std::string& type,
                      const std::vector<int64_t>& shape) {
    std::string result = std::to_string(reinterpret_cast<uintptr_t>(context.identity())) + ":" + type + ":" + name;
    for (const auto extent : shape) result += ":" + std::to_string(extent);
    return result;
}
}

struct Weights::Impl {
    struct Record { size_t offset = 0, bytes = 0; };
    struct Cached {
        Parameter tensor;
        std::vector<int64_t> shape;
        std::vector<float> derived_values;
    };
    struct PreparedEntry {
        const void* context = nullptr;
        std::string key;
        std::vector<int64_t> configuration;
        std::shared_ptr<const PreparedParameterSet> parameters;
        std::vector<std::vector<int64_t>> shapes;
        std::vector<Parameter> storage_owners;
        uint64_t payload_bytes = 0, container_bytes = 0, allocator_allowance_bytes = 0;
    };
    size_t file_bytes = 0, payload_offset = 0;
    const void* mapping = nullptr;
#ifdef _WIN32
    HANDLE fd = INVALID_HANDLE_VALUE;
    HANDLE file_mapping = nullptr;
#else
    int fd = -1;
#endif
    std::string model_network = "ht";
    std::unordered_map<std::string, Record> records;
    std::unordered_map<std::string, Cached> cache;
    // Flat bounded registry: its actual pointer-array capacity is observable,
    // unlike implementation-dependent unordered_map bucket/node overhead.
    std::vector<std::unique_ptr<PreparedEntry>> prepared;
    PreparationStatistics preparation;
    bool shared_mode=false;
    std::weak_ptr<DeviceWeightPool> shared_pool;
    RegisteredModelOwner registered_model;
    std::mutex mutex;

    ~Impl() {
        prepared.clear();
        cache.clear();
#ifdef _WIN32
        if (mapping) UnmapViewOfFile(mapping);
        if (file_mapping) CloseHandle(file_mapping);
        if (fd != INVALID_HANDLE_VALUE) CloseHandle(fd);
#else
        if (mapping) munmap(const_cast<void*>(mapping), file_bytes);
        if (fd >= 0) close(fd);
#endif
    }

    static void validate_prepared_key(const std::string& key, std::span<const int64_t> configuration) {
        if (key.empty() || key.size() > 256 || configuration.empty() || configuration.size() > 32)
            throw std::invalid_argument("prepared parameter key/configuration exceeds its bound");
        // CPU-derived bias values use the pinned source's binary32 FMA rule.
        // A cache must not hide a change to the host rounding environment.
        if (std::fegetround() != FE_TONEAREST)
            throw std::invalid_argument("native NR parameter preparation requires FE_TONEAREST");
    }
    PreparedEntry* lookup_prepared(const void* context, const std::string& key) const {
        for (const auto& entry : prepared)
            if (entry->context == context && entry->key == key) return entry.get();
        return nullptr;
    }
    static bool same_configuration(const PreparedEntry& entry, std::span<const int64_t> configuration) {
        return entry.configuration.size() == configuration.size() &&
            std::equal(entry.configuration.begin(), entry.configuration.end(), configuration.begin());
    }
    static void validate_prepared_entry(Context& context, const PreparedEntry& entry) {
        const auto& tensors = entry.parameters->tensors;
        if (entry.context != context.identity() || tensors.size() != entry.shapes.size() ||
            tensors.size() != entry.storage_owners.size())
            throw std::invalid_argument("prepared parameter identity/count changed");
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto& tensor = tensors[i];
            if (tensor.shape() != entry.shapes[i] || !tensor.same_storage_owner(entry.storage_owners[i]))
                throw std::invalid_argument("prepared parameter storage/shape changed");
            if (tensor.empty()) {
                if (!tensor.shape().empty() || (tensor.legacy_tensor() && tensor.legacy_tensor()->storage.use_count()!=0))
                    throw std::invalid_argument("prepared empty parameter has a shape/foreign owner");
            } else {
                context.validate_parameter(tensor);
                (void)tensor.numel();
            }
        }
    }
    void validate_registered_parameters(Context& context, std::span<const Parameter> tensors) const {
        if (tensors.empty() || tensors.size() > 32)
            throw std::invalid_argument("prepared parameter set requires 1..32 tensor slots");
        bool nonempty_registered_weight = false;
        for (const auto& tensor : tensors) {
            if (tensor.empty()) {
                if (!tensor.shape().empty() || (tensor.legacy_tensor() && tensor.legacy_tensor()->storage.use_count()!=0))
                    throw std::invalid_argument("prepared empty parameter has a shape/foreign owner");
                continue;
            }
            // Establish a real registered shared owner before dereferencing a
            // possibly forged DeviceStorage pointer through Context/Tensor.
            const bool registered = std::any_of(cache.begin(), cache.end(), [&](const auto& item) {
                return item.second.tensor.same_storage_owner(tensor) && item.second.tensor.shape() == tensor.shape();
            });
            if (!registered)
                throw std::invalid_argument("prepared set contains a Tensor that is not a registered immutable weight");
            if (tensor.shape().size() > 8)
                throw std::invalid_argument("prepared parameter belongs to another Context/rank");
            context.validate_parameter(tensor);
            (void)tensor.numel();
            nonempty_registered_weight = true;
        }
        if (!nonempty_registered_weight)
            throw std::invalid_argument("prepared set must retain at least one registered weight/Context owner");
    }
    static void measure_prepared_entry(PreparedEntry& entry) {
        // Actual object/capacity bytes are counted. A separate 64-byte allowance
        // per heap allocation estimates allocator and shared_ptr control-block
        // overhead for the registered Linux profile; this is not a process RSS
        // upper-bound claim. RSS remains an independent Slurm measurement.
        constexpr uint64_t allocation_allowance = 64;
        uint64_t allocations = 2; // unique entry + shared prepared/control block
        const auto& tensors = entry.parameters->tensors;
        entry.payload_bytes = sizeof(PreparedParameterSet) + tensors.capacity() * sizeof(Parameter);
        if (tensors.capacity()) ++allocations;
        // Parameter copies are shared readonly facades; their shape/storage
        // live in the registered cache, not in a fresh per-prepared Tensor.
        entry.container_bytes = sizeof(PreparedEntry) + entry.key.capacity() + 1 +
            entry.configuration.capacity() * sizeof(int64_t) +
            entry.shapes.capacity() * sizeof(std::vector<int64_t>) +
            entry.storage_owners.capacity() * sizeof(Parameter);
        allocations += 1 + (entry.configuration.capacity() != 0) + (entry.shapes.capacity() != 0) +
            (entry.storage_owners.capacity() != 0); // key char storage counted conservatively even for SSO
        for (const auto& shape : entry.shapes) {
            entry.container_bytes += shape.capacity() * sizeof(int64_t);
            if (shape.capacity()) ++allocations;
        }
        entry.allocator_allowance_bytes = allocations * allocation_allowance;
    }

    std::span<const uint8_t> raw(const std::string& key) const {
        const auto found = records.find(key);
        if (found == records.end()) throw std::invalid_argument("missing original NR parameter: " + key);
        const auto* bytes = static_cast<const uint8_t*>(mapping);
        return {bytes + payload_offset + found->second.offset, found->second.bytes};
    }

    std::vector<float> host(const std::string& key, unsigned element_bytes, bool e4m3) const {
        const auto bytes = raw(key);
        if (bytes.size() % element_bytes) throw std::invalid_argument("original NR parameter byte-size/type mismatch: " + key);
        std::vector<float> values(bytes.size() / element_bytes);
        for (size_t i = 0; i < values.size(); ++i) {
            const auto* current = bytes.data() + i * element_bytes;
            if (e4m3) values[i] = rw_e4m3_decode(*current);
            else if (element_bytes == 2) values[i] = f16_decode(static_cast<uint16_t>(little_integer(current, 2)));
            else values[i] = std::bit_cast<float>(static_cast<uint32_t>(little_integer(current, 4)));
        }
        return values;
    }

    Parameter decoded(Context& context, const std::string& key, unsigned element_bytes, bool e4m3,
                   std::vector<int64_t> shape) {
        std::lock_guard lock(mutex);
        const auto type=e4m3 ? "e4m3" : element_bytes == 2 ? "f16" : "f32";
        const auto legacy_identity=cache_key(context,key,type,shape);
        const auto identity = shared_mode ? "shared/"+legacy_identity.substr(legacy_identity.find(':')+1):legacy_identity;
        const auto cached = cache.find(identity);
        if (cached != cache.end()) return cached->second.tensor;
        if(shared_mode) {
            auto pool=shared_pool.lock();
            if(!pool || !registered_model || pool->model().get()!=registered_model.get() ||
                pool->model().owner_before(registered_model) || registered_model.owner_before(pool->model()))
                throw std::runtime_error("canonical immutable pool disappeared; legacy allocation fallback forbidden");
            auto view=pool->acquire_parameter(key,type,shape,false,[&]{return host(key,element_bytes,e4m3);});
            Parameter parameter(std::move(view));cache.emplace(identity,Cached{parameter,shape,{}});return parameter;
        }
        const auto values = host(key, element_bytes, e4m3);
        auto tensor = context.upload(shape, values);
        const auto inserted=cache.emplace(identity, Cached{tensor, shape, {}});
        return inserted.first->second.tensor;
    }
};

Weights::Weights(const std::filesystem::path& safetensors) : implementation_(std::make_shared<Impl>()) {
    auto& impl = *implementation_;
    // Prepared immutable parameter views are the production fast path.  Keep
    // an explicit 0 escape hatch for differential runs that need to measure
    // uncached preparation overhead.
    impl.preparation.enabled = true;
    if (const char* setting = std::getenv("RW_NR_PREPARED_CACHE")) {
        if (std::strcmp(setting, "0") != 0 && std::strcmp(setting, "1") != 0)
            throw std::invalid_argument("RW_NR_PREPARED_CACHE must be exactly 0 or 1");
        impl.preparation.enabled = std::strcmp(setting, "1") == 0;
    }
    if (impl.preparation.enabled) impl.prepared.reserve(impl.preparation.entry_limit);
    impl.preparation.host_container_capacity_bytes = sizeof(impl.prepared) + sizeof(impl.preparation) +
        impl.prepared.capacity() * sizeof(std::unique_ptr<Impl::PreparedEntry>);
    impl.preparation.host_allocator_allowance_bytes = impl.prepared.capacity() ? 64 : 0;
    impl.preparation.host_accounted_bytes = impl.preparation.host_container_capacity_bytes +
        impl.preparation.host_allocator_allowance_bytes;
    impl.preparation.host_accounted_peak_bytes = impl.preparation.host_accounted_bytes;
#ifdef _WIN32
    impl.fd = CreateFileW(safetensors.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
    if (impl.fd == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot open original NR safetensors, Win32 error " + std::to_string(GetLastError()));
    LARGE_INTEGER size{};
    if (GetFileType(impl.fd) != FILE_TYPE_DISK || !GetFileSizeEx(impl.fd, &size) || size.QuadPart < 10)
        throw std::invalid_argument("original NR safetensors is not a complete regular file");
    const auto file_size = static_cast<uint64_t>(size.QuadPart);
#else
    impl.fd = open(safetensors.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (impl.fd < 0) throw std::runtime_error("cannot open original NR safetensors: " + std::string(std::strerror(errno)));
    struct stat metadata{};
    if (fstat(impl.fd, &metadata) != 0 || !S_ISREG(metadata.st_mode) || metadata.st_size < 10)
        throw std::invalid_argument("original NR safetensors is not a complete regular file");
    const auto file_size = static_cast<uint64_t>(metadata.st_size);
#endif
    if (file_size > std::numeric_limits<size_t>::max())
        throw std::overflow_error("original NR safetensors file is too large for host mapping");
    impl.file_bytes = static_cast<size_t>(file_size);
#ifdef _WIN32
    impl.file_mapping = CreateFileMappingW(impl.fd, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (impl.file_mapping)
        impl.mapping = MapViewOfFile(impl.file_mapping, FILE_MAP_READ, 0, 0, 0);
#else
    const auto mapping = mmap(nullptr, impl.file_bytes, PROT_READ, MAP_PRIVATE, impl.fd, 0);
    if (mapping != MAP_FAILED) impl.mapping = mapping;
#endif
    if (!impl.mapping) throw std::runtime_error("cannot map original NR safetensors read-only");
    const auto* bytes = static_cast<const uint8_t*>(impl.mapping);
    const uint64_t header_bytes = little_integer(bytes, 8);
    if (header_bytes < 2 || header_bytes > 64 * 1024 * 1024 || header_bytes > impl.file_bytes - 8)
        throw std::invalid_argument("original NR safetensors header is out of bounds");
    impl.payload_offset = 8 + static_cast<size_t>(header_bytes);
    std::vector<std::unordered_set<std::string>> key_stack;
    auto no_duplicates = [&key_stack](int depth, Json::parse_event_t event, Json& parsed) {
        if (depth > 32) throw std::invalid_argument("original NR safetensors header nesting exceeds bound");
        if (event == Json::parse_event_t::object_start) key_stack.emplace_back();
        else if (event == Json::parse_event_t::object_end) key_stack.pop_back();
        else if (event == Json::parse_event_t::key) {
            if (key_stack.empty() || !key_stack.back().insert(parsed.get<std::string>()).second)
                throw std::invalid_argument("duplicate or unscoped key in original NR safetensors header");
        }
        return true;
    };
    const auto header = Json::parse(bytes + 8, bytes + impl.payload_offset, no_duplicates);
    if (!header.is_object() || header.size() > 65536) throw std::invalid_argument("invalid original NR safetensors header object");
    std::vector<std::pair<size_t, size_t>> intervals;
    const size_t payload_bytes = impl.file_bytes - impl.payload_offset;
    for (auto it = header.begin(); it != header.end(); ++it) {
        const auto& item = it.value();
        if (it.key() == "__metadata__") {
            if (!item.is_object()) throw std::invalid_argument("invalid original NR metadata");
            for (auto metadata_it = item.begin(); metadata_it != item.end(); ++metadata_it)
                if (!metadata_it.value().is_string()) throw std::invalid_argument("NR metadata values must be strings");
            if (item.contains("network")) impl.model_network = item.at("network").get<std::string>();
            continue;
        }
        if (!item.is_object() || item.size() != 3 || item.value("dtype", std::string()) != "U8" ||
            !item.contains("shape") || !item.at("shape").is_array() || !item.contains("data_offsets") ||
            !item.at("data_offsets").is_array() || item.at("data_offsets").size() != 2)
            throw std::invalid_argument("NR parameters must be original raw-U8 safetensors records");
        const auto& shape = item.at("shape");
        if (shape.size() > 8) throw std::invalid_argument("original NR parameter rank exceeds bound");
        size_t count = 1;
        for (const auto& dimension : shape) {
            const size_t extent = nonnegative(dimension, "tensor dimension");
            if (extent == 0 || count > std::numeric_limits<size_t>::max() / extent)
                throw std::overflow_error("original NR parameter shape is empty or overflows");
            count *= extent;
        }
        const size_t start = nonnegative(item.at("data_offsets").at(0), "start offset");
        const size_t end = nonnegative(item.at("data_offsets").at(1), "end offset");
        if (start > end || end > payload_bytes || end - start != count)
            throw std::invalid_argument("original NR parameter shape/byte range mismatch");
        impl.records.emplace(it.key(), Impl::Record{start, count});
        intervals.emplace_back(start, end);
    }
    if (impl.model_network != "ht" && impl.model_network != "small")
        throw std::invalid_argument("unknown original NR network variant");
    if (intervals.empty()) throw std::invalid_argument("original NR safetensors has no parameter records");
    std::sort(intervals.begin(), intervals.end());
    size_t end = 0;
    for (const auto& interval : intervals) {
        if (interval.first != end) throw std::invalid_argument("original NR parameter payload has overlap or holes");
        end = interval.second;
    }
    if (end != payload_bytes) throw std::invalid_argument("original NR safetensors has unaccounted payload bytes");
}

std::string Weights::network() const { return implementation_->model_network; }
std::span<const uint8_t> Weights::raw(const std::string& key) const { return implementation_->raw(key); }
std::span<const uint8_t> Weights::raw(int block, int layer, const std::string& parameter) const {
    return raw(parameter_key(block, layer, parameter));
}
std::vector<float> Weights::host_f16(int block, int layer, const std::string& parameter) const {
    return implementation_->host(parameter_key(block, layer, parameter), 2, false);
}
std::vector<float> Weights::host_f32(int block, int layer, const std::string& parameter) const {
    return implementation_->host(parameter_key(block, layer, parameter), 4, false);
}
std::vector<float> Weights::host_e4m3(int block, int layer, const std::string& parameter) const {
    return implementation_->host(parameter_key(block, layer, parameter), 1, true);
}
Parameter Weights::f16(Context& context, int block, int layer, const std::string& parameter, std::vector<int64_t> shape) {
    return implementation_->decoded(context, parameter_key(block, layer, parameter), 2, false, std::move(shape));
}
Parameter Weights::f32(Context& context, int block, int layer, const std::string& parameter, std::vector<int64_t> shape) {
    return implementation_->decoded(context, parameter_key(block, layer, parameter), 4, false, std::move(shape));
}
Parameter Weights::e4m3(Context& context, int block, int layer, const std::string& parameter, std::vector<int64_t> shape) {
    return implementation_->decoded(context, parameter_key(block, layer, parameter), 1, true, std::move(shape));
}
Parameter Weights::derived(Context& context, const std::string& key, std::vector<int64_t> shape,
                        std::span<const float> values) {
    std::lock_guard lock(implementation_->mutex);
    const auto legacy_identity=cache_key(context,key,"derived",shape);
    const auto identity=implementation_->shared_mode ? "shared/"+legacy_identity.substr(legacy_identity.find(':')+1):legacy_identity;
    const auto found = implementation_->cache.find(identity);
    if (found != implementation_->cache.end()) {
        const auto& previous = found->second.derived_values;
        if (previous.size() != values.size() ||
            (!values.empty() && std::memcmp(previous.data(), values.data(), values.size_bytes()) != 0))
            throw std::invalid_argument("immutable derived NR parameter changed for cache key: " + key);
        return found->second.tensor;
    }
    if(implementation_->shared_mode) {
        auto pool=implementation_->shared_pool.lock();
        if(!pool || pool->model().get()!=implementation_->registered_model.get() ||
            pool->model().owner_before(implementation_->registered_model) || implementation_->registered_model.owner_before(pool->model()))
            throw std::runtime_error("canonical derived pool disappeared; legacy fallback forbidden");
        auto view=pool->acquire_parameter(key,"derived_f32_original_order",shape,true,[&]{return std::vector<float>(values.begin(),values.end());},values);
        Parameter parameter(std::move(view));implementation_->cache.emplace(identity,Impl::Cached{parameter,shape,std::vector<float>(values.begin(),values.end())});return parameter;
    }
    auto tensor = context.upload(shape, values);
    const auto inserted=implementation_->cache.emplace(identity, Impl::Cached{tensor, shape, std::vector<float>(values.begin(), values.end())});
    return inserted.first->second.tensor;
}

std::shared_ptr<const PreparedParameterSet> Weights::find_prepared(Context& context, const std::string& key,
    std::span<const int64_t> configuration) {
    std::lock_guard lock(implementation_->mutex);
    auto& stats = implementation_->preparation;
    ++stats.lookups;
    try {
        Impl::validate_prepared_key(key, configuration);
        if (!stats.enabled) { ++stats.bypasses; return {}; }
        const auto* found = implementation_->lookup_prepared(context.identity(), key);
        if (!found) { ++stats.misses; return {}; }
        if (!Impl::same_configuration(*found, configuration))
            throw std::invalid_argument("prepared parameter configuration changed for key: " + key);
        Impl::validate_prepared_entry(context, *found);
        ++stats.hits;
        return found->parameters;
    } catch (...) {
        ++stats.rejections;
        throw;
    }
}

void Weights::remember_prepared(Context& context, const std::string& key,
    std::span<const int64_t> configuration, std::span<const Parameter> tensors) {
    std::lock_guard lock(implementation_->mutex);
    auto& stats = implementation_->preparation;
    ++stats.completed_preparations;
    try {
        Impl::validate_prepared_key(key, configuration);
        if (!stats.enabled) return;
        implementation_->validate_registered_parameters(context, tensors);
        if (const auto* found = implementation_->lookup_prepared(context.identity(), key)) {
            if (!Impl::same_configuration(*found, configuration) || found->parameters->tensors.size() != tensors.size())
                throw std::invalid_argument("immutable prepared parameter configuration changed");
            Impl::validate_prepared_entry(context, *found);
            for (size_t i = 0; i < tensors.size(); ++i)
                if (!found->parameters->tensors[i].same_storage_owner(tensors[i]) ||
                    found->parameters->tensors[i].shape() != tensors[i].shape())
                    throw std::invalid_argument("immutable prepared parameter storage changed");
            ++stats.duplicate_publications;
            return;
        }
        if (implementation_->prepared.size() >= stats.entry_limit)
            throw std::runtime_error("prepared parameter cache entry quota exhausted");
        // A single candidate is bounded by 32 tensors, rank<=8, key<=256 and
        // configuration<=32. Conservatively reserve 64KiB in the byte quota
        // before allocating candidate metadata; inspect actual capacities too.
        constexpr uint64_t candidate_allowance = 64 * 1024;
        if (stats.host_accounted_bytes > stats.host_byte_limit ||
            candidate_allowance > stats.host_byte_limit - stats.host_accounted_bytes)
            throw std::runtime_error("prepared parameter cache host byte quota exhausted before construction");
        auto entry = std::make_unique<Impl::PreparedEntry>();
        entry->context = context.identity();
        entry->key = key;
        entry->configuration.assign(configuration.begin(), configuration.end());
        auto parameters = std::make_shared<PreparedParameterSet>();
        parameters->tensors.assign(tensors.begin(), tensors.end());
        entry->shapes.reserve(tensors.size());
        entry->storage_owners.reserve(tensors.size());
        for (const auto& tensor : tensors) {
            entry->shapes.push_back(tensor.shape());
            entry->storage_owners.push_back(tensor);
        }
        entry->parameters = std::move(parameters);
        Impl::measure_prepared_entry(*entry);
        const uint64_t bytes = entry->payload_bytes + entry->container_bytes + entry->allocator_allowance_bytes;
        stats.host_accounted_peak_bytes = std::max(stats.host_accounted_peak_bytes, stats.host_accounted_bytes + bytes);
        if (bytes > candidate_allowance || bytes > stats.host_byte_limit - stats.host_accounted_bytes)
            throw std::runtime_error("prepared parameter metadata exceeded its conservative admission envelope");
        stats.host_payload_bytes += entry->payload_bytes;
        stats.host_container_capacity_bytes += entry->container_bytes;
        stats.host_allocator_allowance_bytes += entry->allocator_allowance_bytes;
        stats.host_accounted_bytes += bytes;
        implementation_->prepared.push_back(std::move(entry));
        ++stats.publications;
        stats.entries = implementation_->prepared.size();
    } catch (...) {
        ++stats.rejections;
        throw;
    }
}

PreparationStatistics Weights::preparation_statistics() const {
    std::lock_guard lock(implementation_->mutex);
    return implementation_->preparation;
}
void Weights::clear_device_cache_for_retirement() {
    std::lock_guard lock(implementation_->mutex);
    // Prepared aliases can own exactly the same registered GPU allocations.
    // Clear both while Contexts are still alive to observe physical free errors.
    implementation_->prepared.clear();implementation_->cache.clear();
    decltype(implementation_->prepared){}.swap(implementation_->prepared);
    decltype(implementation_->cache){}.swap(implementation_->cache);
    auto& stats=implementation_->preparation;stats.entries=0;stats.host_payload_bytes=0;
    stats.host_container_capacity_bytes=sizeof(implementation_->prepared)+sizeof(stats);
    stats.host_allocator_allowance_bytes=0;stats.host_accounted_bytes=stats.host_container_capacity_bytes;
}
void Weights::attach_shared_pool(std::weak_ptr<DeviceWeightPool> pool,RegisteredModelOwner model) {
    if(!ModelRegistry::is_registered_owner(model))throw std::invalid_argument("unregistered model owner cannot attach a shared decoder");
    auto owner=pool.lock();
    if(!owner || owner->model().get()!=model.get() || owner->model().owner_before(model) || model.owner_before(owner->model()))
        throw std::invalid_argument("shared decoder model/pool owner mismatch");
    std::lock_guard lock(implementation_->mutex);
    if(implementation_->shared_mode || !implementation_->cache.empty() || !implementation_->prepared.empty())
        throw std::invalid_argument("cannot reattach or promote already materialized legacy weights");
    if(implementation_->model_network!=model->network())throw std::invalid_argument("registered model variant differs from decoder");
    implementation_->shared_pool=std::move(pool);implementation_->registered_model=std::move(model);implementation_->shared_mode=true;
}
} // namespace rw::native_nr
