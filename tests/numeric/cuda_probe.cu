// Standalone GPU probe. Run only from an allocated Slurm job in a published
// release. No weights, training, model substitution or large tensors are used.
// Oracle: E4M3FN is enumerated arithmetically on the host; the device uses CUDA's
// independent conversion API. Sampler expected values are analytical fixtures.
#include "../../src/backends/cuda/probe_kernels.cuh"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t result, const char* expression) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " +
                                 cudaGetErrorString(result));
    }
}
#define RW_CUDA(expression) check((expression), #expression)

std::string json_string(const char* text) {
    std::ostringstream result;
    result << '"';
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text);
         *p; ++p) {
        if (*p == '"' || *p == '\\') result << '\\' << char(*p);
        else if (*p < 0x20) {
            result << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                   << unsigned(*p) << std::dec;
        } else result << char(*p);
    }
    result << '"';
    return result.str();
}

float oracle_decode(unsigned char code) {
    const unsigned magnitude = code & 0x7FU;
    if (magnitude == 0x7FU) return std::numeric_limits<float>::quiet_NaN();
    const unsigned exponent = magnitude >> 3;
    const unsigned fraction = magnitude & 7U;
    const float value = exponent == 0
        ? std::ldexp(float(fraction), -9)
        : std::ldexp(float(8U + fraction), int(exponent) - 10);
    return std::copysign(value, (code & 0x80U) ? -1.0F : 1.0F);
}

unsigned char oracle_encode(float value) {
    if (std::isnan(value)) return 0x7FU;
    const unsigned char sign = std::signbit(value) ? 0x80U : 0;
    const double magnitude = std::fabs(double(value));
    if (magnitude >= 448.0) return sign | 0x7EU;
    // A brute-force closest representable value is deliberately independent of
    // the bit manipulations/rounding algorithm used by the CUDA implementation.
    unsigned best = 0;
    double distance = magnitude;
    for (unsigned candidate = 1; candidate <= 126; ++candidate) {
        const double next_distance = std::fabs(
            magnitude - double(oracle_decode(static_cast<unsigned char>(candidate))));
        if (next_distance < distance ||
            (next_distance == distance && !(candidate & 1U))) {
            best = candidate;
            distance = next_distance;
        }
    }
    return sign | static_cast<unsigned char>(best);
}

bool same_float(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

class Stream {
public:
    Stream() { RW_CUDA(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking)); }
    ~Stream() { if (value_) cudaStreamDestroy(value_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    operator cudaStream_t() const { return value_; }
private:
    cudaStream_t value_ = nullptr;
};

std::size_t allocated_bytes = 0;
std::size_t peak_bytes = 0;
std::size_t quarantined_bytes = 0;

template<class T> class Buffer {
public:
    Buffer(std::size_t count, cudaStream_t stream) : count_(count), stream_(stream) {
        RW_CUDA(cudaMalloc(reinterpret_cast<void**>(&value_), sizeof(T) * count));
        allocated_bytes += sizeof(T) * count;
        if (allocated_bytes > peak_bytes) peak_bytes = allocated_bytes;
    }
    ~Buffer() {
        if (!value_) return;
        // A host exception or failed launch must not return an in-flight buffer
        // to a pool. On an unconfirmable device completion this process retains
        // the allocation until driver/process cleanup, and never reuses it.
        if (cudaStreamSynchronize(stream_) == cudaSuccess) cudaFree(value_);
        else quarantined_bytes += sizeof(T) * count_;
        allocated_bytes -= sizeof(T) * count_;
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data() const { return value_; }
    void upload(const T* source) {
        RW_CUDA(cudaMemcpyAsync(value_, source, sizeof(T) * count_,
                                cudaMemcpyHostToDevice, stream_));
    }
    void download(T* destination) {
        RW_CUDA(cudaMemcpyAsync(destination, value_, sizeof(T) * count_,
                                cudaMemcpyDeviceToHost, stream_));
        RW_CUDA(cudaStreamSynchronize(stream_));
    }
private:
    T* value_ = nullptr;
    std::size_t count_ = 0;
    cudaStream_t stream_ = nullptr;
};

struct Results {
    unsigned decode_cases = 0;
    unsigned decode_failures = 0;
    unsigned encode_cases = 0;
    unsigned encode_failures = 0;
    unsigned gemm_cases = 0;
    unsigned gemm_failures = 0;
    unsigned sampling_cases = 0;
    unsigned sampling_failures = 0;
    double gemm_max_abs_error = 0;
    unsigned total_failures() const {
        return decode_failures + encode_failures + gemm_failures + sampling_failures;
    }
};

void probe_conversions(cudaStream_t stream, Results& results) {
    std::array<unsigned char, 256> codes{};
    std::array<float, 256> decoded{};
    std::vector<float> values;
    for (unsigned index = 0; index < codes.size(); ++index) {
        codes[index] = static_cast<unsigned char>(index);
        values.push_back(oracle_decode(codes[index]));
    }
    // Both signs, every finite adjacent-value midpoint and immediately adjacent
    // binary32 values. This includes the zero/subnormal and exponent boundaries.
    for (unsigned code = 0; code < 126; ++code) {
        const float midpoint = (oracle_decode(static_cast<unsigned char>(code)) +
                                 oracle_decode(static_cast<unsigned char>(code + 1))) / 2;
        for (const float value : {
                std::nextafter(midpoint, -std::numeric_limits<float>::infinity()),
                midpoint,
                std::nextafter(midpoint, std::numeric_limits<float>::infinity())}) {
            values.push_back(value);
            values.push_back(-value);
        }
    }
    for (const float value : {
            0.0F, -0.0F, 448.0F, -448.0F, 449.0F, -449.0F, 464.0F, -464.0F,
            std::numeric_limits<float>::denorm_min(),
            -std::numeric_limits<float>::denorm_min(),
            std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
            -std::numeric_limits<float>::quiet_NaN()}) values.push_back(value);

    std::vector<unsigned char> encoded(values.size());
    Buffer<unsigned char> device_codes(codes.size(), stream);
    Buffer<float> device_decoded(decoded.size(), stream);
    Buffer<float> device_values(values.size(), stream);
    Buffer<unsigned char> device_encoded(encoded.size(), stream);
    device_codes.upload(codes.data());
    device_values.upload(values.data());
    rw::cuda_probe::convert<<<(unsigned(values.size()) + 127) / 128, 128, 0, stream>>>(
        device_codes.data(), device_decoded.data(), device_values.data(),
        device_encoded.data(), unsigned(codes.size()), unsigned(values.size()));
    RW_CUDA(cudaGetLastError());
    device_decoded.download(decoded.data());
    device_encoded.download(encoded.data());
    for (unsigned index = 0; index < codes.size(); ++index) {
        ++results.decode_cases;
        if (!same_float(decoded[index], oracle_decode(codes[index]))) {
            ++results.decode_failures;
            std::cerr << "decode code=" << index << " actual=" << decoded[index]
                      << " expected=" << oracle_decode(codes[index]) << '\n';
        }
    }
    for (unsigned index = 0; index < values.size(); ++index) {
        ++results.encode_cases;
        // CUDA may choose either NaN sign; only the E4M3FN NaN class is defined.
        const bool matches = std::isnan(values[index])
            ? (encoded[index] & 0x7FU) == 0x7FU
            : encoded[index] == oracle_encode(values[index]);
        if (!matches) {
            ++results.encode_failures;
            std::cerr << "encode input_index=" << index << " input=" << values[index]
                      << " actual=" << unsigned(encoded[index]) << " expected="
                      << unsigned(oracle_encode(values[index])) << '\n';
        }
    }
}

void probe_gemm(cudaStream_t stream, Results& results) {
    constexpr unsigned m = 3, n = 5, k = 7;  // intentionally non-tile-aligned
    constexpr std::array<unsigned char, 8> pool{
        0x00, 0x38, 0xB8, 0x40, 0xC0, 0x28, 0xA8, 0x30};
    std::array<unsigned char, m * k> a{};
    std::array<unsigned char, k * n> b{};
    std::array<float, m * n> actual{};
    std::array<unsigned char, m * n> published{};
    Buffer<unsigned char> device_a(a.size(), stream), device_b(b.size(), stream);
    Buffer<float> device_c(actual.size(), stream);
    Buffer<unsigned char> device_published(published.size(), stream);
    for (unsigned fixture = 0; fixture < 6; ++fixture) {
        for (unsigned index = 0; index < a.size(); ++index) {
            switch (fixture) {
            case 0: a[index] = 0; break;  // zero matrix
            case 1: a[index] = (index % k == index / k) ? 0x38 : 0; break;
            case 2: a[index] = index % 2 ? 0xB8 : 0x38; break;
            case 3: a[index] = pool[(index % k) % pool.size()]; break; // repeated rows
            case 4: a[index] = pool[(index * 5 + 3) % pool.size()]; break;
            default:
                a[index] = index % k == index / k
                    ? (index / k == 0 ? 0x7E : index / k == 1 ? 0xFE : 0x01)
                    : 0;
                break;  // finite extrema and subnormal, one exact product
            }
        }
        for (unsigned index = 0; index < b.size(); ++index) {
            constexpr std::array<unsigned char, 8> wide_pool{
                0x7E, 0xFE, 0x01, 0x81, 0x38, 0xB8, 0x40, 0xC0};
            b[index] = fixture == 5
                ? wide_pool[(index * 3 + 1) % wide_pool.size()]
                : pool[(index * 3 + 1) % pool.size()];
        }
        device_a.upload(a.data());
        device_b.upload(b.data());
        rw::cuda_probe::gemm<<<1, 32, 0, stream>>>(
            device_a.data(), device_b.data(), device_c.data(),
            device_published.data(), m, n, k);
        RW_CUDA(cudaGetLastError());
        device_c.download(actual.data());
        device_published.download(published.data());
        for (unsigned row = 0; row < m; ++row) {
            for (unsigned column = 0; column < n; ++column) {
                // Chosen dyadic fixtures have exactly representable products and
                // sums. A wide oracle therefore expects exact binary32 equality.
                double exact = 0;
                for (unsigned inner = 0; inner < k; ++inner) {
                    exact += double(oracle_decode(a[row * k + inner])) *
                             double(oracle_decode(b[inner * n + column]));
                }
                const unsigned index = row * n + column;
                const float expected = float(exact);
                const double error = std::fabs(double(actual[index]) - exact);
                if (error > results.gemm_max_abs_error) results.gemm_max_abs_error = error;
                ++results.gemm_cases;
                if (!same_float(actual[index], expected) ||
                    published[index] != oracle_encode(expected)) {
                    ++results.gemm_failures;
                    std::cerr << "gemm fixture=" << fixture << " row=" << row
                              << " column=" << column << " actual=" << actual[index]
                              << " expected=" << expected << '\n';
                }
            }
        }
    }
}

void probe_sampling(cudaStream_t stream, Results& results) {
    constexpr std::array<float, 6> image{0, 4, 8, 12, 16, 20};
    const std::array<float2, 11> coordinates{{
        {0, 0}, {2, 1}, {-0.25F, 0.5F}, {2.25F, 0.5F}, {0.5F, 0.5F},
        {1.5F, 0}, {-100, -100}, {100, 100}, {0, 0},
        {std::numeric_limits<float>::quiet_NaN(), 0},
        {std::numeric_limits<float>::infinity(), 0}}};
    constexpr std::array<unsigned char, 11> validity{1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1};
    // Direct analytic fixture values, rather than a duplicate bilinear function.
    constexpr std::array<float, 8> expected{0, 20, 6, 14, 8, 6, 0, 20};
    std::array<float, coordinates.size()> output{};
    std::array<unsigned char, coordinates.size()> output_valid{};
    Buffer<float> device_image(image.size(), stream), device_output(output.size(), stream);
    Buffer<float2> device_coordinates(coordinates.size(), stream);
    Buffer<unsigned char> device_validity(validity.size(), stream);
    Buffer<unsigned char> device_output_valid(output_valid.size(), stream);
    device_image.upload(image.data());
    device_coordinates.upload(coordinates.data());
    device_validity.upload(validity.data());
    rw::cuda_probe::bilinear_clamp<<<1, 32, 0, stream>>>(
        device_image.data(), 3, 2, device_coordinates.data(), device_validity.data(),
        device_output.data(), device_output_valid.data(), unsigned(output.size()));
    RW_CUDA(cudaGetLastError());
    device_output.download(output.data());
    device_output_valid.download(output_valid.data());
    for (unsigned index = 0; index < output.size(); ++index) {
        ++results.sampling_cases;
        const bool expected_valid = index < expected.size();
        if (output_valid[index] != expected_valid ||
            (expected_valid && !same_float(output[index], expected[index])) ||
            (!expected_valid && !std::isnan(output[index]))) {
            ++results.sampling_failures;
            std::cerr << "sampling fixture=" << index << " actual=" << output[index]
                      << " validity=" << unsigned(output_valid[index]) << '\n';
        }
    }
}

}  // namespace

int main() {
    const char* job = std::getenv("SLURM_JOB_ID");
    if (!job || !*job) {
        std::cerr << "Allocated Slurm execution is required for GPU probes\n";
        return 64;
    }
    try {
        int device = 0, runtime = 0, driver = 0;
        RW_CUDA(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        RW_CUDA(cudaGetDeviceProperties(&properties, device));
        RW_CUDA(cudaRuntimeGetVersion(&runtime));
        RW_CUDA(cudaDriverGetVersion(&driver));
        Results results;
        Stream stream;
        probe_conversions(stream, results);
        probe_gemm(stream, results);
        probe_sampling(stream, results);
        RW_CUDA(cudaStreamSynchronize(stream));
        std::cout << "{\"project_id\":\"renderweave\",\"scope\":\"cuda_arithmetic_probe_only\","
                  << "\"original_model_executed\":false,\"slurm_job_id\":" << json_string(job)
                  << ",\"device\":" << json_string(properties.name)
                  << ",\"compute_capability\":\"" << properties.major << '.' << properties.minor << '"'
                  << ",\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver
                  << ",\"decode_cases\":" << results.decode_cases
                  << ",\"decode_failures\":" << results.decode_failures
                  << ",\"encode_cases\":" << results.encode_cases
                  << ",\"encode_failures\":" << results.encode_failures
                  << ",\"gemm_cases\":" << results.gemm_cases
                  << ",\"gemm_failures\":" << results.gemm_failures
                  << ",\"gemm_max_abs_error\":" << results.gemm_max_abs_error
                  << ",\"sampling_cases\":" << results.sampling_cases
                  << ",\"sampling_failures\":" << results.sampling_failures
                  << ",\"explicit_device_allocation_peak_bytes\":" << peak_bytes
                  << ",\"quarantined_bytes\":" << quarantined_bytes
                  << ",\"passed\":" << (results.total_failures() == 0 ? "true" : "false") << "}\n";
        return results.total_failures() == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::cout << "{\"project_id\":\"renderweave\",\"scope\":\"cuda_arithmetic_probe_only\","
                  << "\"passed\":false,\"error\":" << json_string(error.what())
                  << ",\"quarantined_bytes\":" << quarantined_bytes << "}\n";
        return 2;
    }
}
