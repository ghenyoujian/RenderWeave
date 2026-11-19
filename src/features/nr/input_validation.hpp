#pragma once
#include <bit>
#include <cstdint>
#include <span>
#include <cstdlib>
#include <cstring>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace rw::nr_sdk {
inline bool portable_valid_sdr(std::span<const float> values) noexcept {
    uint32_t invalid=0;
    for(float value:values) {
        const auto bits=std::bit_cast<uint32_t>(value),magnitude=bits&0x7fffffffu;
        invalid|=uint32_t(magnitude>0x3f800000u)|((bits>>31)&uint32_t(magnitude!=0));
    }
    return invalid==0;
}
inline bool portable_valid_finite(std::span<const float> values) noexcept {
    uint32_t invalid=0;
    for(float value:values)
        invalid|=uint32_t((std::bit_cast<uint32_t>(value)&0x7f800000u)==0x7f800000u);
    return invalid==0;
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) inline bool avx2_valid_sdr(std::span<const float> values) noexcept {
    const auto magnitude_mask=_mm256_set1_epi32(0x7fffffff),one=_mm256_set1_epi32(0x3f800000),zero=_mm256_setzero_si256();
    auto invalid=zero;
    size_t index=0;
    for(;index+8<=values.size();index+=8) {
        const auto bits=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(values.data()+index));
        const auto magnitude=_mm256_and_si256(bits,magnitude_mask);
        const auto negative=_mm256_cmpgt_epi32(zero,bits),is_zero=_mm256_cmpeq_epi32(magnitude,zero);
        invalid=_mm256_or_si256(invalid,_mm256_or_si256(_mm256_cmpgt_epi32(magnitude,one),_mm256_andnot_si256(is_zero,negative)));
    }
    return _mm256_testz_si256(invalid,invalid) && portable_valid_sdr(values.subspan(index));
}
__attribute__((target("avx2"))) inline bool avx2_valid_finite(std::span<const float> values) noexcept {
    const auto exponent=_mm256_set1_epi32(0x7f800000);auto invalid=_mm256_setzero_si256();
    size_t index=0;
    for(;index+8<=values.size();index+=8) {
        const auto bits=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(values.data()+index));
        invalid=_mm256_or_si256(invalid,_mm256_cmpeq_epi32(_mm256_and_si256(bits,exponent),exponent));
    }
    return _mm256_testz_si256(invalid,invalid) && portable_valid_finite(values.subspan(index));
}
__attribute__((target("avx512f"))) inline bool avx512_valid_sdr(std::span<const float> values) noexcept {
    const auto magnitude_mask=_mm512_set1_epi32(0x7fffffff),one=_mm512_set1_epi32(0x3f800000),zero=_mm512_setzero_si512();
    __mmask16 invalid=0;size_t index=0;
    for(;index+16<=values.size();index+=16) {
        const auto bits=_mm512_loadu_si512(values.data()+index),magnitude=_mm512_and_si512(bits,magnitude_mask);
        invalid|=_mm512_cmp_epi32_mask(magnitude,one,_MM_CMPINT_GT) |
            (_mm512_cmp_epi32_mask(bits,zero,_MM_CMPINT_LT)&_mm512_cmp_epi32_mask(magnitude,zero,_MM_CMPINT_NE));
    }
    return invalid==0 && portable_valid_sdr(values.subspan(index));
}
__attribute__((target("avx512f"))) inline bool avx512_valid_finite(std::span<const float> values) noexcept {
    const auto exponent=_mm512_set1_epi32(0x7f800000);__mmask16 invalid=0;size_t index=0;
    for(;index+16<=values.size();index+=16) {
        const auto bits=_mm512_loadu_si512(values.data()+index);
        invalid|=_mm512_cmp_epi32_mask(_mm512_and_si512(bits,exponent),exponent,_MM_CMPINT_EQ);
    }
    return invalid==0 && portable_valid_finite(values.subspan(index));
}
#endif

struct InputScanner {
    bool (*sdr)(std::span<const float>) noexcept;
    bool (*finite)(std::span<const float>) noexcept;
    const char* name;
};
inline const InputScanner& input_scanner() {
    static const InputScanner selected=[] {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        const char* portable=std::getenv("RW_NR_PORTABLE_INPUT_SCAN");
        if(!portable || std::strcmp(portable,"1")!=0) {
            __builtin_cpu_init();
            if(__builtin_cpu_supports("avx512f"))return InputScanner{avx512_valid_sdr,avx512_valid_finite,"avx512f"};
            if(__builtin_cpu_supports("avx2"))return InputScanner{avx2_valid_sdr,avx2_valid_finite,"avx2"};
        }
#endif
        return InputScanner{portable_valid_sdr,portable_valid_finite,"portable"};
    }();
    return selected;
}
inline bool valid_sdr(std::span<const float> values) noexcept {return input_scanner().sdr(values);}
inline bool valid_finite(std::span<const float> values) noexcept {return input_scanner().finite(values);}
}
