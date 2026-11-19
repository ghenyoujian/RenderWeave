// SPDX-License-Identifier: Apache-2.0
// Adapted from MLX-DLSS python/mlxdlss/tools/package_dlss_sr.py.
#include "backbone.hpp"
#include "shape.hpp"

#include <openssl/sha.h>
#include <stdexcept>

namespace rw::sr::source_backbone {

Weights decode_original(std::span<const std::byte> packed) {
    constexpr std::array<unsigned char, 32> expected = {
        0xdd,0xaa,0x59,0xcd,0x82,0xde,0xed,0x53,0xba,0xd9,0xd6,0x5b,0x82,0xb4,0x54,0xf1,
        0x43,0x9f,0xc4,0x87,0x5e,0x42,0x20,0xb3,0xff,0xaf,0xaf,0xb6,0xcd,0x02,0x96,0xa1};
    std::array<unsigned char, 32> digest{};
    if (packed.size() != packed_bytes ||
        !SHA256(reinterpret_cast<const unsigned char*>(packed.data()), packed.size(), digest.data()) || digest != expected)
        throw std::invalid_argument("SR packed backbone differs from original K 310.7.0");

    Weights result;
    result.elements.reserve(packed.size()/2);
    const auto half = [&](std::size_t offset) {
        return std::uint16_t(std::to_integer<unsigned>(packed[offset]) |
                            (std::to_integer<unsigned>(packed[offset+1]) << 8));
    };
    const auto matrix = [&](std::size_t offset, int rows, int columns, bool position, int chunk, std::uint16_t* output) {
        for (int j = 0; j < columns/16; ++j) {
            for (int i = 0; i < rows/16; ++i) {
                const int tile = chunk ? (i/chunk)*(columns/16)*chunk+j*chunk+i%chunk : j*(rows/16)+i;
                for (int lane = 0; lane < 32; ++lane) {
                    for (int reg = 0; reg < 4; ++reg) {
                        for (int lane_half = 0; lane_half < 2; ++lane_half) {
                            const int group=lane/4, member=lane%4;
                            const int row=position ? group+(reg%2)*8 : member*2+(reg%2)*8+lane_half;
                            const int col=position ? member*2+(reg/2)*8+lane_half : group+(reg/2)*8;
                            output[(i*16+row)*columns+j*16+col] = half(offset+tile*512+(lane*8+reg*2+lane_half)*2);
                        }
                    }
                }
            }
        }
    };
    for (int stage=0; stage<stage_count; ++stage) {
        const int c=channels[stage], h=heads[stage];
        auto& weights=result.stages[stage];
        auto cursor=packed_stage_offsets[stage];
        const auto take = [&](int rows, int cols=0, bool position=false, int chunk=0) {
            const auto start=result.elements.size();
            const int count=rows*(cols ? cols : 1);
            result.elements.resize(start+count);
            auto* out=result.elements.data()+start;
            if (cols) matrix(cursor,rows,cols,position,chunk,out);
            else for (int i=0;i<rows;++i) out[i]=half(cursor+2*i);
            cursor+=2*count;
            return start;
        };
        if (stage==0) {
            weights.embedding_bias=take(32);
            weights.embedding=take(16,32);
        } else if (stage>=6) {
            weights.embedding=take(channels[stage-1],4*c);
            weights.embedding_bias=take(4*c);
        }
        weights.norm=take(c);
        for (int head=0;head<h;++head) {
            auto& target=weights.heads[head];
            const std::array<std::size_t*,3> qkv{&target.query,&target.key,&target.value};
            for (int q=0;q<3;++q) {
                *qkv[q]=result.elements.size();
                result.elements.resize(*qkv[q]+c*32);
                for (int part=0;part<c/32;++part)
                    matrix(cursor+part*6144+q*2048,32,32,false,0,result.elements.data()+*qkv[q]+part*1024);
            }
            cursor+=c*192;
        }
        for (int head=0;head<h;++head) weights.heads[head].position=take(64,64,true);
        weights.attention_bias=take(c);
        for (int head=0;head<h;++head) weights.heads[head].projection=take(32,c);
        weights.ff_scale=take(c);
        weights.ff_output_bias=take(c);
        weights.ff_up=take(c,4*c);
        weights.ff_bias=take(4*c);
        weights.ff_down=take(4*c,c,false,2);
        if (stage<5) {
            const int merge=merge_channels(stage);
            weights.merge=take(4*c,merge);
            weights.merge_bias=take(merge);
        } else if (stage==10) {
            weights.output_bias=take(48);
            weights.output=take(32,48);
        }
        if (cursor!=packed_stage_offsets[stage+1]) throw std::logic_error("SR packed stage boundary mismatch");
    }
    return result;
}

} // namespace rw::sr::source_backbone
