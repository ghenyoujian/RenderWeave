// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "backbone.hpp"

namespace rw::sr::source_backbone {

inline constexpr int channels[stage_count]={32,64,64,96,128,160,128,96,64,64,32};
inline constexpr int heads[stage_count]={2,2,2,4,4,8,4,4,2,2,2};
inline constexpr std::array<std::size_t,stage_count+1> packed_stage_offsets{
    0,67264,215872,380928,759552,1384576,2189952,2782848,3161984,3327360,3476352,3545824};

inline constexpr int merge_channels(int stage) {
    const int align=16*heads[stage];
    return (channels[stage+1]+align-1)/align*align;
}

struct Stage {
    int width,height,x,y,grid_x,grid_y;
    std::size_t windows() const { return std::size_t(grid_x)*grid_y; }
};

inline std::array<Stage,stage_count> stage_geometry(Geometry geometry, unsigned phase) {
    constexpr int phases[8][2]={{0,2},{5,6},{4,0},{1,4},{7,5},{2,1},{3,7},{6,3}};
    int x[6], y[6];
    x[5]=phases[phase%8][0]; y[5]=phases[phase%8][1];
    for (int i=4;i>=0;--i) {x[i]=(x[i+1]*2+4)%8; y[i]=(y[i+1]*2+4)%8;}
    std::array<Stage,stage_count> output;
    for (int i=0;i<stage_count;++i) {
        const int level=i<5 ? i : 10-i;
        const int w=geometry.width>>level, h=geometry.height>>level;
        output[i]={w,h,x[level],y[level],(w+x[level]+7)/8,(h+y[level]+7)/8};
    }
    return output;
}

} // namespace rw::sr::source_backbone
