#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

kernel void matmul(const device float *left  [[ buffer(0) ]],
                   const device float *right [[ buffer(1) ]],
                   constant uint2 &ldims     [[ buffer(2) ]],
                   constant uint2 &rdims     [[ buffer(3) ]],
                   device float *out         [[ buffer(4) ]],
                   threadgroup float *lcache [[ threadgroup(0) ]],
                   threadgroup float *rcache [[ threadgroup(1) ]],
                   uint2 gpos                [[ threadgroup_position_in_grid ]],
                   uint2 tpos                [[ thread_position_in_threadgroup ]],
                   uint2 size                [[ threads_per_threadgroup ]]) {
    uint row = gpos.y * size.y + tpos.y;
    uint col = gpos.x * size.x + tpos.x;
    if (row > ldims.y || col > rdims.x) return;
    
    float v = 0;
    for (uint i = 0; i < ldims.x; i += size.x) {
        lcache[tpos.y * size.x + tpos.x] = left[row * ldims.x + i + tpos.x];
        rcache[tpos.y * size.x + tpos.x] = right[(i + tpos.y) * rdims.x + col];

        threadgroup_barrier(mem_flags::mem_none);
        
        for(uint j = 0; j < size.x && i + j < ldims.x; ++j){
            v += lcache[tpos.y * size.x + j] * rcache[j * size.x + tpos.x];
        }
        
        threadgroup_barrier(mem_flags::mem_none);
    }
    out[row * rdims.x + col] = v;
}