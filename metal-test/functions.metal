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
    
    float v = 0;
    for (uint i = 0; i < ldims.x; i += size.x) {
        if (row < ldims.y && i + tpos.x < ldims.x) {
            lcache[tpos.y * size.x + tpos.x] = left[row * ldims.x + i + tpos.x];
        }
        if (i + tpos.y < rdims.y && col < rdims.x) {
            rcache[tpos.y * size.x + tpos.x] = right[(i + tpos.y) * rdims.x + col];
        }

        threadgroup_barrier(mem_flags::mem_none);
        
        for(uint j = 0; j < size.x && i + j < ldims.x; ++j){
            v += lcache[tpos.y * size.x + j] * rcache[j * size.x + tpos.x];
        }
        
        threadgroup_barrier(mem_flags::mem_none);
    }
    
    if (row < ldims.y && col < rdims.x) out[row * rdims.x + col] = v;
}

kernel void matmul2(const device float *left  [[ buffer(0) ]],
                    const device float *right [[ buffer(1) ]],
                    constant uint2 &ldims     [[ buffer(2) ]],
                    constant uint2 &rdims     [[ buffer(3) ]],
                    device float *out         [[ buffer(4) ]],
                    uint2 pos                 [[ thread_position_in_grid ]]) {
    if (pos.x >= rdims.x || pos.y >= ldims.y) return;
    
    float v = 0;
    for(uint i = 0; i < ldims.x; ++i){
        v += left[pos.y * ldims.x + i] * right[i * rdims.x + pos.x];
    }
    out[pos.y * rdims.x + pos.x] = v;
}