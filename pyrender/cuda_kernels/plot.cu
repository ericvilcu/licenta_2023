#ifndef CAM_TYPE
#define CAM_TYPE 0
#endif
#ifndef NDIM
#define NDIM 3
#endif
#include <cuda_runtime.h>
//Step 1 of plotting:
void __global__ translateKernelCharToSurface(cudaSurfaceObject_t output, const uchar4* colors, const int h, const int w, const int hd = 0, const int wd = 0) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        surf2Dwrite(colors[idy+wd + (hd+idx) * w], output, idy * sizeof(uchar4), idx, cudaBoundaryModeClamp);
    }
}
//__global__ clear(float*color,float*weight,int mx,int my){
//    int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    int idy = threadIdx.y + blockDim.y * blockIdx.y;
//    if (idx < h && idy < w) {
//    }
//}

void __global__ determine_depth(float* output, const float* points, int num_points, const float* camera_raw_data){
    Camera camera{camera_raw_data};
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_points) {
        //float k=1.0;
        //float x=k/0.;
        int w=(int)camera.w;
        //int h=(int)camera.h;
        // X,Y,Z, (NDIM 'colors')
        int pos = (3 + NDIM) * idx;
        float3 position = make_float3(points[pos], -points[pos + 1], points[pos + 2]);
        auto rez = camera.mapToScreenCoords(position);
        int2 screen_coords = rez.coords;
        float depth = rez.depth;
        if (!rez.valid)return;
        float* pixel_depth = &output[(NDIM + 1) * (screen_coords.x + screen_coords.y * w) + NDIM];
        atomicMin((int*)pixel_depth, *(int*)(float*)&depth);
        static_assert(sizeof(int) == sizeof(float), "Here so the atomicMin hack above works properly. If this does not work, some other suitable replacement for it should be found, potentially using atomicCAS");
    }
}

void __global__ plot(float* output, float* weights, const float* points, int num_points, const float* camera_raw_data){
    Camera camera{camera_raw_data};
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_points) {
        int w=(int)camera.w;
        //int h=(int)camera.h;
        // X,Y,Z, (NDIM 'colors')
        int pos = (3 + NDIM) * idx;
        float4 position = make_float4(points[pos], -points[pos + 1], points[pos + 2], 1);
        auto rez = camera.mapToScreenCoords(position);
        int2 screen_coords = rez.coords;
        float depth = rez.depth;
        if (!rez.valid)return;
        const float pixel_depth = output[(NDIM + 1) * (screen_coords.x + screen_coords.y * w) + NDIM];
        //todo: move depth test to a header or other special function
        if (pixel_depth * (1 + 0.001) < depth)
            return;
        float weight = 1;// 1 / (depth - pixel_depth + 0.01);
        #pragma unroll
        for (int i = 0; i < NDIM; ++i) {
            float* p = &output[(NDIM + 1) * (screen_coords.x + screen_coords.y * w) + i];
            atomicAdd(p, points[pos + 3 + i]);
        }
        float* pixel_color_weight_location = &weights[screen_coords.x + screen_coords.y * w];
        atomicAdd(pixel_color_weight_location, weight);
    }
}

//todo: plot environment simultaniously
void __global__ bundle(float* plot, float* weights, const int h, const int w) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        int ids = idy + idx * w;
        if (weights[ids] > 0) {
            int ids_m = ids * (NDIM + 1);
            float local_weight = weights[ids];
            for (int i = 0; i < NDIM; ++i)
                plot[ids_m + i] = plot[ids_m + i] / local_weight;
        } else {
            //TODO: environment
        }
    }
}