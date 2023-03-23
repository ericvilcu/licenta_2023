#ifndef STRUCTURAL_REFINEMENT
#define STRUCTURAL_REFINEMENT 0
#endif
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
            int ids_m = ids * (NDIM + 1);
            for (int i = 0; i <= NDIM; ++i)
                plot[ids_m + i] = 0;
            //TODO: environment
        }
    }
}
//todo? switch to normal arrays for camera? as in, a[20], so it passes all elements but not by reference.
void __global__ backward(const float* camera_data, int ndim, float* point_grad, const float* point_data, int num_points, const float* plot, const float* plot_grad, const float* plot_weights){
    Camera camera{camera_data};
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_points) {
        int ids = idx * (3 + ndim);
        //NOTE: d is calculated both here AND in the forward pass and could theoretically be saved.
        const float* this_point_data = &point_data[ids];
        ScreenCoordsWithDepth d = camera.mapToScreenCoords(make_float4(this_point_data[0], -this_point_data[1], this_point_data[2], 1));
        if (d.valid) {
            int pixel = d.coords.x + d.coords.y * camera.w;
            float depth = d.depth;
            float surface_depth = plot[pixel * (ndim + 1) + ndim];
            //I should probably move the depth test to a shared header, as well as a function for weight (which is currently always 1)
            if (surface_depth * (1 + 0.001) < depth || plot_weights[pixel]<=0)return;
            float weight = plot_weights[pixel];
            float weight_fraction = 1 / weight;
            const float* grad_start = &plot_grad[pixel * (ndim + 1)];
            float* point_grad_start = &(point_grad[ids]);
            float* point_color_grad_start = &(point_grad[ids + 3]);
            for (int i = 0; i < ndim; ++i) {
                point_color_grad_start[i] += grad_start[i] * weight_fraction;
            }
            //estimate for point position refinement, based on: https://arxiv.org/pdf/2110.06635.pdf
            //Note: the edges of the image are iffy, so I exclude them.
            if(STRUCTURAL_REFINEMENT && d.coords.x>0 && d.coords.y>0 && d.coords.x<camera.w-1 && d.coords.y<camera.h-1)//todo: verify the implementation at the camera level.
            {
                auto compute_position_grad = [&](int pixel) -> float
                {
                    const float* pixel_data = &plot[pixel * (ndim + 1)];
                    float pixel_depth = pixel_data[ndim];
                    const float* pixel_grad = &plot_grad[pixel * (ndim + 1)];
                    float pixel_weight = plot_weights[pixel];
                    float d;
                    if (pixel_weight <= 0) {
                        //overwrites background
                        d = 0;
                        for (int i = 0; i < ndim; ++i) {
                            d += (this_point_data[3 + i] - pixel_data[i]) * pixel_grad[i];
                        }
                    } else if (depth < pixel_depth + 0.001) {
                        //blended with other points
                        float new_inv_weight = 1 / (pixel_weight + 1);
                        for (int i = 0; i < ndim; ++i) {
                            d += new_inv_weight * (this_point_data[3 + i] - pixel_weight * pixel_data[i]) * pixel_grad[i];
                        }
                    } else {
                        //does nothing, as it would be far behind other points, not contribuiting.
                        d = 0;
                    }
                    return d;
                };
                int pixelX0 = (d.coords.x-1) + d.coords.y * camera.w;
                int pixelX1 = (d.coords.x+1) + d.coords.y * camera.w;
                int pixelY0 = d.coords.x + (d.coords.y-1) * camera.w;
                int pixelY1 = d.coords.x + (d.coords.y+1) * camera.w;
                float grad_X = 0.5f * (-compute_position_grad(pixelX0) + compute_position_grad(pixelX1));
                float grad_Y = 0.5f * (-compute_position_grad(pixelY0) + compute_position_grad(pixelY1));
                float3 pixel_direction = camera.direction_for_pixel(make_float2(d.coords.x+0.5f, d.coords.y+0.5f));
                float3 wanted_direction = camera.direction_for_pixel(make_float2(d.coords.x + grad_X+0.5f, d.coords.y + grad_Y+0.5f));
                float3 gradient_world_space = make_float3((pixel_direction.x - wanted_direction.x) * -depth, (pixel_direction.y - wanted_direction.y) * -depth, (pixel_direction.z - wanted_direction.z) * -depth);
                point_grad_start[0] += gradient_world_space.x;
                point_grad_start[1] += gradient_world_space.y;
                point_grad_start[2] += gradient_world_space.z;
            }
        }
    }
}
