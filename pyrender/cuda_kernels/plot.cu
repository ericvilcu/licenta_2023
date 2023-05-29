#include "cameras.cuh"
#include "environments.cuh"
#ifndef STRUCTURAL_REFINEMENT
#define STRUCTURAL_REFINEMENT 0
#endif
#ifndef CAM_TYPE
#define CAM_TYPE 0
#endif
#ifndef NDIM
#define NDIM 3
#endif
#ifndef CAMERA_GRAD
#define CAMERA_GRAD 0
#endif
#ifndef BLEND_MODE
#define BLEND_MODE 0
#endif
#include <cuda_runtime.h>

#if BLEND_MODE==0//standard
float __hdfi__ test_depth(float my_depth, float min_depth){
    if(min_depth * (1 + 0.001) < my_depth) return -1;
    return 1;
}
#elif BLEND_MODE==1//cutoff
float __hdfi__ test_depth(float my_depth, float min_depth){
    if(min_depth < my_depth) return -1;
    return 1;
}
#elif BLEND_MODE==2//always_visible
float __hdfi__ test_depth(float my_depth, float min_depth){
    return pow(max(1,my_depth/min_depth),2);
}
#elif BLEND_MODE==3//standatd curved
float __hdfi__ test_depth(float my_depth, float min_depth){
    return max(1,my_depth/min_depth - (1.0-0.001));
}
#endif

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

void __global__ plot(float* output, float* weights, const float* points, int num_points, const float* camera_raw_data/*, int32_t* dbg0 = NULL*/){
    Camera camera{camera_raw_data};
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_points) {
        int w=(int)camera.w;
        //int h=(int)camera.h;
        // X,Y,Z, (NDIM 'colors')
        int pos = (3 + NDIM) * idx;
        float4 position = make_float4(points[pos], -points[pos + 1], points[pos + 2], 1);
        //const float* this_point_data = &point_data[ids];
        //ScreenCoordsWithDepth d = camera.mapToScreenCoords(make_float4(this_point_data[0], -this_point_data[1], this_point_data[2], 1));
        auto rez = camera.mapToScreenCoords(position);
        int2 screen_coords = rez.coords;
        int pixel = screen_coords.x + screen_coords.y * w;
        //if(dbg0!=NULL){dbg0[2*idx]=screen_coords.x;dbg0[2*idx+1]=screen_coords.y;}
        float depth = rez.depth;
        if (!rez.valid)return;
        const float pixel_depth = output[(NDIM + 1) * pixel + NDIM];
        //todo: move depth test to a header or other special function
        float weight = test_depth(depth,pixel_depth);
        if (weight<=0)
            return;
        #pragma unroll
        for (int i = 0; i < NDIM; ++i) {
            float* p = &output[(NDIM + 1) * pixel + i];
            atomicAdd(p, points[pos + 3 + i] * weight);
        }
        float* pixel_color_weight_location = &weights[pixel];
        atomicAdd(pixel_color_weight_location, weight);
    }
}

void __global__ bundle(float* plot, float* weights, float* environment_data, const int h, const int w,const float* camera_raw_data){
    Camera camera{camera_raw_data};
    int idy = threadIdx.x + blockDim.x * blockIdx.x;
    int idx = threadIdx.y + blockDim.y * blockIdx.y;
    if (idy < h && idx < w) {
        int pixel = idx + idy * w;
        if (weights[pixel] > 0) {
            int plot_idx = pixel * (NDIM + 1);
            float local_weight = weights[pixel];
            for (int i = 0; i < NDIM; ++i)
                plot[plot_idx + i] = plot[plot_idx + i] / local_weight;
        } else {
            int plot_idx = pixel * (NDIM + 1);
            float3 direction = camera.direction_for_pixel(make_float2(idx,idy));
            sample_environment(&plot[plot_idx],environment_data,direction);//NOTE: this function can be surprisingly heavy.
        }
    }
}
//todo? switch to normal arrays for camera? as in, a[20], so it passes all elements but not by reference.
void __global__ backward(float* camera_gradient, const float* camera_data, float* point_grad, const float* point_data, int num_points, const float* plot, const float* plot_grad, const float* plot_weights/*,int32_t* dbg0 = NULL*/){
    Camera camera{camera_data};
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_points) {
        int ids = idx * (3 + NDIM);
        int w=(int)camera.w;
        int h=(int)camera.h;
        //NOTE: d is calculated both here AND in the forward pass and could theoretically be saved.
        const float* this_point_data = &point_data[ids];
        ScreenCoordsWithDepth d = camera.mapToScreenCoords(make_float4(this_point_data[0], -this_point_data[1], this_point_data[2], 1));
        if (d.valid) {
            int pixel = d.coords.x + d.coords.y * w;
            //if(dbg0!=NULL){dbg0[2*idx]=d.coords.x;dbg0[2*idx+1]=d.coords.y;}
            float depth = d.depth;
            float pixel_depth = plot[pixel * (NDIM + 1) + NDIM];
            float weight = test_depth(depth,pixel_depth);
            const float* grad_start = &plot_grad[pixel * (NDIM + 1)];
            float* point_grad_start = &(point_grad[ids]);
            bool visible= (weight>0);
            if (visible){
                float pixel_weight = plot_weights[pixel];
                float weight_fraction = pixel_weight / weight;
                float* point_color_grad_start = &(point_grad_start[3]);
                for (int i = 0; i < NDIM; ++i) {
                    point_color_grad_start[i] += grad_start[i] * weight_fraction;
                }
            }
            //estimate for point position refinement, based on: https://arxiv.org/pdf/2110.06635.pdf
            //Note: the edges of the image are iffy, so I exclude them.
            //TODO: GO OVER THIS AGAIN; MAYBE REWRITE SOME STUFF
            if(STRUCTURAL_REFINEMENT!=0 && d.coords.x>0 && d.coords.y>0 && d.coords.x<w-1 && d.coords.y<h-1)//todo: verify the implementation at the camera level.
            {
                auto compute_position_grad = [&](int pixel) -> float
                {
                    const float* pixel_data = &plot[pixel * (NDIM + 1)];
                    float pixel_depth = plot[pixel * (NDIM + 1) + NDIM];
                    const float* pixel_grad = &plot_grad[pixel * (NDIM + 1)];
                    float pixel_weight = plot_weights[pixel];
                    float d=0.0;
                    if (pixel_weight <= 0) {
                        //overwrites background
                        for (int i = 0; i < NDIM; ++i) {
                            d += (this_point_data[3 + i] - pixel_data[i]) * pixel_grad[i];
                        }
                    } else{
                        float weight = test_depth(depth,pixel_depth);
                        if (weight<=0) {
                            //does nothing, as it would be far behind other points, not contribuiting.
                        } else {
                            //blended with other points
                            float new_inv_weight = 1 / (pixel_weight+1);
                            for (int i = 0; i < NDIM; ++i) {
                                d += new_inv_weight * (weight*this_point_data[3 + i] - pixel_weight * pixel_data[i]) * pixel_grad[i];
                            }
                        }
                    }
                    return d;
                };
                float stability=1;

#ifdef COMPUTE_STABILITY
#if COMPUTE_STABILITY==1
                auto compute_leave_grad = [&]() -> float
                {
                    if(!visible)return 1;
                    const float* pixel_data = &plot[pixel * (NDIM + 1)];
                    float pixel_depth = plot[pixel * (NDIM + 1) + NDIM];
                    const float* pixel_grad = &plot_grad[pixel * (NDIM + 1)];
                    float pixel_weight = plot_weights[pixel];
                    if (pixel_weight <= 0) {
                        //overwrites background
                        assert(false);//"Pixel reprojection landed in zero-weight spot"
                        return 0.5; //this should not happen;
                    } else{
                        float weight = test_depth(depth,pixel_depth);
                        if (weight<=0) {
                            assert(false);//"Pixel reprojection landed in occluded spot"
                            return 0.5; //this should also not happen;
                        } else if(weight<=pixel_weight){
                            return 0.5;//behind might be background or some other point. too hard to compute either way.
                        } else {
                            //blended with other points
                            float new_inv_weight = 1 / (pixel_weight-1);
                            float effect=0;
                            for (int i = 0; i < NDIM; ++i) {
                                //NOTE: negative gradient means positive movement
                                effect -= new_inv_weight * (this_point_data[3 + i] - pixel_weight * pixel_data[i]) * pixel_grad[i];
                            }
                            //if(effect<=0)return 1;//leaving current pixel will probably have a positive impact
                            return min(1.0f,1/(1+pow(2,effect)));
                        }
                    }
                };
                stability=compute_leave_grad();
#endif
#endif
                int pixelX0 = (d.coords.x-1) + d.coords.y * w;
                int pixelX1 = (d.coords.x+1) + d.coords.y * w;
                int pixelY0 = d.coords.x + (d.coords.y-1) * w;
                int pixelY1 = d.coords.x + (d.coords.y+1) * w;
                float grad_X = 0.5f * (-compute_position_grad(pixelX0) + compute_position_grad(pixelX1)) * stability;
                float grad_Y = 0.5f * (-compute_position_grad(pixelY0) + compute_position_grad(pixelY1)) * stability;
                float3 pixel_direction = camera.direction_for_pixel(make_float2(d.coords.x, d.coords.y));
                float3 wanted_direction = camera.direction_for_pixel(make_float2(d.coords.x + grad_X, d.coords.y + grad_Y));
                float3 gradient_screen_space = make_float3((pixel_direction.x - wanted_direction.x) * -depth, (pixel_direction.y - wanted_direction.y) * -depth, (pixel_direction.z - wanted_direction.z) * -depth);
                point_grad_start[0] += gradient_screen_space.x;
                point_grad_start[1] += gradient_screen_space.y;
                point_grad_start[2] += gradient_screen_space.z;
/*#ifdef CAMERA_GRAD
#pragma message("Warn: Only camera position gradient is implemented, rotation and intrinsics are not.")
                atomicAdd(&camera_gradient[4+9+0],-gradient_screen_space.x);
                atomicAdd(&camera_gradient[4+9+1],-gradient_screen_space.y);
                atomicAdd(&camera_gradient[4+9+2],-gradient_screen_space.z);
#endif*/
            }
        }
    }
}

//Note: plot currently unused
void __global__ backward_pixel(float*cam_data_grad,
    const float*camera_data,
    const float*plot, const float*plot_grad, const float*weights,
    int h,int w,
    const float* environment,float*environment_grad){
    Camera camera{camera_data};
    int idy = threadIdx.x + blockDim.x * blockIdx.x;
    int idx = threadIdx.y + blockDim.y * blockIdx.y;
    if (idy < h && idx < w) {
        int ids = idx + idy * w;
        if (weights[ids] <= 0.0f) {
            int ids_m = ids * (NDIM + 1);
            float3 direction = camera.direction_for_pixel(make_float2(idx, idy));
            backward_environment(plot_grad+ids_m, environment, environment_grad, direction);
        }
    }
}