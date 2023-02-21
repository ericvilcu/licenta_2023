#include "CudaIncludesAndMacros.cuh"
#include "PlotPointsBackwardsPasses.cuh"
#include "cuda_debug_utils.cuh"

template <typename camera_type_partial>
__global__ void backwards_points(const camera_type_partial camera, int ndim, float* point_grad, const float* point_data, int num_points, const float* plot, const float* plot_grad, const float* plot_weights) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_points) {
        int ids = idx * (3 + ndim);
        //NOTE: d is calculated both here AND in the forward pass and could theoretically be saved.
        ScreenCoordsWithDepth d = camera.mapToScreenCoords(make_float4(point_data[ids + 0], -point_data[ids + 1], point_data[ids + 2], 1));
        if (d.valid) {
            int pixel = d.coords.x + d.coords.y * camera.w;
            float depth = d.depth;
            float surface_depth = plot[pixel * (ndim + 1) + ndim];
            //I should probably move the depth test to a shared header, as well as a function for weight (which is currently always 1)
            if (surface_depth * (1 + 0.001) < depth || plot_weights[pixel]<=0)return;
            float weight_fraction = 1 / plot_weights[pixel];
            const float* grad_start = &plot_grad[pixel * (ndim + 1)];
            float* point_grad_start = &(point_grad[ids + 3]);
            for (int i = 0; i < ndim; ++i) {
                //Do not ask me wy this has to be a minus. I do not know.
                //point_grad_start[i] -= grad_start[i] * weight_fraction;
                point_grad_start[i] += grad_start[i] * weight_fraction;
            }
            //todo: point position refinement, somehow.
        }
    }
}

template <typename camera_type_partial>
__global__ void backwards_environment_v2(const camera_type_partial camera, int ndim, float* environment_grad, int environment_resolution, const float* plot_weights, const float* plot_grad) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < camera.h && idy < camera.w) {
        int ids = (idy + idx * camera.w);
        int ids_m = (idy + idx * camera.w) * (ndim + 1);
        if (plot_weights[ids] <= 0) {//weight = 0 would mean no points landed there
            float3 direction = camera.direction_for_pixel(make_int2(idy, idx));
            unsigned int adress = (ndim+1)*pixel_from_cubemap_coords(environment_resolution, cubemap_coords(environment_resolution, direction));
            //Maybe I should use supersampling? There seems to be somme weird static that rendering at lower resolutions seems to cause.
            for (int i = 0; i < ndim + 1; ++i) {
                atomicAdd(&environment_grad[adress + i], plot_grad[ids_m + i]);
            }
            //atomicAdd(&environment_grad[adress + ndim], plot_grad[ids + ndim]);
            //NOTE: this represents depth, and was separated for no good reason.
        }
    }
}


template <typename camera_type_partial>
cudaError_t backwards_environment_for_camera_v2(const camera_type_partial& camera, int ndim, 
    void* environment_grad, int environment_resolution,
    const void* plot_weights, const void* plot_grad) {
    cudaError_t cudaStatus;
    backwards_environment_v2 BEST_2D_KERNEL(camera.h, camera.w) (camera, ndim, (float*)environment_grad, environment_resolution, (float*)plot_weights, (float*)plot_grad);
    AFTER_FUNCTION_CALL_CHECK();
Error:
    return cudaStatus;
}

template <typename camera_type_partial>
cudaError_t backwards_points_for_camera_v2(const camera_type_partial& camera, int ndim,
    const void* point_grad, const void* point_data, int num_points,
    const void* plot, const void* plot_grad, const void* plot_weights) {
    cudaError_t cudaStatus;
    backwards_points BEST_LINEAR_KERNEL(num_points) (camera, ndim, (float*)point_grad, (const float*)point_data, num_points, (const float*)plot, (const float*)plot_grad, (const float*)plot_weights);
    AFTER_FUNCTION_CALL_CHECK();
Error:
    return cudaStatus;
}



template <typename camera_type_partial>
cudaError_t backwards_for_camera_v2(const camera_type_partial& camera, int ndim,
    const void* point_data, void* point_grad, int num_points,
    const void* environment, void* environment_grad, int environment_resolution,
    const void* plot, const void* plot_weights, const void* plot_grad){
    cudaError_t cudaStatus;
    cudaStatus = backwards_environment_for_camera_v2(camera, ndim, environment_grad, environment_resolution, plot_weights, plot_grad);
    STATUS_CHECK();
    cudaStatus = backwards_points_for_camera_v2(camera, ndim, point_grad, point_data, num_points, plot, plot_grad, plot_weights);
    STATUS_CHECK();
    ENSURE_SYNC();
Error:
    return cudaStatus;
}


cudaError_t PlotPointsBackwardsPass_v2(const std::shared_ptr<CameraDataItf> camera, int ndim,
    const void* point_data, void* point_grad, int num_points,
    const void* environment, void* environment_grad, int environment_resolution,
    const void* plot, const void* plot_weights, const void* plot_grad)
{
    //todo? delete environment from signature and calls since it is not used?
    cudaError_t cudaStatus;
    int h = camera->get_height();
    int w = camera->get_width();
    switch (camera->type())
    {
    case INTERACTIVE:
        //this branch shouldn't even be used, as in theory it would mean an interactive camera is being used for training data.
        throw "Can't use interactive camera for backwards pass.";
        /*cudaStatus = backwards_for_camera_v2(((InteractiveCameraData&)(*camera)).prepareForGPU(w, h), ndim,
            point_data, point_grad, num_points,
            environment, environment_grad, environment_resolution, plot, plot_weights, plot_grad);
        STATUS_CHECK();*/
        break;
    case PINHOLE_PROJECTION:
        cudaStatus = backwards_for_camera_v2(((PinholeCameraData    &)(*camera)).prepareForGPU(w, h), ndim,
            point_data, point_grad, num_points,
            environment, environment_grad, environment_resolution, plot, plot_weights, plot_grad);
        STATUS_CHECK();
        break;
    default:
        std::cerr << "INVALID CAMERA FOUND AT " << __LINE__ << " of " << __FILE__ << '\n';
        break;
    }
Error:
    //TODO: implement
    return cudaStatus;
}
