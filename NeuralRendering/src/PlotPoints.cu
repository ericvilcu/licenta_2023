#include "CudaIncludesAndMacros.cuh"
#include "PlotPoints.cuh"
#include "cuda_debug_utils.cuh"

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 600
static_assert(false, "only 6.0 or greater currently supported. (due to AtomicAdd)");
#endif
#endif // __CUDA_ARCH__

void __global__ translateKernel(cudaSurfaceObject_t output, const uchar4* colors, const int h, const int w, const int hd = 0, const int wd = 0) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        surf2Dwrite(colors[idy+wd + (hd+idx) * w], output, idy * sizeof(uchar4), idx, cudaBoundaryModeClamp);
    }
}

void __global__ bundleDepthWithColor_v2(float* output, const float* weighted_colors, const float* depth, const int h, const int w, const int ndim) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        //ndim 'colors', and 1 weight.
        int ids = (idy + idx * w);
        int ids_m = ids * (ndim + 1);
        const float* input_position = &weighted_colors[ids_m];
        float* output_position = &output[ids_m];
        float depth_local = depth[ids];
        float weight = input_position[ndim];
        for (int i = 0; i < ndim; ++i)
            output_position[i] = input_position[i] / weight;
        output_position[ndim] = depth_local;
    }
}

template <typename CAMERA_DATA_TYPE>
void __global__ plotKernel_v2(float* output, const float* depth_aux, const int h, const int w, const float* points, int ndim, const int l, const CAMERA_DATA_TYPE camera) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < l) {
        // X,Y,Z, (ndim 'colors')
        int pos = (3 + ndim) * idx;
        float4 position = make_float4(points[pos], -points[pos + 1], points[pos + 2], 1);
        auto rez = camera.mapToScreenCoords(position);
        int2 screen_coords = rez.coords;
        float depth = rez.depth;
        if (!rez.valid)return;

        float surface_depth = depth_aux[screen_coords.x + screen_coords.y * w];
        //todo: move depth test to a header.
        if (surface_depth * (1 + 0.001) < depth)return;
        float weight = 1;// color.w / (position.z - surface_depth + 0.01);
        for (int i = 0; i < ndim; ++i) {
            float* p = &output[(ndim + 1) * (screen_coords.x + screen_coords.y * w) + i];
            atomicAdd(p, points[pos + 3 + i]);
        }
        float* pixel_color_weight_location = &output[(ndim + 1) * (screen_coords.x + screen_coords.y * w) + ndim];
        atomicAdd(pixel_color_weight_location, weight);
    }
}
/*
* Utility function used to determine the depth for each pixel, such that obstructed pixels will not be drawn.
* Note: this ruins anything that has to do with transparency, and that should be considered and adjusted for.
* TODO: research deferred blending
* https://www.ifi.uzh.ch/dam/jcr:ffffffff-82b7-d340-ffff-ffffec077b00/PointRendering.pdf
*/
template <typename CAMERA_DATA_TYPE>
void __global__ determineDepth(float* depth_aux, const int h, const int w, const float* points, const int l, const CAMERA_DATA_TYPE camera) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < l) {
        int pos = 3 * idx;
        float4 position = make_float4(points[pos], -points[pos + 1], points[pos + 2], 1);
        //int2 screen_coords;
        //float depth;
        //if (!mapScreenCooords(position, screen_coords, w, h, camera, &depth))return;
        auto rez = camera.mapToScreenCoords(position);
        int2 screen_coords = rez.coords;
        float depth = rez.depth;
        if (!rez.valid)return;
        float*max_depth = &depth_aux[screen_coords.x + screen_coords.y * w];
        //Casts are used, but know that the comparison is somehow the same as if they were both left as floats, except maybe when negative or NaN, which they shouldn't be.
        atomicMin((int*)max_depth, *(int*)(float*)&depth);
    }
}

template <typename CAMERA_DATA_TYPE>
void __global__ determineDepth_v2(float* depth_aux, const int h, const int w, const float* points, const int ndim, const int num_points, const CAMERA_DATA_TYPE camera) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_points) {
        // X,Y,Z, (ndim 'colors')
        int pos = (3 + ndim) * idx;
        float4 position = make_float4(points[pos], -points[pos + 1], points[pos + 2], 1);
        //int2 screen_coords;
        //float depth;
        //if (!mapScreenCooords(position, screen_coords, w, h, camera, &depth))return;
        auto rez = camera.mapToScreenCoords(position);
        int2 screen_coords = rez.coords;
        float depth = rez.depth;
        if (!rez.valid)return;
        float* max_depth = &depth_aux[screen_coords.x + screen_coords.y * w];
        //Casts are used, but know that the comparison is somehow the same as if they were both left as floats, except maybe when negative or NaN, which they shouldn't be.
        atomicMin((int*)max_depth, *(int*)(float*)&depth);
    }
}
static_assert(sizeof(int) == sizeof(float), "Here so the atomicMin hack above works properly. If this does not work, some other suitable replacement for it should be found, potentially using atomicCAS");

/*
* Per-pixel process
*/
template <typename CAMERA_TYPE>
void __global__ drawBackgroundandBundleDepth_v2(float* depth_aux, float* gpu_weighted, const int h, const int w, CAMERA_TYPE camera, float* environment_data, int resolution, int ndim) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        int ids = idy + idx * w;
        if (depth_aux[ids] > camera.far_clip) {
            float* env_data = sample_environment_data_v2(environment_data, resolution,/*direction = */camera.direction_for_pixel(make_float2(idy+0.5f, idx+0.5f)), ndim);
            depth_aux[ids] = -1;// camera.far_clip;//+env_data[ndim];//env_data[ndim - 1]
            for (int i = 0; i < ndim + 1; ++i) {
                gpu_weighted[(ndim+1) * ids + i] = env_data[i];
            }
            //this represents depth and was separated for no good reason.
            //gpu_weighted[(ndim + 1) * ids + ndim] = env_data[ndim];
        } else {
            // modified bundleDepthWithColor_v2
            int ids_m = ids * (ndim + 1);
            const float* input_position = &gpu_weighted[ids_m];
            float* output_position = &gpu_weighted[ids_m];
            float depth_local = depth_aux[ids];
            float weight = input_position[ndim];
            depth_aux[ids] = weight;
            for (int i = 0; i < ndim; ++i)
                output_position[i] = input_position[i] / weight;
            output_position[ndim] = depth_local;
        }
    }
}

//Todo? make a single thread do more than 1 pixel to reduce overhead.
void __global__ clear(cudaSurfaceObject_t output, float* depth_aux, int h, int w, uchar4 clear_color, float far_plane = FLT_MAX) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        surf2Dwrite(clear_color, output, idy * sizeof(uchar4), idx, cudaBoundaryModeClamp);
        depth_aux[idy + idx * w] = far_plane;
    }
}

//NOTE: replaceable with cudaMemSet calls?
void __global__ clear_v2(float* buffer, float* depth_aux, int h, int w, int ndim, float clear_color, float far_plane = FLT_MAX) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        for (int i = 0; i < ndim + 1; ++i) {
            //ndim 'colors' + an alpha channel
            buffer[(idy + idx * w) * (ndim + 1) + i] = clear_color;
        }
        depth_aux[idy + idx * w] = far_plane;//todo: macros
    }
}



template <typename CAMERA_TYPE>
inline cudaError_t plotPointsToFloatBufferCameraType_v2(const CAMERA_TYPE& camera, const int h, const int w, const int ndim,
    const void* points_memory, int num_points, const void* environment_memory, int environment_resolution,
    float* gpu_weights, float* gpu_weighted_color, bool needs_clear = true
) {
    cudaError_t cudaStatus = cudaSuccess;
    if (needs_clear) {
        cudaStatus = cudaMemset(gpu_weighted_color, 0, sizeof(float) * (ndim + 1ULL) * h * w);//Note: 0x00000000 in float means 0
        STATUS_CHECK();
        cudaStatus = cudaMemset(gpu_weights, 0x7F, sizeof(float) * h * w);//Note: 0x7F7F7F7F in float is 3.39615e+38, which is almost FLOAT_MAX (3.402823466e+38), so camera.far_clip should be way below it.
        STATUS_CHECK();
    }

    //NOTE: we use the weight memory to store the depth temporarily here.
    determineDepth_v2 BEST_LINEAR_KERNEL(num_points) (gpu_weights, h, w, (float*)points_memory, ndim, num_points, camera);
    AFTER_FUNCTION_CALL_CHECK();
    plotKernel_v2 BEST_LINEAR_KERNEL(num_points) (gpu_weighted_color, gpu_weights, h, w, (float*)points_memory, ndim, num_points, camera);
    AFTER_FUNCTION_CALL_CHECK();

    if (environment_memory != nullptr) {
        drawBackgroundandBundleDepth_v2 BEST_2D_KERNEL(h, w) (gpu_weights, gpu_weighted_color, h, w, camera, (float*)environment_memory, environment_resolution, ndim);
        AFTER_FUNCTION_CALL_CHECK();
    }
    else {
        bundleDepthWithColor_v2 BEST_2D_KERNEL(h, w) (gpu_weighted_color, gpu_weighted_color, gpu_weights, h, w, ndim);
        AFTER_FUNCTION_CALL_CHECK();
    }

Error:
    return cudaStatus;
}

/*
* Buffers are expeted to be cleared.
*/
cudaError_t plotPointsToFloatBuffer_v2(const CameraDataItf& camera, int h, int w, int ndim,
    const void* points_memory, int num_points,
    const void* environment_memory, int environment_resolution,
    float* gpu_color, float* gpu_weights, bool needs_clear = true) {
    cudaError_t cudaStatus = cudaSuccess;

    switch (camera.type())
    {
    case INTERACTIVE: {
        auto& data = (InteractiveCameraData&)camera;
        cudaStatus = plotPointsToFloatBufferCameraType_v2(data.prepareForGPU(h, w), h, w, ndim, points_memory, num_points, environment_memory, environment_resolution, gpu_weights, gpu_color, needs_clear);
        STATUS_CHECK();
        break;
    };
    case PINHOLE_PROJECTION: {
        auto& data = (PinholeCameraData&)camera;
        cudaStatus = plotPointsToFloatBufferCameraType_v2(data.prepareForGPU(h, w), h, w, ndim, points_memory, num_points, environment_memory, environment_resolution, gpu_weights, gpu_color, needs_clear);
        STATUS_CHECK();
        break;
    }
    case RADIAL: {
        auto& data = (RadialCameraData&)camera;
        cudaStatus = plotPointsToFloatBufferCameraType_v2(data.prepareForGPU(h, w), h, w, ndim, points_memory, num_points, environment_memory, environment_resolution, gpu_weights, gpu_color, needs_clear);
        STATUS_CHECK();
        break;
    };
    default:
        break;
    }

Error:
    return cudaStatus;
}

cudaError_t plotPointsToGPUMemory_v2(const std::shared_ptr<CameraDataItf> camera, int ndim,
    const void* point_memory, int num_points,
    const void* environment_memory, int environment_resolution,
    float** memory_color, bool is_preallocated,
    float** memory_weights, bool is_weight_preallocated,
    bool needs_clear)
{
    cudaError_t cudaStatus = cudaSuccess;
    float* gpu_depth_aux = NULL;
    float* gpu_weighted_color = NULL;
    int w = camera->get_width();
    int h = camera->get_height();

    if (!is_preallocated) {
        cudaStatus = cudaMalloc(&gpu_weighted_color, h * w * sizeof(float) * (ndim + 1));//weight/depth is not counted by ndim...
        STATUS_CHECK();
    }
    else gpu_weighted_color = *memory_color;

    if (memory_weights == NULL || !is_weight_preallocated) {
        cudaStatus = cudaMalloc(&gpu_depth_aux, h * w * sizeof(float));
        STATUS_CHECK();
    }
    else gpu_depth_aux = *memory_weights;

    cudaStatus = plotPointsToFloatBuffer_v2(*camera,h,w,ndim,
        point_memory,num_points,environment_memory, environment_resolution,
        gpu_weighted_color, gpu_depth_aux, needs_clear
    );
    STATUS_CHECK();
    if (!is_preallocated) {
        (*memory_color) = gpu_weighted_color;
    }

    if (memory_weights == NULL) {
        cudaStatus = cudaFree(gpu_depth_aux); gpu_depth_aux = NULL;
    }
    else if (!is_weight_preallocated) {
        (*memory_weights) = gpu_depth_aux;
    }

    STATUS_CHECK();
    ENSURE_SYNC();//Note: all commands are essentially given to CUDA stream 1, which means that we only need to check if they're done here.
    //That's it.
    return cudaStatus;

Error:
    if (!is_preallocated && gpu_weighted_color!=NULL)
        cudaFree(gpu_weighted_color);
    if (!is_weight_preallocated && gpu_depth_aux != NULL)
        cudaFree(gpu_depth_aux);
    return cudaStatus;
}


cudaError_t bytesToSubView(const void*& memory, const int sh, const int sw, const int h, const int w, Renderer& renderer, Renderer::ViewType viewType)
{
    cudaError_t cudaStatus = cudaSuccess;
    cudaGraphicsResource_t vbo_res;
    auto& view = renderer.getView(viewType);
    int wv = view.width;
    int hv = view.height;
    if (h != hv || w != wv) {
        view.width = w;
        view.height = h;
        view.needs_update = true;
    }
    cudaArray* array = NULL;

    cudaStatus = cudaGraphicsGLRegisterImage(&vbo_res, view.textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsMapResources(1, &vbo_res, 0);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsSubResourceGetMappedArray(&array, vbo_res, 0, 0);
    STATUS_CHECK();
    cudaResourceDesc viewCudaArrayResourceDesc;
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    viewCudaArrayResourceDesc.res.array.array = array;
    cudaSurfaceObject_t surface;
    cudaStatus = cudaCreateSurfaceObject(&surface, &viewCudaArrayResourceDesc);
    STATUS_CHECK();

    translateKernel BEST_2D_KERNEL(h, w) (surface, (uchar4*)memory, h, w, sh, sw);
    AFTER_FUNCTION_CALL_CHECK();
    ENSURE_SYNC();

Error:
    cudaStatus = cudaDestroySurfaceObject(surface);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsUnmapResources(1, &vbo_res, 0);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsUnregisterResource(vbo_res);
    STATUS_CHECK();
    return cudaStatus;
}


cudaError_t bytesToView(const void*& memory, const int h, const int w, Renderer& renderer, Renderer::ViewType viewType)
{
    cudaError_t cudaStatus = cudaSuccess;
    cudaGraphicsResource_t vbo_res;
    auto& view = renderer.getView(viewType);
    int wv = view.width;
    int hv = view.height;
    if (h != hv || w != wv) {
        view.width = w;
        view.height = h;
        view.needs_update = true;
    }
    cudaArray* array = NULL;
    cudaStatus = cudaGraphicsGLRegisterImage(&vbo_res, view.textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsMapResources(1, &vbo_res, 0);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsSubResourceGetMappedArray(&array, vbo_res, 0, 0);
    STATUS_CHECK();
    cudaResourceDesc viewCudaArrayResourceDesc;
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    viewCudaArrayResourceDesc.res.array.array = array;
    cudaSurfaceObject_t surface;
    cudaStatus = cudaCreateSurfaceObject(&surface, &viewCudaArrayResourceDesc);
    STATUS_CHECK();

    translateKernel BEST_2D_KERNEL(h, w) (surface, (uchar4*)memory, h, w);
    AFTER_FUNCTION_CALL_CHECK();
    ENSURE_SYNC();

Error:
    cudaStatus = cudaDestroySurfaceObject(surface);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsUnmapResources(1, &vbo_res, 0);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsUnregisterResource(vbo_res);
    STATUS_CHECK();
    return cudaStatus;

}

cudaError_t to_CPU(void*& memory, int length)
{
    cudaError_t cudaStatus = cudaError::cudaSuccess;
    void* out;
    out = malloc(length);
#ifndef NO_STATUS_CHECK
    if (out == NULL)return cudaError::cudaErrorMemoryAllocation;
#endif // DEBUG
    cudaStatus = cudaMemcpy(out, memory,length,cudaMemcpyKind::cudaMemcpyDeviceToHost);
    STATUS_CHECK();
    cudaStatus = cudaFree(memory);
    STATUS_CHECK();//I have no clue if this check can even possibly ever fail
    memory = out;
    return cudaSuccess;
Error:
    free(out);
    return cudaStatus;
}

cudaError_t to_GPU(void*& memory, int length)
{
    cudaError_t cudaStatus = cudaError::cudaSuccess;
    void* out=NULL;
    cudaStatus = cudaMalloc(&out, length);
    STATUS_CHECK();
    cudaStatus = cudaMemcpy(out, memory, length, cudaMemcpyKind::cudaMemcpyHostToDevice);
    STATUS_CHECK();
    free(memory);
    memory = out;
    return cudaSuccess;
Error:
    if (out != NULL)cudaFree(out);
    return cudaStatus;
}
