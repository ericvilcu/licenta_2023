#include "CudaIncludesAndMacros.cuh"
#include "PlotPoints.cuh"
#include "cuda_debug_utils.cuh"

/*
* This takes in a rgbw value, where its true value should be  x.rgb/x.w
*/
float4 __device__ rgbaColorFromWeighted(float4 rgbw) {
    if (rgbw.w != 0)
        return make_float4(rgbw.x / rgbw.w, rgbw.y / rgbw.w, rgbw.z / rgbw.w, 1);
    else
        return make_float4(0, 0, 0, 1);
}
/*
* This takes in a rgbw value, where its true value should be  x.rgb/x.w, and sets the fourth component to depth
*/
float4 __device__ rgbaColorFromWeightedWithDepth(float4 rgbw, float depth) {
    if (rgbw.w != 0)
        return make_float4(rgbw.x / rgbw.w, rgbw.y / rgbw.w, rgbw.z / rgbw.w, depth);
    else
        return make_float4(0, 0, 0, -1);//Would infinity be a better value here?
}

//depth and such are here for debug purposes.
void __global__ translateKernel(cudaSurfaceObject_t output, const float4* weighted_colors, const int h, const int w) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        float4 color = weighted_colors[idy + idx * w];
        surf2Dwrite(toRGBA8(rgbaColorFromWeighted(color)), output, idy * sizeof(uchar4), idx, cudaBoundaryModeClamp);
    }
}
void __global__ translateKernel(cudaSurfaceObject_t output, const uchar4* weighted_colors, const int h, const int w) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        surf2Dwrite(weighted_colors[idy + idx * w], output, idy * sizeof(uchar4), idx, cudaBoundaryModeClamp);
    }
}
void __global__ translateKernel(uchar4* output, const float4* weighted_colors, const int h, const int w) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        float4 color = weighted_colors[idy + idx * w];
        output[idy + idx * w] = toRGBA8(rgbaColorFromWeighted(color));
    }
}
void __global__ bundleDepthWithColor(float4* output, const float4* weighted_colors, const float* depth, const int h, const int w) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        float4 color = weighted_colors[idy + idx * w];
        float depth_local = depth[idy + idx * w];
        output[idy + idx * w] = rgbaColorFromWeightedWithDepth(color, depth_local);
    }
}

/*
* This has slight blending.
*/
template <typename CAMERA_DATA_TYPE>
void __global__ plotKernel(float4* output, const float* depth_aux, const int h, const int w, const float* points, const unsigned char* colors, const int l, const CAMERA_DATA_TYPE camera, const float alpha = 0.001) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < l) {
        int pos = 3 * idx;
        float4 position = make_float4(points[pos], -points[pos + 1], points[pos + 2], 1);
        //int2 screen_coords;
        //auto dontcrashplease = camera.mapToScreenCoords(position);
        //float depth;
        //if (!mapScreenCooords(position, screen_coords, w, h, camera, &depth))return;
        //Todo: investigeate why this does not work.
        auto rez = camera.mapToScreenCoords(position);
        int2 screen_coords = rez.coords;
        float depth = rez.depth;
        if (!rez.valid)return;
        float4 color = toRGBA32f(make_uchar4(colors[pos], colors[pos + 1], colors[pos + 2], 255));

        float surface_depth = depth_aux[screen_coords.x + screen_coords.y * w];
        if (surface_depth * (1+alpha) < depth)return;
        float weight = 1;// color.w / (position.z - surface_depth + 0.01);
        float* pr = &output[screen_coords.x + screen_coords.y * w].x;
        float* pg = &output[screen_coords.x + screen_coords.y * w].y;
        float* pb = &output[screen_coords.x + screen_coords.y * w].z;
        float* pw = &output[screen_coords.x + screen_coords.y * w].w;
        atomicAdd(pw, weight);
        //float4 hsv_color_debug = fromHSV(make_float4(1e-2f * surface_depth, 1, color.x, 1));
        atomicAdd(pr, weight * color.x);
        atomicAdd(pg, weight * color.y);
        atomicAdd(pb, weight * color.z);
        //surf2Dwrite(fromHSV(make_float4(1e-2f* max_depth,1,color.x / 255.0f,1)), output, screen_coords.x * sizeof(uchar4), screen_coords.y, cudaBoundaryModeClamp);
    }
}

/*
* Utility function used to determine the depth for each pixel, such that obstructed pixels will not be drawn.
* Note: this ruins anything that has to do with transparency, and that should be considered and adjusted for.
* See deferred blending
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

//That is !!4!! instatiations of this function, at least with 2 current camera types and 2 environment types. One should consider making the environment float only.
template <typename CAMERA_TYPE,typename environment_type>
void __global__ drawBackground(float* depth_aux, float4* gpu_weighted, const int h, const int w, CAMERA_TYPE camera, environment_type*environment_data, int resolution) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        int ids = idy + idx * w;
        if (depth_aux[ids] > camera.far_clip) {
            //if this is not 0, I get NaN.
            float4 data = sample_environment_data(environment_data, resolution,/*direction = */camera.direction_for_pixel(make_int2(idx, idy)));
            gpu_weighted[ids] = make_float4(data.x, data.y, data.z, 1);
            depth_aux[ids] = -1;// camera.far_clip;//+data.w;//data.w
            //We need a way to tell whether something was actually plotted by the environment, so for now we leave data.w alone.
        }
    }

}
static_assert(sizeof(int) == sizeof(float), "Here so the atomicMin hack above works properly. If this does not work, some other suitable replacement for it should be found, potentially using atomicCAS");

//Todo? make a single thread do more than 1 pixel to reduce overhead.
void __global__ clear(cudaSurfaceObject_t output, float* depth_aux, int h, int w, uchar4 clear_color, float far_plane=FLT_MAX) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        surf2Dwrite(clear_color, output, idy * sizeof(uchar4), idx, cudaBoundaryModeClamp);
        depth_aux[idy + idx * w] = far_plane;//todo: macros
    }
}
void __global__ clear(float4* output, float* depth_aux, int h, int w, float4 clear_color, float far_plane = FLT_MAX) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        output[idy + idx * w] = clear_color;
        depth_aux[idy + idx * w] = far_plane;//todo: macros
    }
}
void __global__ clear(float* depth_aux, int h, int w, float far_plane = FLT_MAX) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < h && idy < w) {
        depth_aux[idy + idx * w] = far_plane;//todo: macros
    }
}

template <typename CAMERA_TYPE> 
inline cudaError_t plotPointsToFloatBufferCameraType(float* gpu_depth_aux, float4* gpu_weighted, const int h, const int w, const GPUPoints& points, const CAMERA_TYPE& camera) {
    cudaError_t cudaStatus = cudaSuccess;

    determineDepth BEST_LINEAR_KERNEL(points.num_entries) (gpu_depth_aux, h, w, (float*)points.position_memory_start, points.num_entries, camera);
    cudaStatus = cudaDeviceSynchronize();
    STATUS_CHECK();
    cudaStatus = cudaPeekAtLastError();
    STATUS_CHECK();

    plotKernel BEST_LINEAR_KERNEL(points.num_entries) (gpu_weighted, gpu_depth_aux, h, w, (float*)points.position_memory_start, (unsigned char*)points.color_memory_start, points.num_entries, camera);
    cudaStatus = cudaDeviceSynchronize();
    STATUS_CHECK();
    cudaStatus = cudaPeekAtLastError();
    STATUS_CHECK();

    if (points.environment != nullptr) {
        if (points.environment->data_is_bytes) {
            drawBackground BEST_2D_KERNEL(h, w) (gpu_depth_aux, gpu_weighted, h, w, camera, (uchar4*)points.environment->data, points.environment->resolution);
            cudaStatus = cudaDeviceSynchronize();
            STATUS_CHECK();
            cudaStatus = cudaPeekAtLastError();
            STATUS_CHECK();
        } else {
            drawBackground BEST_2D_KERNEL(h, w) (gpu_depth_aux, gpu_weighted, h, w, camera, (float4*)points.environment->data, points.environment->resolution);
            cudaStatus = cudaDeviceSynchronize();
            STATUS_CHECK();
            cudaStatus = cudaPeekAtLastError();
            STATUS_CHECK();
        }
    }

Error:
    return cudaStatus;
}


/*
* Things are expeted to be cleared.
*/
cudaError_t plotPointsToFloatBuffer(float* gpu_depth_aux,float4* gpu_weighted, const int h, const int w, const GPUPoints& points, const CameraDataItf& camera) {
    cudaError_t cudaStatus = cudaSuccess;

    switch (camera.type())
    {
    case GL: {
        auto& data = (CameraGLData&)camera;
        cudaStatus = plotPointsToFloatBufferCameraType(gpu_depth_aux, gpu_weighted, h, w, points, data.prepareForGPU(h, w));
        STATUS_CHECK();
        break;
    };
    case PINHOLE_PROJECTION: {
        auto& data = (PinholeCameraData&)camera;
        cudaStatus = plotPointsToFloatBufferCameraType(gpu_depth_aux, gpu_weighted, h, w, points, data.prepareForGPU(h, w));
        STATUS_CHECK();
        break;
    }
    default:
        break;
    }

Error:
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
    cudaStatus = cudaDeviceSynchronize();
    STATUS_CHECK();
    cudaStatus = cudaPeekAtLastError();
    STATUS_CHECK();

Error:
    cudaStatus = cudaDestroySurfaceObject(surface);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsUnmapResources(1, &vbo_res, 0);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsUnregisterResource(vbo_res);
    STATUS_CHECK();
    return cudaStatus;

}

//I really want to know how this could be abstracted better.
void modifyIfNeccesary(int& h, int& w, const CameraDataItf& camera) {
    if (h <= 0 || w <= 0) {
        if (h <= 0 && w <= 0) {
            w = camera.get_width();
            h = camera.get_height();
        } else {
            if (h <= 0) {
                h = camera.get_height() * w / camera.get_width();
            } else {
                w = camera.get_width() * h / camera.get_height();
            }
        }
    }
}

cudaError_t plotPointsToGPUMemory(void*& memory, int h, int w, const GPUPoints& points, const CameraDataItf& camera, bool is_preallocated, bool as_bytes)
{
    cudaError_t cudaStatus = cudaSuccess;
    float* gpu_depth_aux = NULL;
    float4* gpu_weighted_color = NULL;
    uchar4* gpu_byte_color = NULL;
    modifyIfNeccesary(h, w, camera);
    cudaStatus = cudaMalloc(&gpu_depth_aux, h * w * sizeof(float));
    STATUS_CHECK();
    if (!is_preallocated || as_bytes) {
        cudaStatus = cudaMalloc(&gpu_weighted_color, h * w * sizeof(float4));
        STATUS_CHECK();
    }
    else gpu_weighted_color = (float4*)memory;

    clear BEST_2D_KERNEL(h, w) (gpu_weighted_color, gpu_depth_aux, h, w, make_float4(0, 0, 0, 0));
    cudaStatus = cudaDeviceSynchronize();
    STATUS_CHECK();
    cudaStatus = cudaPeekAtLastError();
    STATUS_CHECK();

    cudaStatus = plotPointsToFloatBuffer(gpu_depth_aux, gpu_weighted_color, h, w, points, camera);

    STATUS_CHECK();
    if (!as_bytes) {
        bundleDepthWithColor BEST_2D_KERNEL(h, w) (gpu_weighted_color, gpu_weighted_color, gpu_depth_aux, h, w);
        cudaStatus = cudaDeviceSynchronize();
        STATUS_CHECK();
        cudaStatus = cudaPeekAtLastError();
        STATUS_CHECK();
        memory = gpu_weighted_color;
        cudaStatus = cudaFree(gpu_depth_aux);
        //That's it.
        return cudaStatus;
    }
    if (!is_preallocated) {
        cudaStatus = cudaMalloc(&gpu_byte_color, h * w * sizeof(uchar4));
        STATUS_CHECK();
        memory = gpu_byte_color;
    }
    else gpu_byte_color = (uchar4*)memory;
    translateKernel BEST_2D_KERNEL(h,w) (gpu_byte_color, gpu_weighted_color, h, w);
    cudaStatus = cudaDeviceSynchronize();
    STATUS_CHECK();
    cudaStatus = cudaPeekAtLastError();
    STATUS_CHECK();

Error:
    if (!is_preallocated || !as_bytes)
        cudaFree(gpu_weighted_color);
    cudaFree(gpu_depth_aux);
    return cudaStatus;
}

cudaError_t plotPoints(const cudaSurfaceObject_t output, const int h, const int w, const GPUPoints& points, const CameraDataItf& camera)
{
    cudaError_t cudaStatus = cudaSuccess;
    float* gpu_depth_aux=NULL;
    float4* gpu_weighted_color = NULL;
    cudaStatus = cudaMalloc(&gpu_depth_aux, h * w * sizeof(float));
    STATUS_CHECK();
    cudaStatus = cudaMalloc(&gpu_weighted_color, h * w * sizeof(float4));
    STATUS_CHECK();

    clear BEST_2D_KERNEL(h, w) (output, gpu_depth_aux, h, w, make_uchar4(0, 0, 0, 255));
    cudaStatus = cudaDeviceSynchronize();
    STATUS_CHECK();
    cudaStatus = cudaPeekAtLastError();
    STATUS_CHECK();

    cudaStatus = plotPointsToFloatBuffer(gpu_depth_aux, gpu_weighted_color, h, w, points, camera);
    STATUS_CHECK();

    translateKernel BEST_2D_KERNEL(h, w) (output, gpu_weighted_color, h, w);
    cudaStatus = cudaDeviceSynchronize();
    STATUS_CHECK();
    cudaStatus = cudaPeekAtLastError();
    STATUS_CHECK();

    //This is how they do it in the examples, which is really just a finally block.
Error:
    cudaFree(gpu_depth_aux);
    cudaFree(gpu_weighted_color);
	return cudaStatus;
}

cudaError_t plotPointsToRenderer(const Renderer& renderer, const GPUPoints& points, const CameraDataItf& camera, const Renderer::ViewType viewType)
{
    int w, h;
    cudaError_t cudaStatus = cudaSuccess;
    cudaGraphicsResource_t vbo_res;
    const auto& view = renderer.getView(viewType);
    w = view.width;
    h = view.height;
    cudaStatus = cudaGraphicsGLRegisterImage(&vbo_res, view.textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    STATUS_CHECK();
    cudaArray* array;
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

    cudaStatus = plotPoints(surface, h, w, points, camera);
    STATUS_CHECK();

    cudaStatus = cudaDestroySurfaceObject(surface);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsUnmapResources(1, &vbo_res, 0);
    STATUS_CHECK();
    cudaStatus = cudaGraphicsUnregisterResource(vbo_res);
    STATUS_CHECK();
Error:
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
    STATUS_CHECK();
    memory = out;
    return cudaSuccess;
Error:
    free(out);
    return cudaStatus;
}

