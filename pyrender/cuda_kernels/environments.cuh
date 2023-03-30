__hdfi__ float mag(float3 data){
    return sqrt(data.x*data.x+data.y*data.y+data.z*data.z);
}

__hdfi__ float3 norm(float3 data){
    float n=mag(data);
    return make_float3(data.x/n,data.y/n,data.z/n);
}

#ifndef NDIM
#define NDIM 3
#endif

#ifndef ENVIRONMENT_TYPE
#define ENVIRONMENT_TYPE -1
#endif

#if ENVIRONMENT_TYPE==-1
__hdfi__ void sample_environment(float* out,const float* environment_data,float3 direction){
    for (int i = 0; i <= NDIM; ++i)
        out[i] = 0;
}

__hdfi__ void backward_environment(const float* pixel_grad, const float* environment_data, float* environment_grad,float3 direction){
}
#endif

#if ENVIRONMENT_TYPE==0
//Environment type 0 is meant to be the kind of environment game engines provide.
//It should have the following parameters:
constexpr int GROUND1_IDX=0;
// 1. ground color (close)    (NDIM+1) [1->NDIM+1]
constexpr int GROUND2_IDX=GROUND1_IDX+NDIM+1;
// 2. ground color (horizon)  (NDIM+1) [NDIM+2->2*(NDIM+1)]
constexpr int GROUND_BLEND_IDX=GROUND2_IDX+NDIM+1;
// 3. ground horizon blending (1)      [2*(NDIM+1)+1]
constexpr int SKY1_IDX=GROUND_BLEND_IDX+1;
// 4. sky color (close)       (NDIM+1)
constexpr int SKY2_IDX=SKY1_IDX+NDIM+1;
// 5. sky color (horizon)     (NDIM+1)
constexpr int SKY_BLEND_IDX=SKY2_IDX+NDIM+1;
// 6. sky horizon blending    (1)
constexpr int SUN_POS_IDX=SKY_BLEND_IDX+1;
// 7. sun position            (2)
constexpr int SUN_COLOR_IDX=SUN_POS_IDX+2;
// 8. sun color               (NDIM+1)
constexpr int SUN_RADIUS_IDX=SUN_COLOR_IDX+NDIM+1;
// 9. sun radius              (1)
constexpr int SUN_SKY_COLOR_IDX=SUN_RADIUS_IDX+1;
//10. sun sky contribution    (NDIM+1)
constexpr int SUN_SKY_RADIUS_IDX=SUN_SKY_COLOR_IDX+NDIM+1;
//11. sun sky radius          (1)
constexpr int SUN_SKY_BLEND_IDX=SUN_SKY_RADIUS_IDX+1;
//12. sun sky blend           (1)
//TOTAL=6*NDIM+13
//All of these I should be able to apply gradient descent to, as well.
__hdfi__ void sample_environment(float* out,const float* environment_data,float3 direction){
    const float* data=environment_data;
    direction=norm(direction);
    float y=direction.y;
    //TODO
    if(y<=0){
        //sampling ground
        float qd=y*y;
        float c=pow(qd,data[GROUND_BLEND_IDX]);
        float ic=1-c;
        for(int i=0;i<=NDIM;++i)
            out[i]=data[GROUND1_IDX+i]*c+data[GROUND2_IDX]*ic;
    } else{
        //sampling sky
        float qd=y*y;
        float c=pow(qd,data[SKY_BLEND_IDX]);
        float ic=1-c;
        for(int i=0;i<=NDIM;++i)
            out[i]=data[SKY1_IDX+i]*c+data[SKY2_IDX]*ic;
        //todo: sun + sun contribution
    }
}
__hdfi__ void backward_environment(const float* pixel_grad, const float* environment_data, float* environment_grad,float3 direction){
    direction=norm(direction);
    
    float y=direction.y;

    if(y<=0){
        //sampling ground
        float qd=y*y;
        float c=pow(qd,environment_data[GROUND_BLEND_IDX]);
        float ic=1-c;
        float c_grad=0;
        for(int i=0;i<=NDIM;++i){
            atomicAdd(&environment_grad[GROUND1_IDX+i],pixel_grad[i]*c);
            atomicAdd(&environment_grad[GROUND2_IDX+i],pixel_grad[i]*ic);//TODO: re-check math
            c_grad+=(environment_data[GROUND1_IDX+i]-environment_data[GROUND2_IDX+i])*pixel_grad[i];
        }
        float exponent_grad=c * logf(qd) * c_grad;
        atomicAdd(&environment_grad[GROUND_BLEND_IDX],exponent_grad);
    } else{
        //sampling sky
        float qd=y*y;
        float c=pow(qd,environment_data[SKY_BLEND_IDX]);
        float ic=1-c;
        float c_grad=0;
        for(int i=0;i<=NDIM;++i){
            atomicAdd(&environment_grad[SKY1_IDX+i],pixel_grad[i]*c);
            atomicAdd(&environment_grad[SKY2_IDX+i],pixel_grad[i]*ic);//TODO: re-check math
            c_grad+=(environment_data[SKY1_IDX+i]-environment_data[SKY2_IDX+i])*pixel_grad[i];
        }
        float exponent_grad=c * logf(qd) * c_grad;
        atomicAdd(&environment_grad[SKY_BLEND_IDX],exponent_grad);
        //todo: sun + sun contribution
    }
}
#endif

#if ENVIRONMENT_TYPE==1
#ifndef ENVIRONMENT_RESOLUTION
#define ENVIRONMENT_RESOLUTION 1024
#endif


#endif

