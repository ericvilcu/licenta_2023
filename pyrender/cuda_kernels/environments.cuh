__hdfi__ float mag(float3 data){
    return sqrt(data.x*data.x+data.y*data.y+data.z*data.z);
}

__hdfi__ float3 norm(float3 data){
    float n=mag(data);
    return make_float3(data.x/n,data.y/n,data.z/n);
}

__hdfi__ float3 abs3(float3 data){
    return make_float3(abs(data.x),abs(data.y),abs(data.z));
}

__hdfi__ float3 cross(float3 a,float3 b){
    return make_float3((a.y*b.z-a.z*b.y),(a.z*b.x-a.x*b.z),(a.x*b.y-a.y*b.x));
}

__hdfi__ float dot(float3 a,float3 b){
    return a.x*b.x+a.y+b.y+a.z*b.z;
}

__hdfi__ float clampf(float x, float mn, float mx){
    return min(max(x,mn),mx);
}
__hdfi__ float clampui(unsigned int x, unsigned int mn, unsigned int mx){
    return min(max(x,mn),mx);
}

//this is here for intellisiense more than anytinhg
#ifndef NDIM
#define NDIM 3
#endif

#ifndef ENVIRONMENT_TYPE
#define ENVIRONMENT_TYPE -1
#endif

#if ENVIRONMENT_TYPE==-1
__hdfi__ void sample_environment(float* out,const float* environment_data,float3 direction){
    for (int i = 0; i <= NDIM; ++i)
        out[i] = -1;
}

__hdfi__ void backward_environment(const float* pixel_grad, const float* environment_data, float* environment_grad,float3 direction){
}
#endif

#if ENVIRONMENT_TYPE==0
//Environment type 0 is 1 solid color.
__hdfi__ void sample_environment(float* out,const float* environment_data,float3 direction){
    for (int i = 0; i <= NDIM; ++i)
        out[i] = environment_data[i];
}

__hdfi__ void backward_environment(const float* pixel_grad, const float* environment_data, float* environment_grad,float3 direction){
    for (int i = 0; i <= NDIM; ++i)
        atomicAdd(&environment_grad[i], pixel_grad[i]);
}
#endif

#if ENVIRONMENT_TYPE==1
//Environment type 1 is meant to be the kind of environment game engines provide.
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
        #pragma unroll
        for(int i=0;i<=NDIM;++i)
            out[i]=data[SKY1_IDX+i]*c+data[SKY2_IDX]*ic;
        //todo: backward sun + sun contribution

        float sun_radius=environment_data[SUN_RADIUS_IDX];
        float sun_x=environment_data[SUN_POS_IDX+0];
        float sun_z=environment_data[SUN_POS_IDX+1];
        float sun_y=1;
        float3 sun = norm(make_float3(sun_x,sun_y,sun_z));

        float sun_distance= abs(dot(direction,sun));
        //bad
        float sun_factor=min(sun_radius-sun_distance,1.0f);


        if(sun_factor>0)
            #pragma unroll
            for(int i=0;i<=NDIM;++i)
                out[i]+=environment_data[SUN_COLOR_IDX+i]*sun_factor;
        
        float contribution_radius=environment_data[SUN_SKY_RADIUS_IDX];
        float sun_contribution=min(1-sun_distance/contribution_radius,1.0f);

        if(sun_contribution>0){
            float real_contribution=pow(sun_contribution,environment_data[SUN_SKY_BLEND_IDX]);
            #pragma unroll
            for(int i=0;i<=NDIM;++i)
                out[i]+=environment_data[SUN_COLOR_IDX+i]*real_contribution;
        }
        
    }
}
__hdfi__ void backward_environment(const float* pixel_grad, const float* environment_data, float* environment_grad,float3 direction){
    direction=norm(direction);
    
    float y=direction.y;

    if(y<=0){//NOTE: I expect a lot of warps to fall under the same if, so threads keeping each other awake should be less of a problem than in other cases
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
        
        float sun_radius=environment_data[SUN_RADIUS_IDX];
        float sun_x=environment_data[SUN_POS_IDX+0];
        float sun_z=environment_data[SUN_POS_IDX+1];
        float sun_y=1;
        float3 sun = norm(make_float3(sun_x,sun_y,sun_z));

        float sun_distance= abs(dot(direction,sun));
        float sun_distance_grad=0;
        //bad
        float sun_factor=min(sun_radius-sun_distance,1.0f);


        if(sun_factor>0){
            float sun_factor_grad=0;
            #pragma unroll
            for(int i=0;i<=NDIM;++i){
                atomicAdd(&environment_grad[SUN_COLOR_IDX+i],pixel_grad[i]*sun_factor);
                sun_factor_grad+=pixel_grad[i]*environment_data[SUN_COLOR_IDX+i];
            }
            if(sun_radius-sun_distance < 1){
                float sun_radius_grad=sun_factor_grad;
                atomicAdd(&environment_grad[SUN_RADIUS_IDX],sun_radius_grad);
                
                sun_distance_grad+=sun_factor_grad;
            }
        }
        
        float contribution_radius=environment_data[SUN_SKY_RADIUS_IDX];
        float sun_contribution=min(contribution_radius-sun_distance,1.0f);

        //gradient calculations :|
        if(sun_contribution>0){
            float blend=environment_data[SUN_SKY_BLEND_IDX];
            float contribution_fraction=(1-sun_distance/contribution_radius);

            float real_contribution=powf(contribution_fraction,blend);
            float real_contribution_grad=0;
            #pragma unroll
            for(int i=0;i<=NDIM;++i){
                atomicAdd(&environment_grad[SUN_SKY_COLOR_IDX+i],pixel_grad[i]*real_contribution);
                real_contribution_grad+=pixel_grad[i]*environment_data[SUN_SKY_COLOR_IDX+i];
            }
            if(real_contribution < 1){
                float blend_grad=real_contribution * logf(contribution_fraction);
                atomicAdd(&environment_grad[SUN_SKY_BLEND_IDX],blend_grad);

                float grad_fraction=blend * powf(contribution_fraction,(blend-1));
                
                float contribution_radius_grad= -grad_fraction / sun_distance;
                if(contribution_radius_grad>0.1)contribution_radius_grad=0.1;//Needed to avoid some strange bs
                else if(contribution_radius_grad<-0.1)contribution_radius_grad=-0.1;//Needed to avoid some strange bs
                atomicAdd(&environment_grad[SUN_SKY_RADIUS_IDX],contribution_radius_grad);

                sun_distance_grad+= -grad_fraction*contribution_radius;
            }
        }


        if(sun_distance_grad!=0){
            if(direction.y>0.01){
                float projx=direction.x/direction.y;
                float projz=direction.z/direction.y;
                
                float2 direction_to_sun=make_float2(sun_x-projx,sun_z-projz);//direction to sun from our point
                //?NOTE: for 100% corectness, it should be divided by the positive value of the cosine between the direction and the sun, but this has goodenough results

                atomicAdd(&environment_grad[SUN_POS_IDX+0],sun_distance_grad*direction_to_sun.x);
                atomicAdd(&environment_grad[SUN_POS_IDX+1],sun_distance_grad*direction_to_sun.y);
            }
        }
    }
}
#endif

//todo: cubemap?
#ifndef ENVIRONMENT_RESOLUTION
#define ENVIRONMENT_RESOLUTION 1024
#endif

__hdfi__ uint3 cubemap_coords(float3 direction) {
	float3 abs_dir = abs3(direction);
	//rewrite of: https://www.gamedev.net/forums/topic/687535-implementing-a-cube-map-lookup-function/5337472/
	if (abs_dir.z >= abs_dir.y && abs_dir.z >= abs_dir.x) {
		//Positive/negative Z
		int face_idx = (direction.z > 0 ? 1 : 0);
		float D = 0.5f / abs_dir.z;
		uint2 uv = make_uint2((unsigned int)clampf((-direction.x*D+0.5f)*ENVIRONMENT_RESOLUTION,0.0f,(float)(ENVIRONMENT_RESOLUTION-1)),
							  (unsigned int)clampf((-direction.y*D+0.5f)*ENVIRONMENT_RESOLUTION,0.0f,(float)(ENVIRONMENT_RESOLUTION-1)));
		return make_uint3(face_idx, uv.x, uv.y);
	}
	else if (abs_dir.y >= abs_dir.x) {
		//Positive/negative y
		int face_idx = (direction.y > 0 ? 3 : 2);
		float D = 0.5f / abs_dir.y;
		uint2 uv = make_uint2((unsigned int)clampf((-direction.x*D+0.5f)*ENVIRONMENT_RESOLUTION,0.0f,(float)(ENVIRONMENT_RESOLUTION-1)),
							  (unsigned int)clampf((-direction.z*D+0.5f)*ENVIRONMENT_RESOLUTION,0.0f,(float)(ENVIRONMENT_RESOLUTION-1)));
		return make_uint3(face_idx, uv.x, uv.y);
	}
	else {
		//Positive/negative x
		int face_idx = (direction.x > 0 ? 5 : 4);
		float D = 0.5f / abs_dir.x;
		uint2 uv = make_uint2((unsigned int)clampf((-direction.z*D+0.5f)*ENVIRONMENT_RESOLUTION,0.0f,(float)(ENVIRONMENT_RESOLUTION-1)),
							  (unsigned int)clampf((-direction.y*D+0.5f)*ENVIRONMENT_RESOLUTION,0.0f,(float)(ENVIRONMENT_RESOLUTION-1)));
		return make_uint3(face_idx, uv.x, uv.y);
	}
}
__hdfi__ unsigned int pixel_from_cubemap_coords(uint3 data) {
	//clamps may be unnecesary
	return (NDIM+1)*(unsigned int)(data.x * ENVIRONMENT_RESOLUTION * ENVIRONMENT_RESOLUTION + clampui(data.y, 0, ENVIRONMENT_RESOLUTION - 1) + clampui(data.z, 0, ENVIRONMENT_RESOLUTION - 1) * ENVIRONMENT_RESOLUTION);
}


__hdfi__ void sample_environment(float* out,const float* environment_data,float3 direction){
    direction=norm(direction);
    int idx_s = pixel_from_cubemap_coords(cubemap_coords(direction));
    #pragma unroll
    for(int i=0;i<=NDIM;++i)
        out[i]=environment_data[idx_s+i];
}


__hdfi__ void backward_environment(const float* pixel_grad, const float* environment_data, float* environment_grad,float3 direction){
    direction=norm(direction);
    int idx_s = pixel_from_cubemap_coords(cubemap_coords(direction));
    #pragma unroll
    for(int i=0;i<=NDIM;++i)
        atomicAdd(&environment_grad[idx_s+i],pixel_grad[i]);
}

#endif

