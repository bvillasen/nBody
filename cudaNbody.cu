#include "vector3D.hT"


__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

extern "C"{
  
__global__ void main_kernel( const int nParticles, const cudaP Gmass, const int usingAnimation,
			     cudaP *posX, cudaP *posY, cudaP *posZ, 
			     cudaP *velX, cudaP *velY, cudaP *velZ,
			     cudaP *accelX, cudaP *accelY, cudaP *accelZ,
			     cudaP dt, cudaP epsilon, float *cuda_VOB){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  
  if (tid < nParticles){
    //Initialize shared array for positions of other particles
    __shared__ cudaP posX_sh[ %(TPB)s ];
    __shared__ cudaP posY_sh[ %(TPB)s ];
    __shared__ cudaP posZ_sh[ %(TPB)s ];
    __shared__ cudaP accelX_sh[ %(TPB)s ];
    __shared__ cudaP accelY_sh[ %(TPB)s ];
    __shared__ cudaP accelZ_sh[ %(TPB)s ];
    
    //Initialize particle position and velocity
    Vector3D pos( posX[tid], posY[tid], posZ[tid] );
    Vector3D vel( velX[tid], velY[tid], velZ[tid] );
    Vector3D force( 0., 0., 0. );
    Vector3D posOther, deltaPos;
    cudaP dist;
    int idOther;

    for ( int blockNumber=0; blockNumber<gridDim.x; blockNumber++ ){
      if (blockNumber < blockIdx.x ) continue;
      idOther =  blockNumber*blockDim.x + threadIdx.x;
      if (idOther < tid ) continue;
      posX_sh[ threadIdx.x ] = posX[ idOther ];
      posY_sh[ threadIdx.x ] = posY[ idOther ];
      posZ_sh[ threadIdx.x ] = posZ[ idOther ];
      accelX_sh[ threadIdx.x ] = 0;
      accelY_sh[ threadIdx.x ] = 0;
      accelZ_sh[ threadIdx.x ] = 0;
      __syncthreads();
      
      idOther = threadIdx.x;
      for ( int i=0; i<blockDim.x; i++ ){
	if (idOther >= blockDim.x ) idOther = 0;
	posOther.redefine( posX_sh[ idOther ], posY_sh[ idOther ], posZ_sh[ idOther ] );
	deltaPos = posOther - pos;
	dist = sqrt( deltaPos.norm2() + epsilon );
	deltaPos = deltaPos*(1./( dist*dist*dist ) );
	force += deltaPos;
	atomicAdd( &( accelX_sh[ threadIdx.x ] ), -deltaPos.x );
	atomicAdd( &( accelY_sh[ threadIdx.x ] ), -deltaPos.y );
	atomicAdd( &( accelZ_sh[ threadIdx.x ] ), -deltaPos.z );
	idOther++;
      }
      
      __syncthreads();
      idOther =  blockNumber*blockDim.x + threadIdx.x;
      atomicAdd( &( accelZ[ idOther ] ), accelX_sh[ threadIdx.x ] );
      atomicAdd( &( accelY[ idOther ] ), accelY_sh[ threadIdx.x ] );
      atomicAdd( &( accelZ[ idOther ] ), accelZ_sh[ threadIdx.x ] );
    }
    force *= Gmass;
    
    

    
    velX[tid] = vel.x + 0.5*dt*( accelX[tid] + force.x );
    velY[tid] = vel.y + 0.5*dt*( accelY[tid] + force.y );
    velZ[tid] = vel.z + 0.5*dt*( accelZ[tid] + force.z );
    accelX[tid] = force.x;
    accelY[tid] = force.y;
    accelZ[tid] = force.z;
     
    //Save data in animation buffer
    if (usingAnimation ){
      cuda_VOB[3*tid + 0] = float(pos.x);
      cuda_VOB[3*tid + 1] = float(pos.y);
      cuda_VOB[3*tid + 2] = float(pos.z);
    }

  }
}
  
}//Extern C end
  