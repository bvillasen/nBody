#include "vector3D.hT"


extern "C"{
  
__global__ void main_kernel( const int nParticles, const cudaP Gmass,
			     cudaP *posX, cudaP *posY, cudaP *posZ, 
			     cudaP *velX, cudaP *velY, cudaP *velZ,
			     cudaP *accelX, cudaP *accelY, cudaP *accelZ,
			     cudaP dt, cudaP epsilon, float *cuda_VOB){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  
  if (tid < nParticles){
    
    //Initialize particle position and velocity
    Vector3D pos( posX[tid], posY[tid], posZ[tid] );
    Vector3D vel( velX[tid], velY[tid], velZ[tid] );
    
    
    //Initialize shared array for positions of other particles
    __shared__ cudaP posX_sh[ %(TPB)s ];
    __shared__ cudaP posY_sh[ %(TPB)s ];
    __shared__ cudaP posZ_sh[ %(TPB)s ];
    
    
    Vector3D posOther, deltaPos;
    Vector3D force( 0., 0., 0. );
    cudaP dist;
    for ( int blockNumber=0; blockNumber<gridDim.x; blockNumber++ ){
      posX_sh[ threadIdx.x ] = posX[ blockNumber*blockDim.x + threadIdx.x];
      posY_sh[ threadIdx.x ] = posY[ blockNumber*blockDim.x + threadIdx.x];
      posZ_sh[ threadIdx.x ] = posZ[ blockNumber*blockDim.x + threadIdx.x];
      __syncthreads();
      
      for ( int otherParticle=0; otherParticle<blockDim.x; otherParticle++ ){
	posOther.redefine( posX_sh[ otherParticle], posY_sh[ otherParticle], posZ_sh[ otherParticle] );
	deltaPos = posOther - pos;
	dist = sqrt( deltaPos.norm2() + epsilon );
	force += deltaPos*(Gmass/( dist*dist*dist ) );
      }
    }


    velX[tid] = vel.x + 0.5*dt*( accelX[tid] + force.x );
    velY[tid] = vel.y + 0.5*dt*( accelY[tid] + force.y );
    velZ[tid] = vel.z + 0.5*dt*( accelZ[tid] + force.z );
    accelX[tid] = force.x;
    accelY[tid] = force.y;
    accelZ[tid] = force.z;
     
    //Save data in animation buffer
//     if (usingAnimation ){
      cuda_VOB[3*tid + 0] = float(pos.x);
      cuda_VOB[3*tid + 1] = float(pos.y);
      cuda_VOB[3*tid + 2] = float(pos.z);
//     }

  }
}
  
}//Extern C end
  