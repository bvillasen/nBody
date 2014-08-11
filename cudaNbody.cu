#include "vector3D.hT"


__device__ void move( Vector3D &pos, Vector3D &vel, cudaP time ){ 
  Vector3D deltaPos = vel/time;
  pos = pos + deltaPos ;
}


extern "C"{
  
__global__ void main_kernel( const unsigned char usingAnimation, const unsigned char step, const int nParticles, 
			     cudaP *posX, cudaP *posY, cudaP *posZ, 
			     cudaP *velX, cudaP *velY, cudaP *velZ,
			     cudaP *accelX, cudaP *accelY, cudaP *accelZ,
			     cudaP dt, cudaP epsilon, float *cuda_VOB){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
//   int nThreads = blockDim.x * gridDim.x;
  
  if (tid < nParticles){
    
    //Initialize particle position and velocity
    Vector3D pos( posX[tid], posY[tid], posZ[tid] );
    Vector3D vel( velX[tid], velY[tid], velZ[tid] );
    
    
    //Initialize shared array for positions of other particles
    __shared__ cudaP posX_sh[ %(THREADS_PER_BLOCK)s ];
    __shared__ cudaP posY_sh[ %(THREADS_PER_BLOCK)s ];
    __shared__ cudaP posZ_sh[ %(THREADS_PER_BLOCK)s ];
    
    Vector3D posOther, deltaPos;
    Vector3D force( 0., 0., 0. );
    cudaP dist;
    for ( int blockNumber=0; blockNumber<gridDim.x; blockNumber++ ){
      posX_sh[ threadIdx.x ] = posX[ blockNumber*blockDim.x + threadIdx.x];
      posY_sh[ threadIdx.x ] = posY[ blockNumber*blockDim.x + threadIdx.x];
      posZ_sh[ threadIdx.x ] = posZ[ blockNumber*blockDim.x + threadIdx.x];
      __syncthreads();
      
      for ( int otherParticle=0; otherParticle<blockDim.x; otherParticle++ ){
// 	if (blockNumber = blockIdx.x and otherParticle == threadIdx.x) continue; //Force of the same parti
	posOther.redefine( posX_sh[ otherParticle], posY_sh[ otherParticle], posZ_sh[ otherParticle] );
	deltaPos = posOther - pos;
	dist = sqrt( deltaPos.norm2() + epsilon );
	deltaPos = deltaPos/(1./( dist*dist*dist ) );
	force =  force + deltaPos;
      }
//       __syncthreads();
    }
      

//     if (step==0){
//       outPosX[tid] = pos.x + dt*( vel.x + 0.5*dt*force.x );
//       outPosY[tid] = pos.y + dt*( vel.y + 0.5*dt*force.y );
//       outPosZ[tid] = pos.z + dt*( vel.z + 0.5*dt*force.z );
    velX[tid] = vel.x + 0.5*dt*( accelX[tid] + force.x );
    velY[tid] = vel.y + 0.5*dt*( accelY[tid] + force.y );
    velZ[tid] = vel.z + 0.5*dt*( accelZ[tid] + force.z );
    accelX[tid] = force.x;
    accelY[tid] = force.y;
    accelZ[tid] = force.z;
//     }
    
//     if (step==1){
//       velX[tid] = vel.x + 0.5*( accelX[tid] + force.x );
//       velY[tid] = vel.y + 0.5*( accelY[tid] + force.y );
//       velZ[tid] = vel.z + 0.5*( accelZ[tid] + force.z );
//     }
//       
    //Save data in animation buffer
    if (usingAnimation and step== 0){
      cuda_VOB[3*tid + 0] = float(pos.x);
      cuda_VOB[3*tid + 1] = float(pos.y);
      cuda_VOB[3*tid + 2] = float(pos.z);
    }

  }
}
  
}//Extern C end
  