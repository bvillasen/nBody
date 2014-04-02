import numpy as np
import sys, time, os, inspect, datetime
#import h5py as h5
import matplotlib.pyplot as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel


currentDirectory = os.getcwd()
#Add Modules from other directories
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
sys.path.append( toolsDirectory )
from cudaTools import setCudaDevice, getFreeMemory
from tools import printProgress


nParticles = 1024*8
dt = 10.
epsilon = 1e-6

cudaP = "float"
devN = None
usingAnimation = True


#Read in-line parameters
for option in sys.argv:
  if option.find("part")>=0 : nParticles = int(option[option.find("=")+1:])
  if option.find("anim")>=0: usingAnimation = True
  if option == "double": cudaP = "double"
  if option == "float": cudaP = "float"
  if option.find("dev") >= 0 : devN = int(option[-1])


if usingAnimation: import points3D as pAnim #points3D Animation
precision  = {"float":np.float32, "double":np.float64} 
cudaPre = precision[cudaP]

#Set CUDA thread grid dimentions
block = ( 128, 1, 1 )
grid = ( (nParticles - 1)//block[0] + 1, 1, 1 )


if usingAnimation:
  pAnim.nPoints = nParticles
  pAnim.viewXmin, pAnim.viewXmax = -10000., 10000.
  pAnim.viewYmin, pAnim.viewYmax = -10000., 10000.
  pAnim.viewZmin, pAnim.viewZmax = -10000000., 10000000. 
  pAnim.showGrid = False
  pAnim.windowTitle = " CUDA N-body simulation     particles={0}".format(nParticles) 

###########################################################################
###########################################################################
#Initialize and select CUDA device
if usingAnimation:
  pAnim.initGL()
  pAnim.CUDA_initialized = True
  #configAnimation()
cudaDev = setCudaDevice( devN = devN, usingAnimation = usingAnimation )

#Read and compile CUDA code
print "\nCompiling CUDA code\n"
codeFiles = [ "vector3D.h", "cudaNbody.cu"]
for fileName in codeFiles:
  codeString = open(fileName, "r").read().replace("cudaP", cudaP)
  outFile = open( fileName + "T", "w" )
  outFile.write( codeString )
  outFile.close()
cudaCodeStringTemp = open("cudaNbody.cuT", "r").read()
cudaCodeString = cudaCodeStringTemp % { "THREADS_PER_BLOCK":block[0] }
cudaCode = SourceModule(cudaCodeString, no_extern_c=True, include_dirs=[currentDirectory])
mainKernel = cudaCode.get_function("main_kernel" )
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
moveParticles = ElementwiseKernel(arguments="float dt, float *posX, float *posY, float *posZ,\
				     float *velX, float *velY, float *velZ,\
				     float *accelX, float *accelY, float *accelZ",
			      operation = "posX[i] = posX[i] + dt*( velX[i] + 0.5f*dt*accelX[i]);\
				  posY[i] = posY[i] + dt*( velY[i] + 0.5f*dt*accelY[i]);\
				  posZ[i] = posZ[i] + dt*( velZ[i] + 0.5f*dt*accelZ[i]);",
			      name ="moveParticles")
########################################################################
#Initialize all gpu data
print "Initializing Data"
initialMemory = getFreeMemory( show=True )  
posX_h = 5000.*np.random.random( nParticles ).astype(cudaPre) - 2500.
posY_h = 5000.*np.random.random( nParticles ).astype(cudaPre) - 2500.
posZ_h = 5000.*np.random.random( nParticles ).astype(cudaPre) - 2500.
velX_h = 1.*np.random.random( nParticles ).astype(cudaPre) - 0.5
velY_h = 1.*np.random.random( nParticles ).astype(cudaPre) - 0.5
velZ_h = 1.*np.random.random( nParticles ).astype(cudaPre) - 0.5
posX_d = gpuarray.to_gpu( posX_h )
posY_d = gpuarray.to_gpu( posY_h )
posZ_d = gpuarray.to_gpu( posZ_h )
velX_d = gpuarray.to_gpu( velX_h )
velY_d = gpuarray.to_gpu( velY_h )
velZ_d = gpuarray.to_gpu( velZ_h )
#outPosX_d = gpuarray.to_gpu( posX_h  )
#outPosY_d = gpuarray.to_gpu( posY_h  )
#outPosZ_d = gpuarray.to_gpu( posZ_h  )
accelX_d = gpuarray.to_gpu( np.zeros( nParticles, dtype=cudaPre ) )
accelY_d = gpuarray.to_gpu( np.zeros( nParticles, dtype=cudaPre ) )
accelZ_d = gpuarray.to_gpu( np.zeros( nParticles, dtype=cudaPre ) )
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6) 


nAnimIter = 0
def animationUpdate():
  global nAnimIter
  moveParticles( cudaPre(dt), posX_d, posY_d, posZ_d, velX_d, velY_d, velZ_d, accelX_d, accelY_d, accelZ_d )
  mainKernel(np.uint8(True), np.uint8(0), np.int32(nParticles), 
	      posX_d, posY_d, posZ_d, velX_d, velY_d, velZ_d,
	      accelX_d, accelY_d, accelZ_d,
	      cudaPre( dt ), cudaPre(epsilon),
	      np.intp(pAnim.cuda_VOB_ptr),  grid=grid, block=block)
  #temp = ( posX_d, posY_d, posZ_d )
  #posX_d, posY_d, posZ_d = outPosX_d, outPosY_d, outPosZ_d
  #outPosX_d, outPosY_d, outPosZ_d = temp
  #copyDtoD_float3D(outPosX_d, outPosY_d, outPosZ_d, posX_d, posY_d, posZ_d)
  nAnimIter += 1



if usingAnimation:
  pAnim.updateFunc = animationUpdate
  pAnim.startAnimation()