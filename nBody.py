import numpy as np
import sys, time, os 
import h5py as h5
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
from cudaTools import setCudaDevice, getFreeMemory, kernelMemoryInfo
from tools import printProgressTime, timeSplit
from dataAnalysis import *


nParticles = 1024*32
#nParticles = 1024*16*2*32
totalSteps = 100

G    = 6.67384e-11 #m**2/(kg*s**2)
mSun = 1.98e30     #kg
pc   = 3.08e16     #m
initialR =  1*pc #*np.random.random( nParticles ).astype(cudaPre)

G    = 1 #m**2/(kg*s**2)
mSun = 3     #kg
initialR =  5000

dt = 5
epsilon = 0.001

cudaP = "double"
devN = None
usingAnimation = False
showKernelMemInfo = False
plotting = True

#Read in-line parameters
for option in sys.argv:
  if option.find("part")>=0 : nParticles = int(option[option.find("=")+1:])
  if option.find("anim")>=0: usingAnimation = True
  if option.find("mem") >=0: showKernelMemInfo = True
  if option == "double": cudaP = "double"
  if option == "float": cudaP = "float"
  if option.find("dev") >= 0 : devN = int(option[-1])

precision  = {"float":np.float32, "double":np.float64} 
cudaPre = precision[cudaP]


GMass = cudaPre( G*mSun ) 

if usingAnimation: import points3D as pAnim #points3D Animation
#Set CUDA thread grid dimentions
block = ( 160, 1, 1 )
grid = ( (nParticles - 1)//block[0] + 1, 1, 1 )


if usingAnimation:
  pAnim.nPoints = nParticles
  pAnim.viewXmin, pAnim.viewXmax = -2*initialR, 2*initialR
  pAnim.viewYmin, pAnim.viewYmax = -2*initialR, 2*initialR
  pAnim.viewZmin, pAnim.viewZmax = -2*initialR, 2*initialR
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
cudaCodeString = cudaCodeStringTemp % { "TPB":block[0], "gDIM":grid[0], 'bDIM':block[0] }
cudaCode = SourceModule(cudaCodeString, no_extern_c=True, include_dirs=[currentDirectory])
mainKernel = cudaCode.get_function("main_kernel" )
if showKernelMemInfo: 
  kernelMemoryInfo(mainKernel, 'mainKernel')
  sys.exit()
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
moveParticles = ElementwiseKernel(arguments="cudaP dt, cudaP *posX, cudaP *posY, cudaP *posZ,\
				     cudaP *velX, cudaP *velY, cudaP *velZ,\
				     cudaP *accelX, cudaP *accelY, cudaP *accelZ".replace( "cudaP", cudaP ),
			      operation = "posX[i] = posX[i] + dt*( velX[i] + 0.5f*dt*accelX[i]);\
				           posY[i] = posY[i] + dt*( velY[i] + 0.5f*dt*accelY[i]);\
				           posZ[i] = posZ[i] + dt*( velZ[i] + 0.5f*dt*accelZ[i]);",
			      name ="moveParticles")
##################################################################
def loadState( files=["galaxy.hdf5"] ):
  posAll = []
  velAll = []
  for fileName in files:
    dataFile = h5.File( currentDirectory+ "/" + fileName ,'r')
    pos = dataFile.get("posParticles")[...]
    vel = dataFile.get("velParticles")[...]
    print '\nLoading data... \n file: {0} \n particles: {1}\n'.format(fileName, pos.shape[1] )
    posAll.append( pos )
    velAll.append( vel )
    dataFile.close()
  return posAll, velAll
##################################################################
def loadGalaxy( fileName = "galaxy.hdf5", nParticles= 32000  ):
  dataDir = "/home/bruno/Desktop/data/yt/h5/"
  dataFile = h5.File( dataDir + fileName ,'r')
  dataAll = dataFile.get("dataAll")[...]
  print '\nLoading data... \n file: {0} \n particles: {1}\n'.format(fileName, dataAll.shape[0] )
  data = dataAll[:nParticles,:]
  return data

########################################################################
def saveState():
  dataFileName = "galaxy.hdf5"
  dataFile = h5.File(dataFileName,'w')
  dataFile.create_dataset( "posParticles", data=np.array([posX_d.get(), posY_d.get(), posZ_d.get() ]), compression='lzf')
  dataFile.create_dataset( "velParticles", data=np.array([velX_d.get(), velY_d.get(), velZ_d.get() ]), compression='lzf')
  dataFile.close()
  print "Data Saved: ", dataFileName, "\n"
########################################################################
def getRadialDist():
  posX_h, posY_h, posZ_h = posX_d.get(), posY_d.get(), posZ_d.get()
  cm = np.array([ posX_h.mean(), posY_h.mean(), posZ_h.mean() ])
########################################################################
#Initialize all gpu data
print "Initializing Data"
initialMemory = getFreeMemory( show=True )  
##Spherically uniform random distribution for initial positions
initialTheta = 2*np.pi*np.random.rand(nParticles)
initialPhi = np.arccos(2*np.random.rand(nParticles) - 1) 
#initialR = ( 50*pc )**3#*np.random.random( nParticles ).astype(cudaPre)
#initialR = np.power(initialR, 1./3)
posX_h = ( initialR*np.cos(initialTheta)*np.sin(initialPhi) ).astype(cudaPre)
posY_h = ( initialR*np.sin(initialTheta)*np.sin(initialPhi) ).astype(cudaPre)
posZ_h = ( initialR*np.cos(initialPhi) ).astype(cudaPre)
pos_h = ( np.concatenate([ posX_h, posY_h, posZ_h ]) ).astype(cudaPre)
#posX_h[:nParticles/2] += 5000
#posX_h[nParticles/2:] -= 5000
##Spherically uniform random distribution for initial velocity
#initialTheta = 2*np.pi*np.random.rand(nParticles)
#initialPhi = np.arccos(2*np.random.rand(nParticles) - 1)
initialR = 0.
velX_h = initialR*np.cos(initialTheta)*np.sin(initialPhi)
velY_h = initialR*np.sin(initialTheta)*np.sin(initialPhi)
velZ_h = initialR*np.cos(initialPhi)
##################################################################
##Disk Distribution
#initialR = 5000**2 * np.random.random( nParticles )
#initialR = np.sqrt( initialR )
#posX_h = initialR*np.cos(initialTheta)
#posY_h = initialR*np.sin(initialTheta)
#posZ_h = 0*np.random.rand(nParticles)
#velX_h = -1.6*posY_h/initialR
#velY_h =  1.6*posX_h/initialR
#velZ_h = np.zeros(nParticles)
##################################################################
posX_d = gpuarray.to_gpu( posX_h )
posY_d = gpuarray.to_gpu( posY_h )
posZ_d = gpuarray.to_gpu( posZ_h )
pos_d  = gpuarray.to_gpu( pos_h )
velX_d = gpuarray.to_gpu( velX_h.astype(cudaPre) )
velY_d = gpuarray.to_gpu( velY_h.astype(cudaPre) )
velZ_d = gpuarray.to_gpu( velZ_h.astype(cudaPre) )
accelX_d = gpuarray.to_gpu( np.zeros( nParticles, dtype=cudaPre ) )
accelY_d = gpuarray.to_gpu( np.zeros( nParticles, dtype=cudaPre ) )
accelZ_d = gpuarray.to_gpu( np.zeros( nParticles, dtype=cudaPre ) )
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6) 


def animationUpdate():
  global nAnimIter, runningTime
  start, end = cuda.Event(), cuda.Event()
  start.record()
  moveParticles( cudaPre(dt), posX_d, posY_d, posZ_d, velX_d, velY_d, velZ_d, accelX_d, accelY_d, accelZ_d )
  mainKernel( np.int32(nParticles), GMass, np.int32(usingAnimation), 
	      posX_d, posY_d, posZ_d, velX_d, velY_d, velZ_d,
	      accelX_d, accelY_d, accelZ_d,
	      cudaPre( dt ), cudaPre(epsilon),
	      np.intp(pAnim.cuda_VOB_ptr), np.int32(0),  grid=grid, block=block )
  nAnimIter += 1
  end.record()
  end.synchronize()
  secs = start.time_till(end)*1e-3
  runningTime += secs
  if nAnimIter == 50:
    print 'Steps per sec: {0:0.2f}'.format( 50/runningTime  )
    nAnimIter, runningTime = 0, 0
  
def keyboard(*args):
  global viewXmin, viewXmax, viewYmin, viewYmax
  global showGrid, gridCenter
  ESCAPE = '\033'
  if args[0] == ESCAPE:
    print "Ending Simulation"
    sys.exit()
  elif args[0] == "s":
    saveState()
###########################################################################
###########################################################################

nAnimIter = 0
runningTime = 0

#Start Simulation
if plotting: plt.ion(), plt.show()
print "\nStarting simulation"
print " Using {0} precision".format( cudaP )
print ' nParticles: ', nParticles


if usingAnimation:
  pAnim.updateFunc = animationUpdate
  pAnim.keyboard = keyboard
  pAnim.startAnimation()
  
  

print ' nSteps: ', totalSteps
print ''
start, end = cuda.Event(), cuda.Event()
for stepCounter in range(totalSteps):
  start.record()
  if stepCounter%1 == 0: printProgressTime( stepCounter, totalSteps,  runningTime )
  moveParticles( cudaPre(dt), posX_d, posY_d, posZ_d, velX_d, velY_d, velZ_d, accelX_d, accelY_d, accelZ_d )
  mainKernel( np.int32(nParticles), GMass, np.int32(0), 
	      posX_d, posY_d, posZ_d, velX_d, velY_d, velZ_d,
	      accelX_d, accelY_d, accelZ_d,
	      cudaPre( dt ), cudaPre(epsilon),
	      np.int32(0), np.int32(0),  grid=grid, block=block )
  end.record()
  end.synchronize()
  runningTime += start.time_till(end)*1e-3
  
  
h, m, s = timeSplit( runningTime )
print '\n\nTotal time: {0}:{1:02}:{2:02} '.format( h, m, s )





