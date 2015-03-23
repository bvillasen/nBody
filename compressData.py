import numpy as np
import sys, time, os, inspect, datetime
import h5py as h5
from mpi4py import MPI



#Initialize MPI
MPIcomm = MPI.COMM_WORLD
pId = MPIcomm.Get_rank()
nProc = MPIcomm.Get_size()
name = MPI.Get_processor_name()


if pId == 0:
  print "\nMPI nBoby compress"
  print " nProcess: {0}\n".format(nProc) 
MPIcomm.Barrier()
print "[pId {0}] Host: {1}".format( pId, name )
time.sleep(1)

#Load data file
inputDir = '/home_local/bruno/data/nBody/'
inDataFile = inputDir + 'test_{0}.h5'.format(pId)
print "[pId {0}] Input: {1}".format( pId, inDataFile )
inFile = h5.File( inDataFile , "r")
posHD = inFile['pos']
steps = posHD.keys()
nSteps, nParticles = len( steps ), posHD[steps[0]].shape[1]
MPIcomm.Barrier()
time.sleep(1)
print "[pId {0}] nParts: {1}    nSteps: {2}".format( pId, nParticles, nSteps ) 

#Process data
pos = np.zeros([nSteps, nParticles, 3 ], dtype=np.float32)
for step in range(nSteps):
  pos[step] = posHD[steps[step]][...].T
  
#Send data to pId_0
if pId == 0:
  dataOtherAll = {}
  dataOtherAll[0] = pos.copy()
  for idOther in range( 1, nProc ):
    dataOther = np.zeros_like( pos )
    MPIcomm.Recv(dataOther, source=idOther, tag=idOther)
    dataOtherAll[idOther] = dataOther.copy()
    print "[pId {0}] Recived data from: {1}".format( pId, idOther ) 

else:  MPIcomm.Send(pos, dest=0, tag=pId)
######################################################################
#Save all data to one file
if pId == 0:
  outputDir = inputDir
  outDataFile = outputDir + 'test_all.h5'
  print '\n[pId {0}] Saving all data: {1}'.format( pId, outDataFile )
  outFile = h5.File( outDataFile , "w")
  posGroup = outFile.create_group("pos")
  for step in steps:
    posStep = posGroup.create_dataset( step , ( nProc*nParticles, 3  ), "float32", compression="gzip", compression_opts=9)
    for i in range(nProc):
      posStep[i*nParticles:(i+1)*nParticles,:] = dataOtherAll[i]
  print '\n[pId {0}] Data saved'.format( pId )


######################################################################
#Clean and Finalize
MPIcomm.Barrier()
dFile.close()
#Terminate MPI
MPIcomm.Barrier()
MPI.Finalize()
print "##########################################################END-{0}".format(pId)

