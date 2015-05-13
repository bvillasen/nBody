using MPI
using HDF5
# push!(LOAD_PATH, "/Path/To/My/Module/")
using tools

MPI.Init()
comm = MPI.COMM_WORLD
const pId = MPI.Comm_rank(comm)
const nProc = MPI.Comm_size(comm)
const root = 0

const pLeftId   = pId == 0       ? nProc -1  : pId - 1
const pRightId  = pId == nProc-1 ? 0         : pId + 1



const nParticles = 2000
const nParticles_local = nParticles / nProc
const dim = 3

const dt = 0.02
const epsilon = 1

dataDir = "/home_local/bruno/data/nBody/"
snapFileName = dataDir * "snapshots/test_$(pId).h5"

snapshotsFile = h5open( snapFileName, "w")


#Allocate Memory
pos    = zeros( Float64, nParticles*dim )
vel    = zeros( Float64, nParticles*dim )
accel  = zeros( Float64, nParticles*dim )
accel_1 = zeros( Float64, nParticles*dim )
mass = ones( Float64, nParticles )
energy = zeros( Float64, nParticles )

#Initialize data
const R = 10
const dTheta = 2*pi / nParticles
theta = 0.
for i = 0:nParticles-1	
  pos[i*dim + 1] = R * cos( theta )
  pos[i*dim + 2] = R * sin( theta )
  pos[i*dim + 3] = 0
  theta += dTheta
end


function saveState_32( snapshot, outputFile, pos::Array{Float64}, vel::Array{Float64} )
  outputFile["pos/$(snapshot)"] = float32( pos )
  outputFile["vel/$(snapshot)"] = float32( vel )
end


function saveState_64( snapshot, outputFile, pos::Array{Float64}, vel::Array{Float64} )
  outputFile["pos/$(snapshot)"] =  pos 
  outputFile["vel/$(snapshot)"] =  vel 
end
saveState_64( 0,  snapshotsFile, pos, vel )


function advancePositions( pos::Array{Float64}, vel::Array{Float64}, accel::Array{Float64}   )
  for i = 0:nParticles-1
    idx = i*dim
    pos[idx + 1] = pos[idx + 1] + dt*( vel[idx+1] + 0.5*dt*accel[idx + 1] ) 
    pos[idx + 2] = pos[idx + 2] + dt*( vel[idx+2] + 0.5*dt*accel[idx + 2] ) 
    pos[idx + 3] = pos[idx + 3] + dt*( vel[idx+3] + 0.5*dt*accel[idx + 3] ) 
  end
end
    

function getAccel( pId, pos::Array{Float64} ) 
  px = pos[pId*dim + 1]
  py = pos[pId*dim + 2]
  pz = pos[pId*dim + 3]
  ax, ay, az = 0, 0, 0
  for j = 0:nParticles-1
    px_o = pos[j*dim + 1]
    py_o = pos[j*dim + 2]
    pz_o = pos[j*dim + 3]
    deltaX = px_o - px
    deltaY = py_o - py
    deltaZ = pz_o - pz
    distInv = 1 / sqrt( deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ + epsilon )
    ax += deltaX * ( distInv*distInv*distInv )
    ay += deltaY * ( distInv*distInv*distInv )
    az += deltaZ * ( distInv*distInv*distInv )
  end
  return [ ax ay az ]
end

function getAccel_all( pos::Array{Float64}, accel::Array{Float64} ) 
  for pId in 0:nParticles-1
    px = pos[pId*dim + 1]
    py = pos[pId*dim + 2]
    pz = pos[pId*dim + 3]
    ax, ay, az = 0, 0, 0
    for j = 0:nParticles-1
      px_o = pos[j*dim + 1]
      py_o = pos[j*dim + 2]
      pz_o = pos[j*dim + 3]
      deltaX = px_o - px
      deltaY = py_o - py
      deltaZ = pz_o - pz
      distInv = 1 / sqrt( deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ + epsilon )
      ax += deltaX * ( distInv*distInv*distInv )
      ay += deltaY * ( distInv*distInv*distInv )
      az += deltaZ * ( distInv*distInv*distInv )
    end
    accel[pId*dim + 1] = ax
    accel[pId*dim + 2] = ay
    accel[pId*dim + 3] = az
  end
end

function advanceVelocities_all( pos::Array{Float64}, vel::Array{Float64}, accel::Array{Float64}, accel_1::Array{Float64}  )
  for i in 0:nParticles-1
    idx = i*dim
    ax_0 = accel[idx + 1]
    ay_0 = accel[idx + 2]
    az_0 = accel[idx + 3]
    ax_1 = accel_1[idx + 1]
    ay_1 = accel_1[idx + 2]
    az_1 = accel_1[idx + 3]
    vel[idx + 1] += 0.5 * dt * ( ax_0 + ax_1 )
    vel[idx + 2] += 0.5 * dt * ( ay_0 + ay_1 )
    vel[idx + 3] += 0.5 * dt * ( az_0 + az_1 )
  end
end








function advanceVelocities( pos::Array{Float64}, vel::Array{Float64}, accel::Array{Float64} )
  for i in 0:nParticles-1
    idx = i*dim
    ax_0 = accel[idx + 1]
    ay_0 = accel[idx + 2]
    az_0 = accel[idx + 3]
    a_1 = getAccel( i, pos )
    vel[idx + 1] += 0.5 * dt * ( ax_0 + a_1[1] )
    vel[idx + 2] += 0.5 * dt * ( ay_0 + a_1[2] )
    vel[idx + 3] += 0.5 * dt * ( az_0 + a_1[3] )
    accel[idx + 1] = a_1[1]
    accel[idx + 2] = a_1[2]
    accel[idx + 3] = a_1[3]
  end
end
function doTimeStep_all(  pos::Array{Float64}, vel::Array{Float64}, accel::Array{Float64}, accel_1::Array{Float64} )
  advancePositions( pos, vel, accel )
  getAccel_all( pos, accel_1 )
  advanceVelocities_all( pos, vel, accel, accel_1 )
  accel, accel_1 = accel_1, accel 
end

function doTimeStep(  pos::Array{Float64}, vel::Array{Float64}, accel::Array{Float64} )
  advancePositions( pos, vel, accel )
  advanceVelocities( pos, vel, accel )
end


if pId == 0
  print( "\nnBody \n" )
  print( "\nnParticles: $nParticles \n" )
  print( "\nSnap output: $snapFileName\n" )
end


time = [ 0. 0. ]
nPartialSteps = 100
if pId ==0 print( "\nStarting $nPartialSteps steps \n" ) end
for i in 1:nPartialSteps
  if pId==0 tools.printProgress( i-1, nPartialSteps, sum(time) ) end
  time[1] += @elapsed doTimeStep( pos, vel, accel )
  time[2] += @elapsed saveState_32( i,  snapshotsFile, pos, vel )
end

##################################################################
MPI.Barrier(comm)

if pId == 0
  @printf "\n\n[ pID: %d ] totalTime: %.1f secs\n" pId sum(time)
  @printf "[ pID: %d ] Compute time: %.2f secs,  ( %.3f )\n" pId time[1] time[1]/sum(time)
  @printf "[ pID: %d ] Save time: %.2f secs, ( %.3f )\n" pId time[2] time[2]/sum(time)
end


close( snapshotsFile )
MPI.Finalize()



