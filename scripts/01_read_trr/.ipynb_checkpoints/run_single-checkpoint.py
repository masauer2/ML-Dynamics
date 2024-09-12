import MDAnalysis as mda
import numpy as np
from numpy import linalg 
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import warnings
from MLDynamics import io, properties

# I am sick of seeing the pandas warning about whitespace. 
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-id", dest = 'id', type=int, help="Replica ID we want to analyze")
args = parser.parse_args()

####################
# Define constants #
####################
start = time.time()
PDB_COORDINATE_COLUMNS = [5,6,7]

# Project Directory
# Where are all the files located
WORKING_DIR = "/scratch/masauer2/FMA_KEMP/KE_ML/classic_study/template_R1_0"

# Subdirectory that the trajectories (.trr) are located in
# i.e each replica will be in file 02-sample_X (where X ranges from 0 to 19)
TRAJECTORY_DIR = "08-converged/02-sample_"
TRAJECTORY_NAME = "sample-NPT_pbc_rot+trans.trr"

# Number of replicas
N_REPLICAS = 1
REPLICA_ID = int(args.id)
LOG_FILE = f"step1_{REPLICA_ID}.log"

# Reference Structure - this is only used to import the trr file w/ MDAnalysis
# We redefine the reference structure to be the first timestep of each trajectory
# See variable array - startTime
GRO_FILEPATH = "/scratch/masauer2/FMA_KEMP/KE_ML/classic_study/template_R1_0/00-prep/prot.gro"

# trajectoryNames stores the name of each trajectory 
# then we will import all replicas into one long trajectory
end = time.time()

with open(LOG_FILE, 'w', buffering=1) as f:
    f.write(f"Constants have been read in.\n Working Directory = {WORKING_DIR}\nTrajectory Directory = {WORKING_DIR}/{TRAJECTORY_DIR}\nReference Structure = {GRO_FILEPATH}\nTime Elapsed = {end - start}.\n")
    f.flush()
###################################################################################################

# Read in trajectories - already unwrapped and centered
trajectoryNames = []
trr = f"{WORKING_DIR}/{TRAJECTORY_DIR}{REPLICA_ID}/{TRAJECTORY_NAME}"
trajectoryNames.append(trr)

# system = all 20 replicas
system = mda.Universe(GRO_FILEPATH,trajectoryNames)
nTimes = len(system.trajectory)
nAtoms = len(system.atoms)
end = time.time()

with open(LOG_FILE, 'w', buffering=1) as f:
    f.write(f"Trajectory has been read in.\nNumber of Replicas = {N_REPLICAS}\n Number of Frames = {nTimes}\n Number of Atoms = {nAtoms}\nTime Elapsed = {end - start}.\n")
    f.flush()
    
###################################################################################################


# Now read in the coordinates from the system
coordinates = np.zeros((nTimes, nAtoms, 3))
for ts in system.trajectory:
    coordinates[ts.frame] = ts.positions
coordinates = coordinates.reshape((nTimes, nAtoms * 3))

# Extra information that will be used to construct final datastructure (df)
atom_names = system.atoms.names
res_names = system.atoms.resnums
atom_masses = system.atoms.masses

# An unique identifier for all features
column_names = [f"{res_names[iter]}_{atom_names[iter]}" for iter in range(len(atom_names))]

# The mass of each DEGREE OF FREEDOM (not atom!!!)
DOF_masses = [atom_masses[atom] for atom in range(len(atom_masses)) for DOF in ["X", "Y", "Z"]]

end = time.time()

with open(LOG_FILE, 'w', buffering=1) as f:
    f.write(f"Successfully read in system coordinates. Ready to compute displacements.\nTime Elapsed = {end - start}.\n")
    f.flush()
###################################################################################################
displacement = np.zeros((np.shape(coordinates)[0], np.shape(coordinates)[1]))

# Reference time i.e starting time for each simulations based on total time
nTimesPerSimulation = nTimes/N_REPLICAS
startTime = np.zeros(nTimes, dtype=int)
simulationIdx = np.zeros(nTimes, dtype=int)
simulationTime = np.zeros(nTimes, dtype=int)
for dt in range(nTimes):
    startTime[dt] = int(dt/nTimesPerSimulation) * nTimesPerSimulation
    simulationIdx[dt] = int(REPLICA_ID)
    simulationTime[dt] = dt % nTimesPerSimulation

# Compute displacement
for timestep in range(len(coordinates)):
    displacement[timestep] = coordinates[timestep] - coordinates[startTime[timestep]]

end = time.time()

with open(LOG_FILE, 'w', buffering=1) as f:
    f.write(f"Successfully compute displacements. Ready to read mode file.\nTime Elapsed = {end - start}.\n")
    f.flush()
###################################################################################################

# FRESEAN Mode file that is used as the collective variable file for plumed
mode = "/scratch/masauer2/FMA_KEMP/KE_ML/metad_study/template_R1_metad/05-ModeProj/plumed-mode-input.pdb"

# We have one reference structure
nStructure = 1

# We have two modes -> 0 THz Mode 7 and 0 THz Mode 8
nModes = 2

eigenvector = np.array(pd.read_csv(mode, skiprows = lambda x: x in [(nAtoms+2)*i - 1 for i in range(1,1+nModes)] or x in [(nAtoms+2)*i for i in range(1,1+nModes)] or x in [0,(nAtoms+2)*(nModes+nStructure)-1], header=None, delim_whitespace=True)).reshape((nModes+nStructure,nAtoms,10))

# We will ignore the reference structure and read in the two modes
mode_coordinates = eigenvector[nStructure:,:,PDB_COORDINATE_COLUMNS].reshape((2,nAtoms*3))

# Normalize!!
mode_coordinates_norm = [mode_coordinates[i]/(np.linalg.norm(mode_coordinates[i])) for i in range(nModes)]

# Extra information that will be used to construct final datastructure (df)
mode_resnum = eigenvector[nStructure:,:,4].reshape((2,nAtoms))[0]
mode_resname = eigenvector[nStructure:,:,2].reshape((2,nAtoms))[0]
mode_column_names = [f"{mode_resnum[iter]}_{mode_resname[iter]}_{DOF}" for iter in range(len(mode_resnum)) for DOF in ["X", "Y", "Z"]]

# Reshape so the 3 DOF per atom are in 1 line
mode_coordinates_reshaped = [mode_coordinates_norm[i].reshape((nAtoms,3)) for i in range(nModes)]

end = time.time()

with open(LOG_FILE, 'w', buffering=1) as f:
    f.write(f"Successfully read in mode file.\n PLUMED Input File = {mode}\nRead to compute projections.\nTime Elapsed = {end - start}.\n")
    f.flush()
###################################################################################################
# Number of frames = Number of times (this variable can be reset to a small value for testing)
#nFrames = nTimes
nFrames = 1000
# Reshape displacement matrix so that all DOF per atom are one vector 
displacement_perAtom = displacement[:nFrames].reshape((nFrames,nAtoms,3))

# Array to store displacement
projection_perAtom = np.zeros((nModes, np.shape(displacement_perAtom)[0],nAtoms))

for i in range(np.shape(displacement_perAtom)[0]):
    for j in range(np.shape(displacement_perAtom)[1]):
        # Hard coded for 2 modes - extend for any number of modes
        for modeNum in range(nModes):
            projection_perAtom[modeNum,i,j] = np.sqrt(atom_masses[j])*properties.projection(displacement_perAtom[i,j],mode_coordinates_reshaped[modeNum][j],nDim=3)

end = time.time()

with open(LOG_FILE, 'w', buffering=1) as f:
    f.write(f"Successfully computed projections.\n Ready to add labels and output result.\nTime Elapsed = {end - start}.\n")
    f.flush()
###################################################################################################
displacementProjection_Modes = np.zeros((np.shape(displacement_perAtom)[0],nAtoms), dtype=object)
for row in range(len(displacementProjection_Modes)):
    for col in range(len(displacementProjection_Modes[row])):
        displacementProjection_Modes[row, col] = [projection_perAtom[0, row, col], projection_perAtom[1, row, col]]

ML_INPUT = pd.DataFrame(displacementProjection_Modes, columns=column_names)

ML_INPUT_WITHLABELS = io.add_labels(ML_INPUT, ["Simulation Number", "Simulation Time", "Identity"], [simulationIdx, simulationTime, ["Unevolved"]*np.shape(ML_INPUT)[0]])
ML_INPUT_WITHLABELS_arr = ML_INPUT_WITHLABELS.to_numpy()

nPrint = 10
nFramesPerPrint = int(nFrames/nPrint)
for iteration in range(nPrint):
    if iteration != nPrint - 1:
        frameStart = (iteration)*nFramesPerPrint
        frameEnd = (iteration+1)*nFramesPerPrint
        np.save(f"ML_INPUT_{REPLICA_ID}_{iteration}.npy", ML_INPUT_WITHLABELS_arr[frameStart:frameEnd,:])
    elif iteration == nPrint - 1:
        frameStart = (iteration)*nFramesPerPrint
        np.save(f"ML_INPUT_{REPLICA_ID}_{iteration}.npy", ML_INPUT_WITHLABELS_arr[frameStart:,:])

np.savetxt(f"COLUMNS_{REPLICA_ID}.txt", np.array(column_names), fmt='%s')

end = time.time()

with open(LOG_FILE, 'w', buffering=1) as f:
    f.write(f"ML Input file = ML_INPUT_{REPLICA_ID}.pkl. Ready to move onto step 2!\nTime Elapsed = {end - start}.\n")
    f.flush()
###################################################################################################
