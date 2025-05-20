# Load the various modules we need for the whole project
module -q purge
module load modules/2.3-20240529
module load gcc/11.4.0
module load openmpi/4.0.7
module load hdf5/mpi-1.14.3
module load intel-oneapi-mkl/2024.0.0
module load netcdf-c/4.9.2
module load flexiblas/3.4.2
module load cmake/3.27.9
module load python/3.10.13
module load eigen/3.4.0
module load llvm/14.0.6
module load git/2.42.0
module load vscode/1.85.2-nix
module load hwloc/2.9.1
module load boost/1.84.0
module load slurm

# Load spack (configure for your own install)
. ~cedelmaier/Projects/Software/spack/share/spack/setup-env.sh
# Activate the spack Trilinos16 CPU environment
spack env activate ~cedelmaier/spack_builds/Trilinos_16_0_0_CPU/.
