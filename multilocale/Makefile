env:
	@printchplenv

%.x: %.chpl
	CHPL_COMM=gasnet CHPL_COMM_SUBSTRATE=mpi CHPL_TASKS=fifo \
	  CHPL_TARGET_COMPILER=mpi-gnu \
		chpl -o $@ -O --fast $< -lfftw3_mpi -lfftw3

run-%: %.x
	AMMPI_MPI_THREAD=multiple MPICH_MAX_THREAD_SAFETY=multiple \
		mpirun -np 4 ./$<_real -nl 4
