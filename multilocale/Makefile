env:
	@printchplenv

fft.x: fft.chpl
	CHPL_COMM=gasnet CHPL_COMM_SUBSTRATE=mpi CHPL_TASKS=fifo \
	  CHPL_TARGET_COMPILER=mpi-gnu \
		chpl -o fft.x -O --fast fft.chpl -lfftw3_mpi -lfftw3

run: 
	AMMPI_MPI_THREAD=multiple MPICH_MAX_THREAD_SAFETY=multiple \
		mpirun -np 4 ./fft.x_real -nl 4
