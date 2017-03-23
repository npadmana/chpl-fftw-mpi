env:
	@printchplenv

all:
	CHPL_COMM=none CHPL_TASKS=fifo \
	  CHPL_TARGET_COMPILER=mpi-gnu \
		chpl -O --fast fft.chpl -lfftw3_mpi -lfftw3
