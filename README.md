A demonstration that Chapel can interoperate with MPI and 
a distributed FFTW.

We have a simple workload :
  * we initialize a 3D real array, distributed as FFTW expects.
  * We randomly fill it
  * We compute the sum and sum of squares of this array.
  * We FFT this, and recompute the sum of squares. These should agree. 
  * As a trivial test, we also look at the k=0,0,0 element and verify that
it equals the sum.
  * We FFT this back, and verify that we get the original array.

We consider two cases : SPMD Chapel + MPI and multi-locale Chapel+MPI.

