/* SPMD test of Chapel with MPI.
 */

use SysCTypes;
use MPI;
use C_MPI;
use FFTW;
use Random;
require "fftw3-mpi.h";

// Define configuration constants
config const Ng=128; // Grid size-- must be divisible by MPI size
const Ng2=Ng+2;
const invNg3 = 1.0/(Ng:real)**3;

// Setup FFTW
fftw_mpi_init();

// Get useful information
const rank=commRank();
const size=commSize();
const ranks=0.. #size;

// Verify that size divides Ng
if (Ng%size)!=0 {
  writeln("mpi size must divide Ng");
  MPI_Abort(CHPL_COMM_WORLD, 1);
}

if (rank==0) {
  writeln("Hello Brad! This is SPMD Chapel running with MPI");
  writef("Chapel is running with %i locales\n",numLocales);
}
for irank in ranks {
  if (rank==irank) then
    writef("This is MPI rank %i of size %i \n",rank, size);
  MPI_Barrier(CHPL_COMM_WORLD);
}

// Define the domains we need
const xRange = (rank*Ng/size).. #(Ng/size); // This is different for the different MPI ranks
const yRange = 0.. #Ng;
const zRange = 0.. #Ng2;
const Dall = {xRange, yRange, zRange};
const Dx = Dall[..,..,0.. #Ng];
// Special case the frequencies for Parseval's thm
const Dk2 = {xRange, yRange, 2..(Ng-1)}; // These get multiplied by 2
const Dk1_0 = {xRange, yRange, 0..1}; // no double counting
const Dk1_1 = {xRange, yRange, Ng..Ng+1}; // no double counting

// Define the arrays
var A, B : [Dall] real;

// Set up the random number generator
// and fill the array
var rng = makeRandomStream(real,seed=(rank+1)*1234);
A = 0.0;
rng.fillRandom(A[Dx]);
B = A;

var sum1, sum2 : real;
{
  var localsum = + reduce A[Dx];
  var localsum2 = + reduce A[Dx]**2;
  MPI_Reduce(localsum, sum1, 1, MPI_DOUBLE, MPI_SUM, 0, CHPL_COMM_WORLD);
  MPI_Reduce(localsum2, sum2, 1, MPI_DOUBLE, MPI_SUM, 0, CHPL_COMM_WORLD);
}
if (rank==0) {
  writef("Total sum A=%er, sum A^2 = %er \n",sum1, sum2);
}

// FFTW forward
var fwd = fftw_mpi_plan_dft_r2c_3d(Ng, Ng, Ng, B, B, CHPL_COMM_WORLD, FFTW_ESTIMATE); 
execute(fwd);
destroy_plan(fwd);

// Check k=0,0,0
if rank==0 {
  writef("Element at k=(0,0,0) = %er \n",B[0,0,0]);
  writef("Error = %er \n", B[0,0,0]/sum1 - 1);
  writef("Imaginary component (expected=0) : %er \n", B[0,0,1]);
}
var ksum2 : real;
{
  var localsum2 = 2*(+ reduce B[Dk2]**2);
  localsum2 += (+ reduce B[Dk1_0]**2);
  localsum2 += (+ reduce B[Dk1_1]**2);
  MPI_Reduce(localsum2, ksum2, 1, MPI_DOUBLE, MPI_SUM, 0, CHPL_COMM_WORLD);
}
if (rank==0) {
  ksum2 *= invNg3;
  writef("Total sum B^2 = %er , error= %er\n",ksum2, ksum2/sum2 - 1);
}

// Backward transform
var rev = fftw_mpi_plan_dft_c2r_3d(Ng, Ng, Ng, B, B, CHPL_COMM_WORLD, FFTW_ESTIMATE); 
execute(rev);
destroy_plan(rev);
B *= invNg3;

// Max difference
var diff : real;
{
  var localdiff = max reduce abs(A[Dx]-B[Dx]);
  MPI_Reduce(localdiff, diff, 1, MPI_DOUBLE, MPI_MAX, 0, CHPL_COMM_WORLD);
}
if (rank==0) {
  writef("Max diff = %er\n",diff);
}

// Finalize
fftw_mpi_cleanup();
if rank==0 {
  writeln("Goodbye, Brad! I hope you enjoyed this distributed FFTW example");
}


//////////////////////////////////////////////
// Declarations here
/////////////////////////////////////////////
extern proc fftw_mpi_init();
extern proc fftw_mpi_cleanup();
extern proc fftw_mpi_plan_dft_r2c_3d(n0 : c_ptrdiff, n1 : c_ptrdiff, n2 : c_ptrdiff,
                                     inarr : [], outarr : [],
                                     comm : MPI_Comm, flags : c_uint) : fftw_plan;
extern proc fftw_mpi_plan_dft_c2r_3d(n0 : c_ptrdiff, n1 : c_ptrdiff, n2 : c_ptrdiff,
                                     inarr : [], outarr : [],
                                     comm : MPI_Comm, flags : c_uint) : fftw_plan;
