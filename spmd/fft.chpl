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
const xRange = (rank*Ng/size).. #(Ng/size);
const yRange = 0.. #Ng;
const zRange = 0.. #Ng2;
const Dall = {xRange, yRange, zRange};
const Dx = Dall[..,..,0.. #Ng];
const Dre = {xRange, yRange, zRange by 2};
const Dim = {xRange, yRange, zRange by 2 align 1};

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
  var localsum = + reduce A;
  var localsum2 = + reduce A**2;
  MPI_Reduce(localsum, sum1, 1, MPI_DOUBLE, MPI_SUM, 0, CHPL_COMM_WORLD);
  MPI_Reduce(localsum2, sum2, 1, MPI_DOUBLE, MPI_SUM, 0, CHPL_COMM_WORLD);
}
if (rank==0) {
  writef("Total sum A=%er, sum A^2 = %er \n",sum1, sum2);
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