/* Multilocale test of Chapel with MPI.

   I haven't tried to see if this can be optimized in terms
   of communication....
 */

use SysCTypes;
use MPI;
use C_MPI;
use FFTW;
use Random;
use PrivateDist;
use BlockDist;
require "fftw3-mpi.h";

// Define configuration constants
config const Ng=128; // Grid size-- must be divisible by MPI size
const Ng2=Ng+2;
const invNg3 = 1.0/(Ng:real)**3;

// Setup FFTW
forall loc in PrivateSpace {
  fftw_mpi_init();
}

// Get useful information
const size=commSize();

// Verify that size divides Ng
if ((Ng%size)!=0) && (Ng%2==0) {
  writeln("mpi size must divide Ng and Ng must be even");
  MPI_Abort(CHPL_COMM_WORLD, 1);
}

writeln("Hello Brad! This is Multilocale Chapel running with MPI");
writef("Chapel is running with %i locales\n",numLocales);

forall loc in PrivateSpace {
  const rank=commRank();
  for irank in 0.. #size {
    if rank==irank then writef("This is MPI rank %i of size %i \n",rank, size);
    Barrier(CHPL_COMM_WORLD);
  }
}

const DSpace={0..#Ng,0..#Ng,0..#Ng2};
var targets : [0..#numLocales,0..0,0..0] locale;
targets[..,0,0]=Locales;
const D : domain(3) dmapped Block(boundingBox=DSpace, targetLocales=targets) = DSpace;

// Initialize arrays
var A, B : [D] real;
fillRandom(A, seed=1234);
B = A;

// Get the sum
var sum1, sum2 : real;
forall a in A[..,..,0..#Ng] with (+ reduce sum1,
                    + reduce sum2) {
  sum1 += a;
  sum2 += a**2;
}
writef("Total sum A=%er, sum A^2 = %er \n",sum1, sum2);

// Now FFT
forall loc in PrivateSpace {
  var idx = B.localSubdomain().low;
  Barrier(CHPL_COMM_WORLD);
  {
    // MPI calls
    var fwd = fftw_mpi_plan_dft_r2c_3d(Ng, Ng, Ng, B[idx], B[idx], CHPL_COMM_WORLD, FFTW_ESTIMATE);
    execute(fwd);
    destroy_plan(fwd);
  }
}

// This element is on the main process
writef("Element at k=(0,0,0) = %er \n",B[0,0,0]);
writef("Error = %er \n", B[0,0,0]/sum1 - 1);
writef("Imaginary component (expected=0) : %er \n", B[0,0,1]);

// Parseval's theorem
// Inverse transform
var ksum2 : real;
ksum2 = 2*(+ reduce B[..,..,2..(Ng-1)]**2);
ksum2 += (+ reduce B[..,..,0..1]**2);
ksum2 += (+ reduce B[..,..,Ng..(Ng+1)]**2);
ksum2 *= invNg3;
writef("Total sum B^2 = %er , error= %er\n",ksum2, ksum2/sum2 - 1);
 
// Inverse transform
forall loc in PrivateSpace {
  var idx = B.localSubdomain().low;
  Barrier(CHPL_COMM_WORLD);
  {
    // MPI calls
    var rev = fftw_mpi_plan_dft_c2r_3d(Ng, Ng, Ng, B[idx], B[idx], CHPL_COMM_WORLD, FFTW_ESTIMATE);
    execute(rev);
    destroy_plan(rev);
  }
}
B *= invNg3;

var diff = max reduce abs(A[..,..,0..#Ng] - B[..,..,0..#Ng]);
writef("Max diff = %er\n",diff);

forall loc in PrivateSpace {
  fftw_mpi_cleanup();
}
writeln("Goodbye, Brad! I hope you enjoyed this distributed FFTW example");

//////////////////////////////////////////////
// Declarations here
/////////////////////////////////////////////
extern proc fftw_mpi_init();
extern proc fftw_mpi_cleanup();
extern proc fftw_mpi_plan_dft_r2c_3d(n0 : c_ptrdiff, n1 : c_ptrdiff, n2 : c_ptrdiff,
                                     ref inarr , ref outarr,
                                     comm : MPI_Comm, flags : c_uint) : fftw_plan;
extern proc fftw_mpi_plan_dft_c2r_3d(n0 : c_ptrdiff, n1 : c_ptrdiff, n2 : c_ptrdiff,
                                     ref inarr, ref outarr,
                                     comm : MPI_Comm, flags : c_uint) : fftw_plan;
