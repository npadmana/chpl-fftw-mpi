/* Multilocale test of Chapel with MPI.

   I haven't tried to see if this can be optimized in terms
   of communication....
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
coforall loc in Locales do on loc {
  fftw_mpi_init();
}

// Get useful information
const size=commSize();

// Verify that size divides Ng
if ((Ng%size)!=0) && (Ng%2==0) {
  writeln("mpi size must divide Ng and Ng must be even");
  MPI_Abort(CHPL_COMM_WORLD, 1);
}

writeln("Hello Brad! This is SPMD Chapel running with MPI");
writef("Chapel is running with %i locales\n",numLocales);

coforall loc in Locales do on loc {
  const rank=commRank();
  for irank in 0.. #size {
    if rank==irank then writef("This is MPI rank %i of size %i \n",rank, size);
    MPI_Barrier(CHPL_COMM_WORLD);
  }
}

// Now for the distributed data structure. We just build
// something easy here, although this could be easily abstracted.
class LocalArrays {
  var rank : int; // Cache this for simplicity
  var D : domain(3);
  var A, B : [D] real;

  proc LocalArrays() {
    rank=commRank();
    // size is not local
    D = {(rank*Ng/size)..#(Ng/size),0.. #Ng, 0.. #Ng2};
  }
}
var aList : [LocaleSpace] LocalArrays;
var localsum1, localsum2 : [LocaleSpace] real;

// Initialize
coforall loc in Locales do on loc
{
  aList[here.id] = new LocalArrays();
  ref me = aList[here.id];
  var rng = makeRandomStream(real,seed=(me.rank+1)*1234);
  me.A = 0.0;
  rng.fillRandom(me.A[..,..,0.. #Ng]);
  me.B = me.A;
  localsum1[here.id] = + reduce me.A[..,..,0.. #Ng];
  localsum2[here.id] = + reduce me.A[..,..,0.. #Ng]**2;
}


// Back in locale 1, let's do the sum
var sum1, sum2 : real;
sum1 = + reduce localsum1;
sum2 = + reduce localsum2;
writef("Total sum A=%er, sum A^2 = %er \n",sum1, sum2);

// Now FFT
coforall loc in Locales do on loc
{
  ref me = aList[here.id];
  var fwd = fftw_mpi_plan_dft_r2c_3d(Ng, Ng, Ng, me.B, me.B, CHPL_COMM_WORLD, FFTW_ESTIMATE);
  execute(fwd);
  destroy_plan(fwd);
}

// This element is on the main process
{
  ref me = aList[here.id];
  writef("Element at k=(0,0,0) = %er \n",me.B[0,0,0]);
  writef("Error = %er \n", me.B[0,0,0]/sum1 - 1);
  writef("Imaginary component (expected=0) : %er \n", me.B[0,0,1]);
}

// Parseval's theorem
// Inverse transform
coforall loc in Locales do on loc
{
  ref me = aList[here.id];
  var tmp : real;
  tmp = 2*(+ reduce me.B[..,..,2..(Ng-1)]**2);
  tmp += (+ reduce me.B[..,..,0..1]**2);
  tmp += (+ reduce me.B[..,..,Ng..(Ng+1)]**2);
  localsum2[here.id] = tmp;
}
var ksum2 = + reduce localsum2;
ksum2 *= invNg3;
writef("Total sum B^2 = %er , error= %er\n",ksum2, ksum2/sum2 - 1);

// Inverse transform
coforall loc in Locales do on loc
{
  ref me = aList[here.id];
  var rev = fftw_mpi_plan_dft_c2r_3d(Ng, Ng, Ng, me.B, me.B, CHPL_COMM_WORLD, FFTW_ESTIMATE);
  execute(rev);
  destroy_plan(rev);
  me.B *= invNg3;
  var Dx = me.D[..,..,0.. #Ng];
  localsum1[here.id] = max reduce abs(me.A[Dx]-me.B[Dx]);
}
var diff = max reduce localsum1;
writef("Max diff = %er\n",diff);


coforall loc in Locales do on loc
{
  fftw_mpi_cleanup();
  delete aList[here.id];
}
writeln("Goodbye, Brad! I hope you enjoyed this distributed FFTW example");

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
