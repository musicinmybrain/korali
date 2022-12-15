/*******************/
/*  SOLVER OPTIONS */
/*******************/

// OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.0 -nu $NU -poissonTol 1.0e-3 -tend 0 -muteAll 0 -verbose 1 -poissonTolRel 0 -poissonSolver cuda_iterative"

// mpi option
// std::string OPTIONS = " -bpdx 3 -bpdy 8 -levelMax 4 -Rtol 0.1 -Ctol 0.01 -extent 1.4 -CFL 0.22 -tdump 0.0 -nu 0.000243 -poissonTol 1.0e-3 -tend 0 -muteAll 0 -verbose 1 -poissonTolRel 0 -poissonSolver iterative";

// Cuda option
std::string OPTIONS = " -bpdx 3 -bpdy 4 -levelMax 4 -Rtol 0.1 -Ctol 0.01 -extent 0.7 -CFL 0.4 -tdump 0.0 -nu 0.000243 -poissonTol 1.0e-3 -tend 0 -muteAll 0 -verbose 1 -poissonTolRel 0 -poissonSolver cuda_iterative";

std::string OBJECTS = " -shapes windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=0.1 ypos=0.25 bForced=1 bFixed=1 xvel=0.15 tAccel=0 bBlockAng=1 \n\
windmill semiAxisX=0.0405 semiAxisY=0.0135 xpos=0.1 ypos=0.45 bForced=1 xvel=0.15 tAccel=0 bBlockAng=1";
