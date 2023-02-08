//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

/*******************/
/*  SOLVER OPTIONS */
/*******************/

/**** Training options ****/

// 2x1x0.5 DOMAIN
std::string OPTIONS="-bpdx 8 -bpdy 4 -bpdz 2 -tdump 0 -tend 0 -CFL 0.7 -lambda 1e8 -bMeanConstraint 2 -nu 0.000008 -levelMax 5 -levelStart 3 -levelMaxVorticity 5 -Rtol 4.00 -Ctol 1.00 -extentx 2.0 -poissonTol 1e-5 -poissonTolRel 1e-3 -checkpointsteps 10000000";

/*********************/
/* INITIAL POSITIONS */
/*********************/

// 4 SWIMMERS
// small domain
std::vector<std::vector<double>> initialPositions{{
	{0.60, 0.50},
	{0.90, 0.425},
	{0.90, 0.575},
	{1.20, 0.50}
}};