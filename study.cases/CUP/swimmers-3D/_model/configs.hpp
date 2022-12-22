//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

/*******************/
/*  SOLVER OPTIONS */
/*******************/

/**** Training options ****/

// 2x1x0.5 DOMAIN
std::string OPTIONS="-bpdx 4 -bpdy 2 -bpdz 1 -tdump 0.1 -tend 50 -CFL 0.7 -dt  -lambda 1e8 -use-dlm 0 -nu 0.00001 -levelMax 6 -levelStart 4 -levelMaxVorticity 5 -Rtol 4.00 -Ctol 1.00 -extentx 2.0  -poissonTol 1e-5 -poissonTolRel 1e-3";

/*********************/
/* INITIAL POSITIONS */
/*********************/

// 4 SWIMMERS
// small domain
std::vector<std::vector<double>> initialPositions{{
	{0.60, 0.50},
	{0.90, 0.40},
	{0.90, 0.60},
	{1.20, 0.50}
}};