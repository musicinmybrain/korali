//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

/*******************/
/*  SOLVER OPTIONS */
/*******************/

/**** Training options ****/

// 2x1 DOMAIN
// std::string OPTIONS = "-bpdx 4 -bpdy 2 -levelMax 7 -levelStart 4 -Rtol 2 -Ctol 1 -extent 2 -CFL 0.4 -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

// 2x2 DOMAIN
// std::string OPTIONS = "-bpdx 4 -bpdy 4 -levelMax 7 -levelStart 4 -Rtol 1 -Ctol 0.01 -extent 2 -CFL 0.4 -poissonSolver cuda_iterative -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

// 4x2 DOMAIN
// std::string OPTIONS = "-bpdx 4 -bpdy 2 -levelMax 7 -levelStart 4 -Rtol 2 -Ctol 1 -extent 4 -CFL 0.4 -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

// 4x4 DOMAIN
std::string OPTIONS = "-bpdx 4 -bpdy 4 -levelMax 7 -levelStart 4 -Rtol 2 -Ctol 1 -extent 4 -CFL 0.4 -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

// 0.8x0.4 PERIODIC DOMAIN
// dx=0.3, dy=0.2 (same as other diamond school)
std::string OPTIONS_periodic = "-bpdx 3 -bpdy 2 -levelMax 6 -levelStart 4 -Rtol 5 -Ctol 0.01 -extent 0.6 -BC_x periodic -BC_y periodic -CFL 0.4 -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

/**** Testing options ****/

// 2x1 DOMAIN (with accurate solver)
std::string OPTIONS_testing = "-bpdx 4 -bpdy 2 -levelMax 7 -levelStart 4 -Rtol 2 -Ctol 1 -extent 2 -CFL 0.4 -poissonTol 1e-6 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 0";

// 4x2 DOMAIN
// std::string OPTIONS_testing = "-bpdx 4 -bpdy 2 -levelMax 7 -levelStart 4 -Rtol 2 -Ctol 1 -extent 4 -CFL 0.4 -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

// 4x4 DOMAIN
// std::string OPTIONS_testing = "-bpdx 4 -bpdy 4 -levelMax 7 -levelStart 4 -Rtol 2 -Ctol 1 -extent 4 -CFL 0.4 -poissonSolver cuda_iterative -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 0";

/*********************/
/* INITIAL POSITIONS */
/*********************/

// PERIODIC FISH

// dx=0.3, dy=0.2
// Extent = [0.6, 0.4]
// double margin = 0.1;
// std::vector<std::vector<double>> initialPositions{{
// 	{0.15, margin + 0.05},
// 	{0.45, margin + 0.15}
// }};

// dx=0.3, dy=0.2 (two fundamental building blocks diamond school)
// std::vector<std::vector<double>> initialPositions{{
// 	{0.15, margin + 0.10},
// 	{0.45, margin + 0.30},
// 	{0.75, margin + 0.10},
// 	{1.05, margin + 0.30}
// }};

// SCHOOLS

// 4 SWIMMERS
// small domain
// std::vector<std::vector<double>> initialPositions{{
// 	{0.60, 0.50},
// 	{0.90, 0.40},
// 	{0.90, 0.60},
// 	{1.20, 0.50}
// }};

// large domain
// std::vector<std::vector<double>> initialPositions{{
// 	{0.60, 1.00},
// 	{0.90, 0.90},
// 	{0.90, 1.10},
// 	{1.20, 1.00}
// }};

// 9 SWIMMERS
// large domain
// std::vector<std::vector<double>> initialPositions{{
// 	{0.60, 1.00},
// 	{0.90, 0.90},
// 	{0.90, 1.10},
// 	{1.20, 0.80},
// 	{1.20, 1.00},
// 	{1.20, 1.20},
// 	{1.50, 0.90},
// 	{1.50, 1.10},
// 	{1.80, 1.00}
// }};

// 16 SWIMMERS
// std::vector<std::vector<double>> initialPositions{{
// 	{0.60, 1.00},
// 	{0.90, 0.90},
// 	{0.90, 1.10},
// 	{1.20, 0.80},
// 	{1.20, 1.00},
// 	{1.20, 1.20},
// 	{1.50, 0.70},
// 	{1.50, 0.90},
// 	{1.50, 1.10},
// 	{1.50, 1.30},
// 	{1.80, 0.80},
// 	{1.80, 1.00},
// 	{1.80, 1.20},
// 	{2.10, 0.90},
// 	{2.10, 1.10},
// 	{2.40, 1.00}
// }};

// 25 SWIMMERS
// std::vector<std::vector<double>> initialPositions{{
//	{0.60, 1.00},
// 	{0.90, 0.90},
// 	{0.90, 1.10},
// 	{1.20, 0.80},
// 	{1.20, 1.00},
// 	{1.20, 1.20},
// 	{1.50, 0.70},
// 	{1.50, 0.90},
// 	{1.50, 1.10},
// 	{1.50, 1.30},
// 	{1.80, 0.60},
// 	{1.80, 0.80},
// 	{1.80, 1.00},
// 	{1.80, 1.20},
// 	{1.80, 1.40},
// 	{2.10, 0.70},
// 	{2.10, 0.90},
// 	{2.10, 1.10},
// 	{2.10, 1.30},
// 	{2.40, 0.80},
// 	{2.40, 1.00},
// 	{2.40, 1.20},
// 	{2.70, 0.90},
// 	{2.70, 1.10},
// 	{3.00, 1.00},
// }};

// COLUMN WITH 4 FISH
// std::vector<std::vector<double>> initialPositions{{
// 	{0.60, 0.50},
// 	{0.90, 0.50},
// 	{1.20, 0.50},
// 	{1.50, 0.50}
// }};

// 5 COLUMNS WITH 4 FISH = 20 FISH
// std::vector<std::vector<double>> initialPositions{{
// 	{ 0.60, 0.90 },
// 	{ 0.60, 1.10 },
// 	{ 0.90, 0.80 },
// 	{ 0.90, 1.00 },
// 	{ 0.90, 1.20 },
// 	{ 1.20, 0.90 },
// 	{ 1.20, 1.10 },
// 	{ 1.50, 0.80 },
// 	{ 1.50, 1.00 },
// 	{ 1.50, 1.20 },
// 	{ 1.80, 0.90 },
// 	{ 1.80, 1.10 },
// 	{ 2.10, 0.80 },
// 	{ 2.10, 1.00 },
// 	{ 2.10, 1.20 },
// 	{ 2.40, 0.90 },
// 	{ 2.40, 1.10 },
// 	{ 2.70, 0.80 },
// 	{ 2.70, 1.00 },
// 	{ 2.70, 1.20 }
// }};

// 25 COLUMNS WITH 4 FISH = 100 FISH
std::vector<std::vector<double>> initialPositions{{
	{ 0.60, 0.90 },
	{ 0.60, 1.10 },
	{ 0.60, 1.30 },
	{ 0.60, 1.50 },
	{ 0.60, 1.70 },
	{ 0.60, 1.90 },
	{ 0.60, 2.10 },
	{ 0.60, 2.30 },
	{ 0.60, 2.50 },
	{ 0.60, 2.70 },
	{ 0.60, 2.90 },
	{ 0.60, 3.10 },
	{ 0.90, 0.80 },
	{ 0.90, 1.00 },
	{ 0.90, 1.20 },
	{ 0.90, 1.40 },
	{ 0.90, 1.60 },
	{ 0.90, 1.80 },
	{ 0.90, 2.00 },
	{ 0.90, 2.20 },
	{ 0.90, 2.40 },
	{ 0.90, 2.60 },
	{ 0.90, 2.80 },
	{ 0.90, 3.00 },
	{ 0.90, 3.20 },
	{ 1.20, 0.90 },
	{ 1.20, 1.10 },
	{ 1.20, 1.30 },
	{ 1.20, 1.50 },
	{ 1.20, 1.70 },
	{ 1.20, 1.90 },
	{ 1.20, 2.10 },
	{ 1.20, 2.30 },
	{ 1.20, 2.50 },
	{ 1.20, 2.70 },
	{ 1.20, 2.90 },
	{ 1.20, 3.10 },
	{ 1.50, 0.80 },
	{ 1.50, 1.00 },
	{ 1.50, 1.20 },
	{ 1.50, 1.40 },
	{ 1.50, 1.60 },
	{ 1.50, 1.80 },
	{ 1.50, 2.00 },
	{ 1.50, 2.20 },
	{ 1.50, 2.40 },
	{ 1.50, 2.60 },
	{ 1.50, 2.80 },
	{ 1.50, 3.00 },
	{ 1.50, 3.20 },
	{ 1.80, 0.90 },
	{ 1.80, 1.10 },
	{ 1.80, 1.30 },
	{ 1.80, 1.50 },
	{ 1.80, 1.70 },
	{ 1.80, 1.90 },
	{ 1.80, 2.10 },
	{ 1.80, 2.30 },
	{ 1.80, 2.50 },
	{ 1.80, 2.70 },
	{ 1.80, 2.90 },
	{ 1.80, 3.10 },
	{ 2.10, 0.80 },
	{ 2.10, 1.00 },
	{ 2.10, 1.20 },
	{ 2.10, 1.40 },
	{ 2.10, 1.60 },
	{ 2.10, 1.80 },
	{ 2.10, 2.00 },
	{ 2.10, 2.20 },
	{ 2.10, 2.40 },
	{ 2.10, 2.60 },
	{ 2.10, 2.80 },
	{ 2.10, 3.00 },
	{ 2.10, 3.20 },
	{ 2.40, 0.90 },
	{ 2.40, 1.10 },
	{ 2.40, 1.30 },
	{ 2.40, 1.50 },
	{ 2.40, 1.70 },
	{ 2.40, 1.90 },
	{ 2.40, 2.10 },
	{ 2.40, 2.30 },
	{ 2.40, 2.50 },
	{ 2.40, 2.70 },
	{ 2.40, 2.90 },
	{ 2.40, 3.10 },
	{ 2.70, 0.80 },
	{ 2.70, 1.00 },
	{ 2.70, 1.20 },
	{ 2.70, 1.40 },
	{ 2.70, 1.60 },
	{ 2.70, 1.80 },
	{ 2.70, 2.00 },
	{ 2.70, 2.20 },
	{ 2.70, 2.40 },
	{ 2.70, 2.60 },
	{ 2.70, 2.80 },
	{ 2.70, 3.00 },
	{ 2.70, 3.20 }
}};
