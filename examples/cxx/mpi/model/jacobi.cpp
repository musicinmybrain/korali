/**********************************************************************/
// A generic stencil-based 3D Jacobi solver                           //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin                                             //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "jacobi.h"

struct Neighbor {
 int rankId;
 double* recvBuffer;
 double* sendBuffer;
};

void jacobi(std::vector<double> xdata, std::vector<double>& parameters, std::vector<double>& results, MPI_Comm comm)
{
 int myRank, rankCount;
 MPI_Comm_rank(comm, &myRank);
 MPI_Comm_size(comm, &rankCount);

 int N = 128;
 int nIters = 50;
 int nx, ny, nz;
 Neighbor X1, X0, Y0, Y1, Z0, Z1;

 int dims[3] = {0,0,0};
 MPI_Dims_create(rankCount, 3, dims);
 int px = dims[0];
 int py = dims[1];
 int pz = dims[2];

 nx = N / px; // Internal grid size (without halo) on X Dimension
 ny = N / py; // Internal grid size (without halo) on Y Dimension
 nz = N / pz; // Internal grid size (without halo) on Z Dimension

 if(N % px > 0) { if (myRank == 0) printf("Jacobi3D Model Error: N (%d) should be divisible by px (%d)\n", N, px); MPI_Abort(comm, -1); }
 if(N % py > 0) { if (myRank == 0) printf("Jacobi3D Model Error: N (%d) should be divisible by py (%d)\n", N, py); MPI_Abort(comm, -1); }
 if(N % pz > 0) { if (myRank == 0) printf("Jacobi3D Model Error: N (%d) should be divisible by pz (%d)\n", N, pz); MPI_Abort(comm, -1); }

 int fx = nx + 2; // Grid size (with halo) on X Dimension
 int fy = ny + 2; // Grid size (with halo) on Y Dimension
 int fz = nz + 2; // Grid size (with halo) on Z Dimension

 double *U, *Un;
 U  = (double*) calloc(sizeof(double), fx*fy*fz);
 Un = (double*) calloc(sizeof(double), fx*fy*fz);

 int periodic[3] = {false, false, false};
 MPI_Comm gridComm; // now everyone creates a cartesian topology
 MPI_Cart_create(comm, 3, dims, periodic, true, &gridComm);

 MPI_Cart_shift(gridComm, 0, 1, &X0.rankId, &X1.rankId);
 MPI_Cart_shift(gridComm, 1, 1, &Y0.rankId, &Y1.rankId);
 MPI_Cart_shift(gridComm, 2, 1, &Z0.rankId, &Z1.rankId);

 for (int k = 0; k < fz; k++)
 for (int j = 0; j < fy; j++)
 for (int i = 0; i < fx; i++)
   U[k*fy*fx + j*fx + i] = 1;

 if (X0.rankId == MPI_PROC_NULL) for (int i = 0; i < fy; i++) for (int j = 0; j < fz; j++) U[j*fx*fy + i*fx  ] = parameters[0];
 if (Y0.rankId == MPI_PROC_NULL) for (int i = 0; i < fx; i++) for (int j = 0; j < fz; j++) U[j*fx*fy + fx + i] = parameters[2];
 if (Z0.rankId == MPI_PROC_NULL) for (int i = 0; i < fx; i++) for (int j = 0; j < fy; j++) U[fx*fy + j*fx + i] = parameters[4];

 if (X1.rankId == MPI_PROC_NULL) for (int i = 0; i < fy; i++) for (int j = 0; j < fz; j++) U[j*fx*fy + i*fx + (nx+1)] = parameters[1];
 if (Y1.rankId == MPI_PROC_NULL) for (int i = 0; i < fx; i++) for (int j = 0; j < fz; j++) U[j*fx*fy + (ny+1)*fx + i] = parameters[3];
 if (Z1.rankId == MPI_PROC_NULL) for (int i = 0; i < fx; i++) for (int j = 0; j < fy; j++) U[(nz+1)*fx*fy + j*fx + i] = parameters[5];

 MPI_Datatype faceXType, faceYType, faceZType;

 MPI_Type_vector(fz*fy, 1,  fx,    MPI_DOUBLE, &faceXType); MPI_Type_commit(&faceXType);
 MPI_Type_vector(fz,    fx, fx*fy, MPI_DOUBLE, &faceYType); MPI_Type_commit(&faceYType);
 MPI_Type_vector(fy,    fx, fx,    MPI_DOUBLE, &faceZType); MPI_Type_commit(&faceZType);

 MPI_Request request[12];

 MPI_Barrier(comm);

 for (int iter=0; iter<nIters; iter++)
 {
  for (int k = 1; k <= nz; k++)
  for (int j = 1; j <= ny; j++)
  for (int i = 1; i <= nx; i++)
  {
   double sum = 0.0;
   sum += U[fx*fy*k + fx*j  + i + 0]; // Central
   sum += U[fx*fy*k + fx*j  + i - 1]; // Y0
   sum += U[fx*fy*k + fx*j  + i + 1]; // Y1
   sum += U[fx*fy*k + fx*(j-1)  + i]; // X0
   sum += U[fx*fy*k + fx*(j+1)  + i]; // X1
   sum += U[fx*fy*(k-1) + fx*j  + i]; // Z0
   sum += U[fx*fy*(k+1) + fx*j  + i]; // Z1
   Un[fx*fy*k + fx*j + i] = sum/7.0;
  }
  double *temp = U; U = Un; Un = temp;

  int request_count = 0;

  MPI_Irecv(&U[0            ], 1, faceXType, X0.rankId, 0, gridComm, &request[request_count++]);
  MPI_Irecv(&U[(nx+1)       ], 1, faceXType, X1.rankId, 0, gridComm, &request[request_count++]);
  MPI_Irecv(&U[0            ], 1, faceYType, Y0.rankId, 0, gridComm, &request[request_count++]);
  MPI_Irecv(&U[fx*(ny+1)    ], 1, faceYType, Y1.rankId, 0, gridComm, &request[request_count++]);
  MPI_Irecv(&U[0            ], 1, faceZType, Z0.rankId, 0, gridComm, &request[request_count++]);
  MPI_Irecv(&U[fx*fy*(nz+1) ], 1, faceZType, Z1.rankId, 0, gridComm, &request[request_count++]);

  MPI_Isend(&U[1            ], 1, faceXType, X0.rankId, 0, gridComm, &request[request_count++]);
  MPI_Isend(&U[nx           ], 1, faceXType, X1.rankId, 0, gridComm, &request[request_count++]);
  MPI_Isend(&U[fx           ], 1, faceYType, Y0.rankId, 0, gridComm, &request[request_count++]);
  MPI_Isend(&U[fx*ny        ], 1, faceYType, Y1.rankId, 0, gridComm, &request[request_count++]);
  MPI_Isend(&U[fx*fy        ], 1, faceZType, Z0.rankId, 0, gridComm, &request[request_count++]);
  MPI_Isend(&U[fx*fy*nz     ], 1, faceZType, Z1.rankId, 0, gridComm, &request[request_count++]);

  MPI_Waitall(request_count, request, MPI_STATUS_IGNORE);
 }

 int myCoords[3];
 int myCartRank;
 MPI_Comm_rank(gridComm, &myCartRank);
 MPI_Cart_coords(gridComm, myCartRank, 3, myCoords);

 double dx = 1.0 / dims[0];
 double dy = 1.0 / dims[1];
 double dz = 1.0 / dims[2];

 double startX = dx * myCoords[0];
 double startY = dy * myCoords[1];
 double startZ = dz * myCoords[2];

 double endX = startX + dx;
 double endY = startY + dy;
 double endZ = startZ + dz;

 for (size_t i = 0; i < xdata.size(); i += 3)
 {
  MPI_Barrier(comm);

  double result = 0.0;

  double xPos = xdata[i + 0];
  double yPos = xdata[i + 1];
  double zPos = xdata[i + 2];

  bool foundValue = false;

  if(xPos >= startX && xPos <= endX)
  if(yPos >= startY && yPos <= endY)
  if(zPos >= startZ && zPos <= endZ)
  {
   double h = 1.0/N;
   int ix = ceil((xPos-startX)/h);
   int iy = ceil((yPos-startY)/h);
   int iz = ceil((zPos-startZ)/h);

   foundValue = true;
   result = U[fx*fy*iz + fx*iy + ix];
  }

  if (foundValue == true) if (myRank != 0) MPI_Send(&result, 1, MPI_DOUBLE, 0, 0, comm);

  if (myRank == 0)
  {
   if (foundValue == false) MPI_Recv(&result, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);
   results.push_back(result);
  }
 }

 free(U);
 free(Un);
 MPI_Type_free(&faceXType);
 MPI_Type_free(&faceYType);
 MPI_Type_free(&faceZType);
 MPI_Comm_free(&gridComm);
}
