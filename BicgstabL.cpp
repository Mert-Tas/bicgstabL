//
// BicgstabL source file
//

#include <iostream>
#include <chrono>
#include <fstream>
#include <ctime>
#include "BicgstabL.h"

BicgstabL::BicgstabL()    = default;
BicgstabL::~BicgstabL()    = default;

void BicgstabL::accumulate(ll size, realVector &x, realNum a, const realVector &y)
{
    for (ll m = 0; m < size; ++m)
    {
        x[m] += a * y[m];
    }
}

// // Ax = b -> result vector X  size: matSize
int BicgstabL::solve(const int L, const ll matSize, int maxIters, const double tolerance,
                     Eigen::SparseMatrix<realNum, Eigen::RowMajor> &sparseMatrixA,
                     const realVector& vectorB, realVector &vectorX)
{
    int errCode = 0;
    //int k = -L;

    // r, u : vector of Eigen::Vectors of matSize
    vector<realVector> r(L + 1);
    vector<realVector> u(L + 1);
    for (int i = 0; i <= L; ++i)
    {
        r[i].resize(matSize);
        u[i].resize(matSize);
    }

    // Set u0 vector to all zeros. Not needed values are initially zero
    // u[0] = realVector::Zero(matSize);

    // rtilde = r[0] = b - Ax
    realVector rtilde(L + 1);
    r[0] = vectorB - sparseMatrixA * vectorX;
    rtilde = r[0];

    // Sleipjen normalizes rtilde in his code; it seems to help slightly
    {
        realNum s = 1.0 / rtilde.norm();
        for (int m = 0; m < matSize; ++m)
            rtilde[m] *= s;
    }

    realVector gamma(L + 1);
    realVector gammaP(L + 1);
    realVector gammaPP(L + 1);
    realVector tau(L * L);
    realVector sigma(L + 1);

    realNum rho = 1.0;
    realNum alpha = 0.0;
    realNum omega = 1.0;

    realNum residual = 0.0;
    // from meep used in while exit condition
    realNum bNorm = vectorB.norm();
    if (bNorm == 0.0)
        bNorm = 1.0;

    int iter = 0;
    bool breakValueReached = false;
    // Keep current time for printing residual values every MIN_OUTPUT_TIME seconds
    clock_t begin_time = clock();

    // repeat until || r(k+l) || is small enough
    while ((residual = r[0].norm()) > tolerance * bNorm)
    {
        ++iter;
        //k = k + L;    // k = 0;

        // Print residual on every MIN_OUTPUT_TIME seconds
        if ( (float( clock () - begin_time ) /  CLOCKS_PER_SEC) > MIN_OUTPUT_TIME )
        {
            printf("Residual[%d] = %g\n", iter, residual / bNorm);
            begin_time = clock();
        }

        rho = -omega * rho;

        // Bi-CG Part
        for (int j = 0; j < L; ++j)
        {
            // From meep. Early break condition if omega is too small (~= 0)
            if (fabs(rho) < BREAK_VALUE)
            {
                cout << "rho: " << rho << " < " << BREAK_VALUE << endl;
                errCode = -1;
                breakValueReached = true;
                break;
            }

            realNum rho1 = r[j].dot(rtilde);
            realNum beta = alpha * rho1 / rho;
            rho = rho1;

            // cout << "rho =  " << rho << " omega = " << omega << endl;

            for (int i = 0; i <= j; ++i)
            {
                for (ll m = 0; m < matSize; ++m)
                {
                    u[i](m) = r[i](m) - beta * u[i](m);
                }
            }

            // Matrix multiplication u(j+1) = sparseA * u(j)
            u[j + 1] = sparseMatrixA * u[j];

            // alpha = rho / u(j+1).dot(rtilde)
            alpha = rho /( u[j+1].dot(rtilde));

            for (int i = 0; i <= j; ++i)
            {
                // r[i] += -alpha * u[i+1]
                accumulate(matSize, r[i], -alpha, u[i + 1]);
            }

            // Matrix multiplication  r[j + 1] = sparseMatrixA * r[j];
            // A(r[j], r[j + 1], Adata)
            r[j + 1] = sparseMatrixA * r[j];

            // Vector and scalar multiplication x += alpha * u[0];
            accumulate(matSize, vectorX, alpha, u[0]);
        }
        // end of Bi-CG part

        // Exit program if break is encountered
        if (breakValueReached)
            break;

        // Minimum Residual part
        for (int j = 1; j <= L; ++j)
        {
            for (int i = 1; i < j; ++i)
            {
                int ij = (j - 1) * L + (i - 1);
                // tau(ij) = 1 / sigma(i) * dot(rj, rji);
                tau[ij] = r[j].dot(r[i]) / sigma[i];
                accumulate(matSize, r[j], -tau[ij], r[i]);
            }
            // sigma(j) = rj.dot(rj)
            sigma[j] = r[j].dot(r[j]);

            // gammaPrime = (1 / sigma(j)) * r0.dot(rj)
            gammaP[j] = r[0].dot(r[j]) / sigma[j];
        }

       // omega = gamma[L] = gammaP[L];
        gamma[L] = gammaP[L];
        omega  = gamma[L];

        // gamma = tau(inverse) * gammaPrime
        for (int j = L - 1; j >= 1; --j)
        {
            gamma[j] = gammaP[j];
            for (int i = j + 1; i <= L; ++i)
            {
                gamma[j] -= tau[(i - 1) * L + (j - 1)] * gamma[i];
            }
        }

        // gammaPP = tau * S gamma
        for (int j = 1; j < L ; ++j)
        {
            gammaPP[j] = gamma[j + 1];
            for (int i = j + 1; i < L; ++i)
            {
                gammaPP[j] += tau[(i - 1) * L + (j - 1)] * gamma[i + 1];
            }
        }

        // Update vectorX, r0, u0
        accumulate(matSize, vectorX, gamma[1], r[0]);
        accumulate(matSize, r[0], -gammaP[L], r[L]);
        accumulate(matSize, u[0], -gamma[L], u[L]);

        // meep TODO: use blas DGEMV (for L > 2)
        // BLAS2_GEMV  or BLAS3_GEMM
        for (int j = 1; j < L; ++j)
        {
            accumulate(matSize, u[0], -gamma[j], u[j]);
            accumulate(matSize, vectorX, gammaPP[j], r[j]);
            accumulate(matSize, r[0], -gammaP[j], r[j]);
        }

        // Exit if maxIters is reached
        if (iter == maxIters)
        {
            cout << "Max iters " << maxIters << " reached, exiting.\n";
            errCode = 1;
            break;
        }
        // Put u(k+L-1) = u0,   r(k+L) = r0,   x(k+L) = x0

    }
    // end of while

    double finalResidual = r[0].norm() / bNorm;
    printf("\nFinal residual: %g\nMax. iterations: %d\n", finalResidual, iter);

    return errCode;
}
// End of bicgstabL()

