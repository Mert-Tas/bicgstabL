//
// BicgstabL header file
//

#ifndef BICGSTABL_H
#define BICGSTABL_H

#include <eigen3/Eigen/Sparse>

#using namespace std;

// If realNum is defined as double, realVector should be Eigen::VectorXd
// If realNum is defined as float , realVector should be Eigen::VectorXf
typedef double realNum;
typedef Eigen::VectorXd realVector;

/// References
/// Article & Pseudocode: GERARD L.G. SLEIJPEN AND DIEDERIK R. FOKKEMA
/// Algorithm: meep::bicgstab() https://github.com/NanoComp/meep/blob/master/src/bicgstab.cpp
class BicgstabL
{
private:
    // Exit value if rho is smaller than this
    constexpr static realNum BREAK_VALUE = 1e-30;
    // Output no more often than this many seconds
    constexpr static realNum MIN_OUTPUT_TIME = 4.0;

public:
    BicgstabL();
    ~BicgstabL();

    /// Calculates Vector_x += Scalar_a * Vector_y for each element
    static void accumulate(ll size, realVector &x, realNum a, const realVector &y);

    /// The BiCGSTAB(L) algorithm.
    /// An improvement over the classical Bicgstab algorithm.
    /// BiCGSTAB(1) algorithm is reported that if the eigenvalues of matrix A
    /// is complex values than it may stagnate.
    ///
    /// BiCGSTAB(2) algorithm can deal with complex values by discarding
    /// the imaginary part.
    /// From Meep::BiCGSTAB(L) :
    ///        The reason that we use this generalization of BiCGSTAB is that the
    ///        BiCGSTAB(1) algorithm was observed by Sleijpen and Fokkema to have
    ///        poor (or even failing) convergence when the linear operator has
    ///        near-pure imaginary eigenvalues.  This is precisely the case for
    ///        our problem (the eigenvalues of the timestep operator are i*omega),
    ///        and we observed precisely such stagnation of convergence.  The
    ///        BiCGSTAB(2) algorithm was reported to fix most such convergence
    ///        problems, and indeed L > 1 seems to converge well for us.

    ///@param L:        Level (mostly taken 2)
    ///       matSize:  Side length of the square A matrix or the length of vector B and vector X
    ///       maxIters: Maximum number of iterations allowed for the algorithm to loop
    ///       tolerance: Break value for the algorithm to converge and stop.
    ///       matrix A : Sparse matrix A in the equation Ax = b
    ///       vector b : A x vectorX
    ///       vector x : Resultant vector
    ///@result vector x, errorCode
    static int solve(int L, ll matSize, int maxIters, double tolerance,
                     Eigen::SparseMatrix<realNum, Eigen::RowMajor> &sparseMatrixA,
                     const realVector& vectorB, realVector &vectorX);

};

#endif //BICGSTABL_H
