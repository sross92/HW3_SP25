#Utilized ChatGPT in helping write the code and definitions, while providing logic feedback and checks.
#import region
import numpy as np
import scipy.linalg as la
#endregion

def is_symmetric(matrix):
    """Check if the matrix is symmetric."""
    return np.allclose(matrix, matrix.T)

def is_positive_definite(matrix):
    """Check if the matrix is positive definite by attempting Cholesky decomposition."""
    try:
        # Attempt Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def cholesky_solve(A, b):
    """Solve the system using Cholesky decomposition."""
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)  # Solve Ly = b
    x = np.linalg.solve(L.T, y)  # Solve L^T x = y
    return x

def doolittle_solve(A, b):
    """Solve the system using LU decomposition."""
    P, L, U = la.lu(A)
    y = np.linalg.solve(L, b)  # Solve Ly = b
    x = np.linalg.solve(U, y)  # Solve Ux = y
    return x


def main():
    # Matrix A1 and vector b1 (first system)
    A1 = np.array([[1, -1, 3, 2],
                   [-1, 5, -5, -2],
                   [3, -5, 19, 3],
                   [2, -2, 3, 21]], dtype=float)
    b1 = np.array([15, -35, 94, 1], dtype=float)

    # Matrix A2 and vector b2 (second system)
    A2 = np.array([[4, 2, 4, 0],
                   [2, 2, 3, 2],
                   [4, 3, 6, 3],
                   [0, 2, 3, 9]], dtype=float)
    b2 = np.array([20, 36, 60, 122], dtype=float)

    # Solve for system 1
    print("Solving System 1...")
    if is_symmetric(A1):
        if is_positive_definite(A1):
            print("Matrix A1 is symmetric and positive definite. Using Cholesky method.")
            x1 = cholesky_solve(A1, b1)
        else:
            print("Matrix A1 is symmetric but not positive definite. Using Doolittle method.")
            x1 = doolittle_solve(A1, b1)
    else:
        print("Matrix A1 is not symmetric. Using Doolittle method.")
        x1 = doolittle_solve(A1, b1)

    print("Solution for System 1:", x1)

    # Solve for system 2
    print("\nSolving System 2...")
    if is_symmetric(A2):
        if is_positive_definite(A2):
            print("Matrix A2 is symmetric and positive definite. Using Cholesky method.")
            x2 = cholesky_solve(A2, b2)
        else:
            print("Matrix A2 is symmetric but not positive definite. Using Doolittle method.")
            x2 = doolittle_solve(A2, b2)
    else:
        print("Matrix A2 is not symmetric. Using Doolittle method.")
        x2 = doolittle_solve(A2, b2)

    print("Solution for System 2:", x2)


if __name__ == "__main__":
    main()
