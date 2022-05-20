# fast_ed
This is a fast, low memory class for diagonalizing the square lattice J1-J2 Heisenberg model (https://en.wikipedia.org/wiki/J1_J2_model).
The class can either diagonalize the full Hamiltonian matrix with a dense, high memory model (i.e. all eigenstates and eigenvalues) or it can use a sparse, fast, low memory method and compute the kth lowest eigenstates and eigenvalues using the Lanczos algorithm.
