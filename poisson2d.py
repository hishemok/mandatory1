import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = self.L/self.N
        x = np.linspace(0, self.L, self.N+1)
        y = np.linspace(0, self.L, self.N+1)
        self.xij , self.yij = np.meshgrid(x,y,indexing = 'ij')
        return self.xij , self.yij 

    def D2(self):
        """Return second order differentiation matrix"""

        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N +1),  format='lil')
        D2[0,:4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        return D2

    def laplace(self):
        """Return vectorized Laplace operator"""
        I = np.eye(self.N+1)
        D2 = self.D2()
        laplace_x = sparse.kron(D2,I)/self.h**2 #second order differentiation matrix in x direction
        laplace_y = sparse.kron(I,D2)/self.h**2 # and in y direction
        tot = laplace_x + laplace_y
        return tot.tocsr()  

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        #For N+1 x N+1 | exs: 9x9, top 0,1,...,8 left 0,9,18,...,72 right 8,17,26,...,80 bottom 72,73,...,80. total 81=9x9
        top = np.arange(self.N+1)
        bottom = np.arange(self.N + 1) + self.N * (self.N + 1)
        left = np.arange(self.N+1, self.N**2 + self.N+1, self.N + 1)
        right = np.arange(self.N, self.N**2 + self.N, self.N + 1)

        return np.concatenate([top, bottom, left, right])

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        A = self.laplace()
        boundaries = self.get_boundary_indices()
        A[boundaries] = 0 
        A[boundaries, boundaries] = 1 #Dirichlet boundary conditions
      
        b = sp.lambdify((x, y), self.f)(self.xij, self.yij).flatten()
        e = sp.lambdify((x, y), self.ue)(self.xij, self.yij).flatten()
        b[boundaries] = e[boundaries]

        return A,b

    def l2_error(self, u):
        """Return l2-error norm"""
        ue = sp.lambdify((x, y), self.ue)(self.xij, self.yij)
        return np.sqrt(self.h**2*np.sum((ue-u)**2))

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A.tocsr(), b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
    
        i = int(x / self.h)
        j = int(y / self.h)

        #find the lower corner points
        x_low = i * self.h
        y_low = j * self.h
        #find the upper corner points
        x_up = x_low + self.h
        y_up = y_low + self.h

        x_dist = np.array([x_up - x, x - x_low])
        y_dist = np.array([y_up - y, y - y_low])

        f = self.U
        bl = f[i, j] #bottom left
        br = f[i, j + 1] #bottom right
        tl = f[i + 1, j] #top left
        tr = f[i + 1, j + 1] #top right
        U = np.array([[bl, br], [tl, tr]])

        
        ev = x_dist @ U @ y_dist / (self.h**2)
        return ev
    


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    print(sol.eval(0.52, 0.63))
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3

if __name__ == '__main__':
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed")

