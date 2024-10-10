import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        x = np.linspace(0, 1, self.N+1)
        y = np.linspace(0, 1, self.N+1)
        
        self.xij , self.yij = np.meshgrid(x,y,indexing = 'ij',sparse=sparse)

    def D2(self, N):
        """Return second order differentiation matrix"""
        A = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N +1),  format='lil')
        return A

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx = sp.pi*self.mx
        ky = sp.pi*self.my
        return sp.sqrt((kx**2 + ky**2))*self.c

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        """Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """

        self.create_mesh(N)
        D = self.apply_bcs() 

        init0 = sp.lambdify((x, y), self.ue(mx, my).subs(t, 0))
        self.Uprev = init0(self.xij, self.yij)

        c_dt_sq = .5 *  (self.c * self.dt) ** 2

        self.U = self.Uprev + c_dt_sq * (D @ self.Uprev + self.Uprev @ D.T) 


    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.h / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = sp.lambdify((x, y), self.ue(self.mx, self.my).subs(t, t0))
        ue = ue(self.xij, self.yij)

        return np.sqrt(self.h**2 * np.sum((ue - u)**2))

    def apply_bcs(self):
        """Apply the boundary conditions"""
        mat = self.D2(self.N)
        mat[0] = 0
        mat[-1] = 0
        return mat/self.h**2


    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.Nt = Nt
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        self.h = 1/self.N
        self.create_mesh(N)
        D2 = self.apply_bcs() 


        self.initialize(N, mx, my)

        if store_data > 0:
            solutions = {0: self.Uprev.copy()}
        if store_data == 1:
            solutions = {0: self.Uprev.copy(), 1: self.U.copy()}
    
        #time step loop
        c_dt_sq = (self.c * self.dt) ** 2
        for i in range(1,self.Nt):
            self.Unext = 2*self.U -self.Uprev + c_dt_sq * (D2 @ self.U + self.U @ D2.T) 
            self.Uprev = self.U
            self.U = self.Unext

            if store_data > 0 and i % store_data == 0:
                solutions[i] = self.Uprev.copy()

        if store_data == -1:
            l2_err = self.l2_error(self.U, self.dt*Nt)
            return self.h, l2_err
        
        return solutions

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

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
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        """Return second order differentiation matrix"""
        A = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N +1),  format='lil')
        return A

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        """Apply the boundary conditions"""
        mat = self.D2(self.N)
        mat[0,1] = 2 #first row (-2,2,...) 
        #-2*U0 + 2*U1 = 0
        mat[-1,-2] = 2 #last row (...,-2,2) 
        #-2*UN-1 + 2*UN = 0 
        #derivative = 0 at boundaries Neumann Applied (Check- emoji)
        return mat/self.h**2

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d(m):
    '''m_x = m_y = m
    C = 1/sqrt(2)
    check for l2 < 1e-12'''
    sol = Wave2D()
    solN = Wave2D_Neumann()

    tol = 1e-12

    cfl = 1/np.sqrt(2)
    Nt = 20
    N = 64

    _, err_dir = sol(N, Nt, cfl=cfl, mx=m, my=m, store_data=-1)
    _, err_neu = solN(N, Nt, cfl=cfl, mx=m, my=m,store_data=-1)

    assert err_dir < tol, f"Test failed for Dirichlet BC: Error = {err_dir}, Tolerance = {tol}"
    assert err_neu < tol, f"Test failed for Neumann BC: Error = {err_neu}, Tolerance = {tol}"



def create_gif(boundary_condition: str, filename='solution.gif'):
    """
    Create a GIF of the 3D plot of the wave equation solution.

    Parameters
    ----------
    boundary_condition : str
        Specify whether to use 'dirichlet' or 'neumann' boundary conditions.
    filename : str
        The name of the output GIF file.
    """

    N = 200
    Nt = 200
    cfl = 0.5
    mx = 2
    my = 2

    # Run the solver
    if boundary_condition == 'dirichlet':
        sol = Wave2D()
        solutions = sol(N, Nt, cfl=cfl, mx=mx, my=my, store_data=1)
    elif boundary_condition == 'neumann':
        sol = Wave2D_Neumann()
        solutions = sol(N, Nt, cfl=cfl, mx=mx, my=my, store_data=1)
    else:
        raise ValueError("Invalid boundary condition specified. Use 'dirichlet' or 'neumann'.")
    
    print(f'Creating GIF for {boundary_condition} boundary conditions...')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('U(t,x,y)')
    ax.set_title('Wave Equation Solution')


    x,y = sol.xij, sol.yij

    zmin, zmax = np.min(list(solutions.values())), np.max(list(solutions.values()))

    def update_plot(i, solutions):
        if i % 4 == 0:
            ax.clear()
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('U(t,x,y)')
            ax.set_title(f'Wave Equation Solution with  {boundary_condition.capitalize()} BC')
            ax.plot_surface(x, y, solutions[i], cmap='viridis')
            ax.set_zlim(zmin, zmax)
            return ax


    ani = animation.FuncAnimation(fig, update_plot, frames=solutions.keys(), fargs=(solutions,), repeat=True, interval = 100)


    print("Saving GIF...")
    path = 'report/figures/' + filename
    ani.save(path, writer='pillow', fps=25)
 


if __name__ == "__main__":
    # test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d(2)
    create_gif('dirichlet','dirichletwave.gif')
    create_gif('neumann','neumannwave.gif')
    print("All tests passed")
