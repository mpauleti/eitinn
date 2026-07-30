import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from fenics import (
    Constant,
    FiniteElement,
    Function,
    FunctionSpace,
    Measure,
    MeshFunction,
    MixedElement,
    SubDomain,
    TestFunction,
    TrialFunction,
    VectorElement,
    assemble,
    between,
    dx,
    grad,
    inner,
    split,
)


class ForwardProblem:
    """
    ForwardProblem Object for EIT 2D.
    """

    def __init__(self, elec_mesh, z):
        self.mesh = elec_mesh
        self.radius = elec_mesh.radius
        self.elec_pos = elec_mesh.electrodes.position
        self.L = len(self.elec_pos)  # L electrodes
        self.z = z
        # See the function below, which uses the ElectrodeDomain class
        # to set the electrodes region.
        self.__electrodes()

    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, value):
        if isinstance(value, float):
            self.__z = np.full(self.L, value)
            return
        if not isinstance(value, np.ndarray):
            raise TypeError("`z` must be a float or 1D np.ndarray.")
        if value.ndim == 1 and len(value) == 1:
            value = value[0]
            if not isinstance(value, float):
                raise TypeError("`z` must be a float or 1D np.ndarray.")
            self.__z = np.full(self.L, value)
            return
        if value.shape != (self.L,):
            raise ValueError(
                f"`z` must have shape `(L,)` (L = {self.L}), "
                f"but got an array with shape {value.shape}."
            )
        self.__z = np.array(value, dtype=float)

    def __electrodes(self):
        """
        Auxiliary function that defines subdomains with electrodes
        and calculates their sizes.
        """
        sub_domains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        sub_domains.set_all(0)  # Marking all vertex/edges with False

        # Pass electrode position to mesh
        # Here we have an array of objects that provide information
        # about the vertices of each electrode in the mesh.
        list_e = [
            ElectrodeDomain(self.mesh.vertices_elec[i], self.L) for i in range(self.L)
        ]

        # Mark electrodes in subdomain
        # We pass the information to sub_domains with .mark(),
        # where the index is the electrode_index (index >= 1).
        for index, elec in enumerate(list_e, 1):
            elec.mark(sub_domains, index)

        # Defining integration Domain on electrodes.
        # Save in memory, set new attribute.
        self.de = Measure("ds", domain=self.mesh, subdomain_data=sub_domains)

        # Calc elec_size.
        # Save in memory, set new attribute.
        self.elec_size = np.array(
            [assemble(Constant(1) * self.de(i + 1)) for i in range(self.L)]
        )

        # Save in memory, set new attribute.
        self.list_e = list_e

    def solve_forward(self, V, I_all, gamma):
        """
        Solver ForwardProblem for EIT 2D.
        """
        de = self.de  # Getting integral domain from memory.
        Intde = self.elec_size  # Size of electrodes.
        mesh = self.mesh
        V_FuncSpace = FunctionSpace(mesh, V)

        I_all = np.array(I_all)
        # Verify if it is a matrix or a vector
        l = len(I_all) if I_all.ndim == 2 else 1

        ### FEM definition
        # Vector in R^L for the electrodes
        RL = VectorElement("R", mesh.ufl_cell(), 0, dim=self.L)
        R = FiniteElement("R", mesh.ufl_cell(), 0)  # Constant for lagrMult
        # Defining product space V x R^L x R
        W = FunctionSpace(mesh, MixedElement([V, RL, R]))

        u0 = TrialFunction(W)  # Functions to be reconstructed.
        v0 = TestFunction(W)  # Test functions.

        u, un, ul = split(u0)
        v, vn, vl = split(v0)

        # Integral(gamma*<grad_u, grad_v>) dOmega + lagrMult
        A_inner = assemble(gamma * inner(grad(u), grad(v)) * dx)

        # Integral(v_i*u_mult + u_i*v_mult) d(electrode_i)
        lagrMult = np.sum(
            [(vn[i] * ul + un[i] * vl) * de(i + 1) for i in range(self.L)]
        )
        self.A_lagr = assemble(lagrMult)

        # Integral((1/zi)*(u - U_i)*(v - V_i)) d(electrode_i)
        A_imp_0 = [
            assemble((u - un[i]) * (v - vn[i]) * de(i + 1)) for i in range(self.L)
        ]
        self.A_imp_0 = A_imp_0
        A_imp = np.sum(self.A_imp_0 * 1 / self.z)

        # Make matrix to solve Ax = b
        # We only assemble this matrix once.
        # If we have multiple measurements, we reuse it.
        A = A_inner + A_imp + self.A_lagr
        A = scipy.sparse.csc_matrix(A.array())

        # Split the w function into 3 parts:
        # The function in H (= V), the vector in R^L, and the constant lagrMult.
        w = Function(W)  # Define a zero function based in W
        # u = w.split()[0]  # Get only the function in H
        dm0 = W.sub(0).dofmap()
        dm1 = W.sub(1).dofmap()

        # Ax = sum(I_i*V_i)...
        # We integrate over the electrodes and divide by their size.
        # If we don't do this, we get an error.
        b0 = [assemble(vn[i] * (1 / Intde[i]) * de(i + 1)) for i in range(self.L)]

        u_list = []
        U_list = []
        for j in range(l):
            I = I_all[j] if l != 1 else I_all  # Is it one measure or several?
            # Make b vector as a linear combination using `sum`
            b = sum(b0[i] * I[i] for i in range(self.L))
            w = Function(W)  # Define a zero function based in W
            U_vec = w.vector()  # Return a vector (x = U)

            # Solve system AU = b
            U_vec[:] = scipy.sparse.linalg.spsolve(A, b[:])

            # Append the results in the list
            u_aux = Function(V_FuncSpace)
            u_aux.vector()[:] = w.vector().vec()[dm0.dofs()]
            u_list.append(u_aux)
            U_list.append(w.vector().vec()[dm1.dofs()])

        return np.array(u_list), np.array(U_list)

    def eval_forward_op(self, V, I_all, gamma):
        self.u_arr, self.U_arr = self.solve_forward(V, I_all, gamma)
        self.U_arr = self.U_arr.flatten()

        return self.U_arr


# SubDomain class belongs to FEniCS, we use it to define the electrode domain.
class ElectrodeDomain(SubDomain):
    """
    Auxiliary function for ForwardProblem to define the electrode domain.
    We expect that we have a circular domain and electrodes in the boundary.
    This routine determines the vertices where the electrodes
    are defined and marks the mesh.
    """

    def __init__(
        self, mesh_vertex, L
    ):  # Observe that mesh_vertex corresponds to electrode i.
        super().__init__()
        self.mesh_vertex = np.array(
            mesh_vertex
        ).T  # Getting electrode vertices from the mesh
        self.L = L  # Setting the number of electrodes
        self.X = np.max(self.mesh_vertex[0])  # Max value axis x
        self.X1 = np.min(self.mesh_vertex[0])  # Min value axis x
        self.Y = np.max(self.mesh_vertex[1])  # Max value axis y
        self.Y1 = np.min(self.mesh_vertex[1])  # Max value axis y

    # CANNOT rename this function to something like `is_inside`.
    # FEniCS requires it to be called `inside` for using `mark()`.
    def inside(self, x, on_boundary):
        """
        Function that returns True if the vertex is in the electrode region.
        """
        # FEniCS function that evaluates the SubDomain, setting
        # True or False for each vertex.
        # Here we implement a strategy to verify if
        # the vertex is part of an electrode.
        # FEniCS provides only the vertices on the boundary.
        # After that we verify if the vertex is inside
        # a "box" at (X1,X) x (Y1,Y).

        if not on_boundary:
            return False

        in_x_bounds = between(x[0], ((self.X), (self.X1))) or between(
            x[0], ((self.X1), (self.X))
        )
        in_y_bounds = between(x[1], ((self.Y), (self.Y1))) or between(
            x[1], ((self.Y1), (self.Y))
        )
        return in_x_bounds and in_y_bounds
