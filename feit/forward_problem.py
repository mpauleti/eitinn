import numpy as np
from fenics import (
    Constant,
    FiniteElement,
    Function,
    FunctionSpace,
    LUSolver,
    Measure,
    MeshFunction,
    MixedElement,
    SubDomain,
    TestFunction,
    TrialFunction,
    Vector,
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
        self.num_electrodes = len(self.elec_pos)
        self.z = z
        # See the function below, which uses the ElectrodeDomain class
        # to set the electrodes region.
        self.__electrodes()
        self.__space_cache = {}

    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, value):
        if isinstance(value, float):
            self.__z = np.full(self.num_electrodes, value)
            return
        if not isinstance(value, np.ndarray):
            raise TypeError("`z` must be a float or 1D np.ndarray.")
        if value.ndim == 1 and len(value) == 1:
            value = value[0]
            if not isinstance(value, float):
                raise TypeError("`z` must be a float or 1D np.ndarray.")
            self.__z = np.full(self.num_electrodes, value)
            return
        if value.shape != (self.num_electrodes,):
            raise ValueError(
                "`z` must have shape `(num_electrodes,)` "
                f"(num_electrodes = {self.num_electrodes}), "
                f"but got an array with shape {value.shape}."
            )
        self.__z = np.array(value, dtype=float)

    def __electrodes(self):
        """
        Auxiliary function that defines subdomains with electrodes
        and calculates their sizes.
        """
        sub_domains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        sub_domains.set_all(0)  # Initialize all boundary facets to default subdomain

        # Pass electrode position to mesh
        # Here we have an array of objects that provide information
        # about the vertices of each electrode in the mesh.
        list_e = [
            ElectrodeDomain(self.mesh.vertices_elec[i], self.num_electrodes)
            for i in range(self.num_electrodes)
        ]

        # Mark electrodes in subdomain
        # We pass the information to sub_domains with .mark(),
        # where the index is the electrode_index (index >= 1).
        for index, elec in enumerate(list_e, 1):
            elec.mark(sub_domains, index)

        # Defining integration domain on electrodes.
        self.de = Measure("ds", domain=self.mesh, subdomain_data=sub_domains)

        # Compute the arc length of each electrode via boundary integration.
        self.elec_size = np.array(
            [assemble(Constant(1) * self.de(i + 1)) for i in range(self.num_electrodes)]
        )

        self.list_e = list_e

    def __get_cached_spaces(self, V):
        if V in self.__space_cache:
            return self.__space_cache[V]

        mesh = self.mesh

        # FEM definition
        # Turns FiniteElement V into a standard scalar FunctionSpace
        V_FuncSpace = FunctionSpace(mesh, V)
        # Vector in R^L for the electrodes
        RL = VectorElement("R", mesh.ufl_cell(), 0, dim=self.num_electrodes)
        # Constant for Lagrange multiplier
        R = FiniteElement("R", mesh.ufl_cell(), 0)
        # Defining product space V x R^L x R
        W = FunctionSpace(mesh, MixedElement([V, RL, R]))

        # Trial and test functions for the coupled CEM system
        u0 = TrialFunction(W)
        v0 = TestFunction(W)

        u, u_elec, u_lambda = split(u0)
        v, v_elec, v_lambda = split(v0)

        de = self.de  # Integration domain on electrodes.

        # Lagrange Multiplier term to set a ground reference (sum of U_i = 0).
        # Without this, the voltage system has infinite solutions (u + const).
        # Integral(V_i * U_lambda + U_i * V_lambda) d(electrode_i)
        lagr_mult_form = sum(
            (v_elec[i] * u_lambda + u_elec[i] * v_lambda) * de(i + 1)
            for i in range(self.num_electrodes)
        )

        # Contact impedance term in the weak formulation:
        # Integral((1/zi) * (u - U_i) * (v - V_i)) d(electrode_i)
        # Note: (1/zi) is multiplied later to complete the term.
        A_imp_0_forms = [
            (u - u_elec[i]) * (v - v_elec[i]) * de(i + 1)
            for i in range(self.num_electrodes)
        ]

        # Right-hand side (b) of the linear system Ax = b.
        # In the weak formulation, the current pattern term is sum(I_i * V_i).
        # Since V_i is constant on the electrode,
        # Integral(V_i) d(electrode_i) = V_i * size(electrode_i).
        # To isolate V_i, we must divide the integral by the electrode size.
        b0 = [
            assemble(v_elec[i] * (1 / self.elec_size[i]) * de(i + 1))
            for i in range(self.num_electrodes)
        ]
        b0_np = np.array([b.get_local() for b in b0])

        dm0 = W.sub(0).dofmap()
        dm1 = W.sub(1).dofmap()

        cached = (V_FuncSpace, W, dm0, dm1, lagr_mult_form, A_imp_0_forms, b0_np)
        self.__space_cache[V] = cached
        return cached

    def solve_forward(self, V, I_all, gamma):
        """
        Solver ForwardProblem for EIT 2D.
        """
        V_FuncSpace, W, dm0, dm1, lagr_mult_form, A_imp_0_forms, b0_np = (
            self.__get_cached_spaces(V)
        )
        I_all = np.array(I_all)

        # Verify if it is a matrix or a vector
        num_patterns = len(I_all) if I_all.ndim == 2 else 1

        u0 = TrialFunction(W)
        v0 = TestFunction(W)
        u = split(u0)[0]
        v = split(v0)[0]

        # Integral(gamma * <grad_u, grad_v>) dOmega
        A_inner_form = gamma * inner(grad(u), grad(v)) * dx

        # Integral((1/zi) * (u - U_i) * (v - V_i)) d(electrode_i)
        A_imp_form = sum(
            (1 / self.z[i]) * A_imp_0_forms[i] for i in range(self.num_electrodes)
        )

        # Make matrix to solve Ax = b
        A_form = A_inner_form + A_imp_form + lagr_mult_form
        A_fenics = assemble(A_form)

        solver = LUSolver(A_fenics)

        # Precompute right-hand sides for all current patterns in bulk
        if I_all.ndim == 2:
            B_np = I_all @ b0_np  # shape: (num_patterns, N_dofs)
        else:
            B_np = (I_all @ b0_np)[np.newaxis, :]  # shape: (1, N_dofs)

        b_vec = Vector()
        x_vec = Vector()
        A_fenics.init_vector(b_vec, 0)
        A_fenics.init_vector(x_vec, 1)

        u_list = []
        U_list = []
        for j in range(num_patterns):
            b_vec.set_local(B_np[j])
            b_vec.apply("insert")
            solver.solve(x_vec, b_vec)
            U_vec_val = x_vec.get_local()

            # Append the results in the list
            u_aux = Function(V_FuncSpace)
            u_aux.vector()[:] = U_vec_val[dm0.dofs()]
            u_list.append(u_aux)
            U_list.append(U_vec_val[dm1.dofs()])

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

    def __init__(self, mesh_vertex, num_electrodes):
        # Observe that mesh_vertex corresponds to electrode i.
        super().__init__()
        self.mesh_vertex = np.array(
            mesh_vertex
        ).T  # Getting electrode vertices from the mesh
        self.num_electrodes = num_electrodes
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
