from .forward_problem import ForwardProblem

import numpy as np
from scipy.sparse import csr_matrix

from fenics import *


class InverseProblem(ForwardProblem):
    """
    InverseProblem Object for EIT 2D.

    Class for solving the inverse problem of the Electrical Impedance Tomography (EIT) in 2D.
    The inverse problem aims to reconstruct the electrical conductivity distribution within a
    domain based on measured electrical potentials at the boundary (electrodes).
    """

    def __init__(self, elec_mesh, data, I_all, z):
        super().__init__(elec_mesh, z)

        # Function Space CG degree 1 is necessary
        self.V = FiniteElement("CG", elec_mesh.ufl_cell(), 1)
        self.Q_DG = FunctionSpace(self.mesh, "DG", 0)
        self.I_all = I_all  # Current pattern used in data
        self.U_delta = np.array(data)  # Electrodes potencial in array
        # Verify if it is a matrix or a vector
        self.I_all = np.array(self.I_all)
        self.l = len(self.I_all) if self.I_all.ndim == 2 else 1

        ### Solver configuration
        self.step_limit = 20  # Outer iteration
        self.weight = np.ones(elec_mesh.num_cells())  # Initial weight function
        self.use_weight = (
            True  # Are you going to use the weight function in the Jacobian matrix?
        )
        self.min_v = 0.005  # Minimal value for `gamma_n` elements
        self.verbose = True

        ### Inner iteration configuration
        self.inner_method = "Lp"  # Default inner regularization method
        self.ME_reg = 15e-4  # Minimal Error step
        self.inner_step_limit = 50
        self.use_constant_inner_limit = True
        self.inner_step_start = 500

        ### Newton tolerance parameters
        self.mu_start = 0.3  # Used to compute the first two values of mu
        self.mu_max = 0.999  # Max mu in (0,1]
        self.mu_theta = 0.9  # Decreasing factor for last mu
        self.mu_constant = None

        self.penalty_beta = 1.0  # Regularization penalty term
        self.TV_EPS = 1e-4  # Smooth TV

        ### Banach spaces parameters
        self.Lp_space = 2  # Input space X = Lp
        self.Lr_space = 2  # Data space Y = Lr

        ### Creating a vector with all cell volumes. It's usefull for integrals in Lp(Omega).
        cell_list = [cell.volume() for cell in cells(elec_mesh)]
        self.cell_arr = np.array(cell_list)

        # Make a vector with electrodes size that are chosen for the problem.
        # This vector is used in norm_funcLp(dOmega).
        self.size_elec_vec = np.tile(self.elec_size, self.l)

    def solve_inverse(self):
        """
        Solves the inverse problem using a regularization method.

        This method is the main solver of the inverse problem.
        It iteratively solves the EIT problem using a Newton-like method.
        The solution is stored in the `gamma_n` attribute.
        """
        ## Creating basic variables
        self.step = 0  # Save outer step (singleton)
        self.inner_step_list = []  # Save steps from inner loop
        res_list = []  # Save residuals along the iterations
        mu_list = []  # Save mu along the iterations
        gamma_all = [self.initial_guess_primal]  # Saving all gamma_n
        # p = self.Lp_space  # Input space X = Lp
        r = self.Lr_space  # Data space Y = Lr
        y_delta = self.U_delta

        Fx_n = self.eval_forward_op(self.V, self.I_all, self.gamma_n)

        b_n = y_delta - Fx_n
        misfit_n = self.norm_funcLp(b_n, "dOmega", ord=r)

        # Save status
        # Residual list
        res_list.append(misfit_n / self.norm_funcLp(y_delta, "dOmega", ord=r) * 100)

        # Print information
        if self.verbose:
            print(
                f"Stop when reaching {self.tau*self.noise_level*100:.4f}% of the residual."
            )
            print(f"Initial residual (%): {res_list[0]:.6f}")

        #### Solver
        # n = 0 (self.step)
        x_n = np.copy(self.initial_guess_primal)
        xi_n = np.copy(self.initial_guess_dual)

        # While discrepancy principle or limit of steps.
        th = self.tau * self.noise_level  # threshold
        while res_list[self.step] / 100 > th and self.step < self.step_limit:
            mu_n = self.calc_tolerance(
                res_list[self.step] / 100, constant_value=self.mu_constant
            )
            A_n = self.calc_jacobian()

            x_n, xi_n, inner_step = self.solve_innerNewton(x_n, xi_n, b_n, mu_n, A_n)
            self.step += 1

            self.Cellsgamma_n = np.copy(x_n)
            self.gamma_n.vector()[:] = self.Cellsgamma_n

            Fx_n = self.eval_forward_op(self.V, self.I_all, self.gamma_n)

            b_n = y_delta - Fx_n
            misfit_n = self.norm_funcLp(b_n, "dOmega", ord=r)

            ## Saving progress
            gamma_all.append(self.Cellsgamma_n)
            res_list.append(misfit_n / self.norm_funcLp(y_delta, "dOmega", ord=r) * 100)
            self.inner_step_list.append(inner_step)
            mu_list.append(mu_n)

            # Print information
            if self.verbose:
                print(
                    f"Residual (%): {res_list[self.step]:.6f}    Step: {self.step:<4}  Inner step (prev): {inner_step:<4}  mu (prev): {self.mu:.6f}"
                )

        # Vectors to memory object
        self.gamma_all = np.array(gamma_all)
        self.res_list = res_list
        self.mu_list = mu_list

        return

    def solve_innerNewton(self, x_n, xi_n, b_n, mu_n, A_n):
        """
        Solve the linearized problem (inner step).

        This method is used to solve the inner step of the Newton method.
        It iteratively updates the solution `gamma_n` using different regularization methods.
        """
        p = self.Lp_space
        r = self.Lr_space
        pstar = p / (p - 1)
        # rstar = r / (r - 1)
        kappa = np.maximum(p, 2)
        norm_b_n = self.norm_funcLp(b_n, "dOmega", ord=r)

        # If we are using weights, compute and store them.
        if self.use_weight and self.step == 0:
            self.weight = self.weight_func(A_n)

        # Adjoint operator (A_n)*
        if self.use_weight:
            D = np.diag(1 / self.weight)
            A_nstar = D @ A_n.T  # Add weight
        else:
            A_nstar = A_n.T

        k = 0

        x_nk = np.copy(x_n)
        xi_nk = np.copy(xi_n)

        # linear_diff = A_n(x_nk - x_n) - b_n
        #             = A_n(s_nk) - b_n
        # For k = 0, linear_diff = -b_n
        linear_diff = -b_n
        linear_res = norm_b_n  # norm of `linear_diff`

        th_linear = mu_n * norm_b_n  # threshold
        kmax_n = self.calc_inner_step_limit(use_constant=self.use_constant_inner_limit)
        while linear_res >= th_linear and k < kmax_n:
            ##### Gradient method, section 5.1
            ## Compute direction vector
            if r == 2:
                w_nk = A_nstar @ linear_diff
            else:
                w_nk = A_nstar @ _dm(linear_diff, r)
            ## Compute step-size: minimal error method
            ## Described at section 7.3
            if p == 2 and r == 2:
                lamb_nk = self.ME_reg * (
                    linear_res**2 / self.norm_funcLp(w_nk, "Omega") ** 2
                )
            else:
                d = self.norm_funcLp(w_nk, "Omega", ord=pstar)  # denominator

                lamb1 = linear_res ** (r * (kappa - 1)) / d**kappa
                lamb2 = linear_res ** (r * (p - 1)) / d**p

                lamb_nk = self.ME_reg * np.minimum(lamb1, lamb2)
            ####

            xi_nk -= lamb_nk * w_nk
            x_nk = self.grad_conjugate(xi_nk, x0=x_nk)

            x_nk, xi_nk = self.filter_low_values(x_nk, xi_nk)

            linear_diff = A_n @ (x_nk - x_n) - b_n
            linear_res = self.norm_funcLp(linear_diff, "dOmega", ord=r)

            k += 1

        return x_nk, xi_nk, k

    def calc_inner_step_limit(self, *, use_constant):
        if use_constant:
            return self.inner_step_limit

        step = self.step
        if step < 2:
            return self.inner_step_start

        inner_steps = self.inner_step_list

        kmax = inner_steps[step - 1] + inner_steps[step - 2]
        # kmax = np.minimum(kmax, self.inner_step_limit)
        return kmax

    def filter_low_values(self, x, xi):
        # Do not have values too close to zero
        mask = x < self.min_v
        if any(mask):
            x[mask] = self.min_v
            xi = self.__primal_to_dual(x)

        return x, xi

    def grad_conjugate(self, xi, *, x0):
        """Computes the gradient of the conjugate function, grad(f*)."""
        if self.inner_method == "Lp":
            p = self.Lp_space
            if p == 2:
                return np.copy(xi)

            pstar = p / (p - 1)
            return _dm(xi, pstar)

        elif self.inner_method == "Sparse1":
            p = self.Lp_space
            beta = self.penalty_beta

            sgn = np.sign(xi)
            shift = np.abs(xi) - beta
            mask = shift < 0
            shift[mask] = 0
            s = shift

            if p == 2:
                return sgn * s

            pw = 1 / (p - 1)
            s **= pw
            return sgn * s

        elif self.inner_method == "Sparse2":
            p = self.Lp_space
            beta = self.penalty_beta

            sgn = np.sign(xi)
            shift = np.abs(xi) - 1
            mask = shift < 0
            shift[mask] = 0
            s = beta * shift

            if p == 2:
                return sgn * s

            pw = 1 / (p - 1)
            s **= pw
            return sgn * s

        elif self.inner_method in ["TV1", "TV2"]:
            L = self.TV_L
            # objfunc = lambda x: self.__objfunc(x, xi, L)
            objgrad = lambda x: self.__objgrad(x, xi, L)

            return _minacc(objgrad, x0, ord=self.Lp_space)

    # def __eval_TV(self, x, L):
    #     EPS = self.TV_EPS

    #     Lx = L @ x
    #     TV_smooth = np.sqrt(Lx**2 + EPS)

    #     return np.sum(TV_smooth)

    def __grad_TV(self, x, L):
        EPS = self.TV_EPS

        Lx = L @ x
        q = Lx / np.sqrt(Lx**2 + EPS)

        return L.T @ q

    # def __objfunc(self, x, xi, L):
    #     p = self.Lp_space
    #     beta = self.penalty_beta

    #     TV_x = self.__eval_TV(x, L)

    #     Lp = (1 / p) * np.sum(np.abs(x) ** p)

    #     affine_x = np.inner(xi, x)

    #     if self.inner_method == "TV2":
    #         return TV_x + (1 / beta) * Lp - affine_x
    #     return beta * TV_x + Lp - affine_x

    def __objgrad(self, x, xi, L):
        p = self.Lp_space
        beta = self.penalty_beta

        grad_TV_x = self.__grad_TV(x, L)

        if self.inner_method == "TV2":
            return grad_TV_x + (1 / beta) * _dm(x, p) - xi
        return beta * grad_TV_x + _dm(x, p) - xi

    def calc_jacobian(self):
        """
        Calculate the Jacobian matrix.

        Reference: Section 5.2.1
        On Inexact Newton Methods for Inverse Problems in Banach Spaces, 2015.
        Fabio Margotti
        https://publikationen.bibliothek.kit.edu/1000048606
        """
        I2_all = []  # Construct a new current pattern for Jacobian calculation.
        for i in range(self.L):
            # I2_i = 1 at electrode i and zero otherwise
            I2 = np.zeros(self.L)
            I2[i] = 1
            I2_all.append(I2)

        u_arr = self.u_arr
        bu_arr, _ = self.solve_forward(self.V, I2_all, self.gamma_n)

        # Separating electrodes data
        select_data = np.tile(range(self.L), self.l)
        select_data = np.split(
            select_data, np.where(select_data[:-1] == self.L - 1)[0] + 1
        )

        Q_DG = VectorFunctionSpace(self.mesh, "DG", 0, dim=2)
        grad_u = np.array(
            [project(grad(u), Q_DG).vector()[:].reshape(-1, 2) for u in u_arr]
        )
        grad_bu = np.array(
            [project(grad(bu), Q_DG).vector()[:].reshape(-1, 2) for bu in bu_arr]
        )

        jac_all = []
        for h in range(self.l):  # For each experiment
            deriv = np.array(
                [
                    -np.sum(
                        grad_bu[j] * grad_u[h], axis=1
                    )  # Get the function value in each element
                    for j in select_data[h]  # For each electrode
                ]
            )

            jac = deriv * self.cell_arr  # Matrix * Volume_cell

            for row in jac:
                jac_all.append(row)

        jac_all = np.array(jac_all)
        return jac_all

    def weight_func(self, jac):
        """
        Determine the weights for the Jacobian matrix and apply them.

        The weights are used to (possibly) improve the convergence of the Newton-like method.
        """
        # norm(jac_cols) * 1 / vol_cell_n * 1 / gamma_cell_n
        w = np.linalg.norm(jac, axis=0) * (1 / self.cell_arr) * (1 / self.Cellsgamma_n)
        return w

    def calc_tolerance(self, norm_res, *, constant_value):
        """
        Determine the tolerance parameter mu for the Newton inner step.

        Reference: Section 6
        On the regularization of nonlinear ill-posed problems via inexact Newton iterations, 1999.
        Andreas Rieder
        http://dx.doi.org/10.1088/0266-5611/15/1/028
        """
        if constant_value is not None:
            self.mu = constant_value
            return constant_value

        step = self.step
        inner_steps = self.inner_step_list

        if step < 2:
            mu_til = self.mu_start
        elif inner_steps[step - 1] >= inner_steps[step - 2]:
            mu_til = 1 - (inner_steps[step - 2] / inner_steps[step - 1]) * (1 - self.mu)
        else:
            mu_til = self.mu_theta * self.mu

        th = self.tau * self.noise_level
        mu = self.mu_max * np.maximum(th / norm_res, mu_til)

        self.mu = mu
        return mu

    def norm_funcLp(self, func_arr, domain, *, ord=2):
        """
        Computes the Lp norm of a function for the given domain ("Omega" or "dOmega").
        """
        p = ord
        w = self.weight

        if domain == "Omega":
            dx = self.cell_arr
            norm = np.sum(np.abs(func_arr) ** p * w * dx)
        elif domain == "dOmega":
            ds = self.size_elec_vec
            norm = np.sum(np.abs(func_arr) ** p * ds)

        norm **= 1 / p
        return norm

    def set_initial_guess(self, x0):
        self.initial_guess_primal = x0
        self.Cellsgamma_n = x0
        self.gamma_n = Function(self.Q_DG)
        self.gamma_n.vector()[:] = x0

        if self.inner_method in ["TV1", "TV2"]:
            L = self.__assemble_L()
            self.TV_L = L

        self.initial_guess_dual = self.__primal_to_dual(x0)

    def __assemble_L(self):
        mesh = self.mesh
        D = mesh.topology().dim()
        mesh.init()  # Build connectivity between facets and cells
        cellsnum = mesh.num_cells()

        L = []
        for edge in edges(mesh):
            cells_in_edge = edge.entities(D)
            Lrow = np.zeros(cellsnum)
            if len(cells_in_edge) == 2:  # Only interior edges
                Lrow[cells_in_edge[0]], Lrow[cells_in_edge[1]] = (
                    +edge.length(),
                    -edge.length(),
                )
                L.append(Lrow)

        L = csr_matrix(L)
        return L

    def __primal_to_dual(self, x):
        p = self.Lp_space

        if self.inner_method == "Lp":
            if p == 2:
                return np.copy(x)
            return _dm(x, p)

        elif self.inner_method == "Sparse1":
            beta = self.penalty_beta

            if p == 2:
                return beta * _subL1(x) + x
            return beta * _subL1(x) + _dm(x, p)

        elif self.inner_method == "Sparse2":
            beta = self.penalty_beta

            if p == 2:
                return _subL1(x) + (1 / beta) * x
            return _subL1(x) + (1 / beta) * _dm(x, p)

        elif self.inner_method == "TV1":
            beta = self.penalty_beta

            L = self.TV_L
            grad_TV_x = self.__grad_TV(x, L)

            if p == 2:
                return beta * grad_TV_x + x
            return beta * grad_TV_x + _dm(x, p)

        elif self.inner_method == "TV2":
            beta = self.penalty_beta

            L = self.TV_L
            grad_TV_x = self.__grad_TV(x, L)

            if p == 2:
                return grad_TV_x + (1 / beta) * x
            return grad_TV_x + (1 / beta) * _dm(x, p)

    def set_Lebesgue(self, p, r):
        self.Lp_space = p
        self.Lr_space = r

    def set_penalty_beta(self, beta):
        self.penalty_beta = beta

    def set_Newton_parameters(self, **kwargs):
        """
        Set parameters to determine the tolerance parameter mu for the Newton inner step.

        Parameters
        ----------
        mu_start : float, optional
            Used to compute the first two values of mu.
            Default is 0.3.
        mu_max : float, optional
            Max value of mu in (0,1].
            Default is 0.999.
        mu_theta : float, optional
            Decreasing factor for last mu.
            Default is 0.9.
        mu_constant : float, optional
            Use constant value for mu across all iterations.
            Default is None.
        """
        valid_attributes = {
            "mu_start",
            "mu_max",
            "mu_theta",
            "mu_constant",
        }
        for key, value in kwargs.items():
            if key not in valid_attributes:
                raise ValueError(f"{key} is not a valid attribute")
            setattr(self, key, value)

    def set_noise_parameters(self, tau, noise_level):
        """
        Set noise parameters for stopping rule (discrepancy principle).
        """
        self.tau = tau
        self.noise_level = noise_level

    def set_solverconfig(self, **kwargs):
        """
        Set solver configuration for the inverse problem.

        Parameters
        ----------
        step_limit : int, optional
            Maximum number of steps (outer iteration).
            Default is 20.
        use_weight : bool, optional
            To use (or not) a weight function in the Jacobian matrix.
            Default is True.
        min_v : float, optional
            Minimal value for `gamma_n` elements.
            Default is 0.005.
        verbose : bool, optional
            Print infos along the iterations.
            Default is True.
        """
        valid_attributes = {
            "step_limit",
            "use_weight",
            "min_v",
            "verbose",
        }
        for key, value in kwargs.items():
            if key not in valid_attributes:
                raise ValueError(f"{key} is not a valid attribute")
            setattr(self, key, value)

    def set_inner_solverconfig(self, **kwargs):
        """
        Set solver configuration for the inner iteration.

        Parameters
        ----------
        inner_method : string, optional
            Set which method to use for solving the linearized problem.
            Valid options are ["Lp", "Sparse1", "Sparse2", "TV1", "TV2"].
            Currently, all options implement a gradient method, but with different penalty terms.
            Default is "Lp".
        ME_reg : float, optional
            Small constant value used to compute the Minimal Error step for gradient method.
            Default is 15e-4.
        inner_step_limit : int, optional
            Maximum number of steps (inner iteration).
            Default is 50.
        use_constant_inner_limit : bool, optional.
            Keep `inner_step_limit` constant across all iterations.
            Default is True.
        inner_step_start : int, optional
            Maximum number of steps for the first two iterations.
            Used only if `use_constant_inner_limit` is `False`.
            Default is 500.
        """
        valid_attributes = {
            "inner_method",
            "ME_reg",
            "inner_step_limit",
            "use_constant_inner_limit",
            "inner_step_start",
        }
        for key, value in kwargs.items():
            if key not in valid_attributes:
                raise ValueError(f"{key} is not a valid attribute")
            setattr(self, key, value)


def _dm(x, p):  # duallity mapping
    return np.sign(x) * np.abs(x) ** (p - 1)


def _subL1(x):
    g = np.sign(x)
    mask = np.isclose(x, 0)
    g[mask] = 0  # any value in [-1, 1]
    return g


def _minacc(
    fgrad: callable,
    x0: np.ndarray,
    *,
    s=1e-4,
    tol=1e-6,
    max_iter=100,
    ord=2,
):
    """
    Implement Nesterov's Accelerated Gradient method.
    Form II, Algorithm 12 from: https://arxiv.org/abs/2101.09545
    """
    yt = x0
    xt = x0

    At = 0.0
    at = 1.0

    t = 0
    while t < max_iter:
        gt = fgrad(yt)

        xnew = yt - s * gt  # Update position

        # Convergence check
        xdiff = xnew - xt
        if np.linalg.norm(xdiff, ord=ord) / np.linalg.norm(xt, ord=ord) <= tol:
            xt = xnew  # Update before exiting
            t += 1
            break

        At += at
        anew = 0.5 * (1 + np.sqrt(4 * At + 1))

        yt = xnew + ((at - 1) / anew) * xdiff
        xt = xnew

        at = anew

        t += 1

    return xt
