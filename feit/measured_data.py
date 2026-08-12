from pathlib import Path

import numpy as np
import scipy.io as spio
import scipy.optimize
from fenics import FiniteElement, Function, FunctionSpace

from . import mesh as fmesh
from .forward_problem import ForwardProblem

# CONSTANTS
# Estimation results with
#   resolution        = 26
#   num_elec_vertices = 20
#   num_gap_vertices  = 8
# For other values, you may run `estimate_background_and_impedance.py`.
BACK_COND = 0.927201922207184  # Background conductivity
Z_IMP = 0.011067821315443  # Impedance of each electrode


def get_tank_mesh(resolution, num_elec_vertices, num_gap_vertices):
    # Basic definitions
    radius = 14.0  # Tank radius: 14cm
    num_electrodes = 16
    elec_length = 2.5  # 2.5cm
    coverage_fraction = (num_electrodes * elec_length) / (2 * np.pi * radius)
    rotation_angle = np.pi / 2 - 0.5 * elec_length / radius

    # Return object with angular position of each electrode
    electrodes = fmesh.Electrodes(
        num_electrodes,
        coverage_fraction,
        rotation_angle,
        anticlockwise=False,
    )

    # Defining mesh
    elec_mesh = fmesh.generate_electrodes_mesh(
        radius,
        resolution,
        num_elec_vertices,
        num_gap_vertices,
        electrodes,
    )
    return elec_mesh


def get_data_from_experiment(exp_case):
    mat = spio.loadmat(_get_filepath(f"eit_data/data_mat_files/datamat_{exp_case}.mat"))
    Uel = mat.get("Uel").T
    CP = mat.get("CurrentPattern").T

    # Selecting potentials
    Uel_f = Uel[-15:]  # Matrix of measurements

    # Current
    I_all = CP[-15:] / np.sqrt(2)

    # Converting potentials to ground pattern
    U0_m = np.zeros_like(Uel_f)
    for i, U_potential in enumerate(Uel_f):
        U0_m[i] = convert_data(U_potential)

    U_delta = U0_m.flatten()

    return U_delta, I_all


def convert_data(U):
    """Convert data from different measurement patterns to the ground pattern."""
    # See: https://arxiv.org/abs/1704.01178
    num_electrodes = len(U)

    U_til = np.zeros(num_electrodes)
    for i in range(1, num_electrodes):
        U_til[i] = np.sum(U[:i])

    c = np.sum(U_til)
    return c / num_electrodes - U_til


def calc_noise_level(U_noised: np.ndarray, I: np.ndarray, *, ord_="fro") -> float:
    """
    Estimate the noise level in potential measurements obtained
    from a grounded electrode system.

    Reference:
    A model-aware inexact Newton scheme for electrical impedance
    tomography, 2016.
    Robert Winkler
    https://publikationen.bibliothek.kit.edu/1000054135
    """
    num_patterns, num_electrodes = np.shape(I)

    U_noised = U_noised.reshape(num_patterns, num_electrodes)
    Iplus = np.linalg.pinv(I)

    Ev = (Iplus @ U_noised) - (Iplus @ U_noised).T

    norm_Ev = np.linalg.norm(Ev, ord=ord_)
    norm_Iplus = np.linalg.norm(Iplus, ord=ord_)

    vCEM = norm_Ev**2 / (2 * (num_electrodes - 1)) * norm_Iplus ** (-2)

    delta = np.sqrt(num_patterns * num_electrodes * vCEM)

    noise_level = delta / np.linalg.norm(U_noised, ord=ord_)
    return noise_level


def estimate_cond_iter(U0, I_all, elec_mesh, z=None, *, zmin=1e-4, ord_=2):
    Q_DG = FunctionSpace(elec_mesh, "DG", 0)
    # Solution space CG (Continuous Galerkin)
    VD = FiniteElement("CG", elec_mesh.ufl_cell(), 1)

    if z is None:
        _, num_electrodes = np.shape(I_all)
        z = np.full(num_electrodes, zmin)

    # Define the functional
    def func(zi, U0, I_all, elec_mesh, z, ord_):
        cond, z = estimate_cond(U0, I_all, elec_mesh, zi)
        z = np.maximum(zmin, z)

        gamma = Function(Q_DG)
        gamma.vector()[:] = np.full(elec_mesh.num_cells(), cond, dtype=float)

        fp = ForwardProblem(elec_mesh, z)

        _, U_arr = fp.solve_forward(VD, I_all, gamma)
        U_arr = U_arr.flatten()

        residual = (
            np.linalg.norm(U_arr - U0, ord=ord_) / np.linalg.norm(U0, ord=ord_) * 100
        )
        return residual

    result = scipy.optimize.minimize(
        func,
        zmin,
        method="BFGS",
        args=(U0, I_all, elec_mesh, z, ord_),
    )
    z = result["x"]

    cond, z = estimate_cond(U0, I_all, elec_mesh, z)
    return cond, z


def estimate_cond(U0, I_all, elec_mesh, z, *, is_kuopio_tank=True):
    """
    Estimate the conductivity of the background based on noisy voltage measurements.
    Returns a tuple with conductivity and potential estimates.
    """
    Q_DG = FunctionSpace(elec_mesh, "DG", 0)
    # Solution space CG (Continuous Galerkin)
    VD = FiniteElement("CG", elec_mesh.ufl_cell(), 1)

    num_patterns, num_electrodes = np.shape(I_all)
    U0 = U0.reshape(num_patterns, num_electrodes)

    gamma = Function(Q_DG)
    gamma.vector()[:] = np.ones(elec_mesh.num_cells())

    fp = ForwardProblem(elec_mesh, z)
    _, U = fp.solve_forward(VD, I_all, gamma)

    z0 = np.max(z)

    if is_kuopio_tank:
        elec_length = 2.5
    else:
        elec_length = elec_mesh.radius * (
            elec_mesh.electrodes.position[0][1] - elec_mesh.electrodes.position[0][0]
        )

    a_vec = np.array(
        [
            np.dot(U[i] - I_all[i] * z0 / elec_length, I_all[i])
            for i in np.arange(num_patterns)
        ]
    )
    b_vec = np.array(
        [np.dot(I_all[i], I_all[i] / elec_length) for i in np.arange(num_patterns)]
    )
    c_vec = np.array([np.dot(U0[i], I_all[i]) for i in np.arange(num_patterns)])

    A = np.array([a_vec, b_vec]).T

    result = scipy.optimize.nnls(A, c_vec, maxiter=20)
    rho, z = result[0]
    cond = 1 / rho
    return cond, z


def _get_filepath(path):
    # Get the directory containing this file
    current_dir = Path(__file__).resolve().parent
    # Construct the path to a file relative to this directory
    filepath = str(current_dir / path)

    return filepath
