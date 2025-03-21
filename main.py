import FEIT
import FEIT.plot as fplot

import os
import sys
import functools
from time import perf_counter
import numpy as np


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reconstructions", exist_ok=True)
    os.makedirs("residuals", exist_ok=True)

    # Mesh Parameters
    resolution = 26  # Mesh resolution
    N_in = 20  # Number of vertex on electrodes
    N_out = 8  # Number of vertex on gaps

    # Defining mesh
    elec_mesh = FEIT.msd.get_tank_mesh(resolution, N_in, N_out)

    FEIT.mesh.print_mesh_config(elec_mesh)

    ## Plotting mesh
    fplot.plot_electrodes_mesh(elec_mesh, save=True)
    fplot.plot_electrodes_mesh_with_tank(elec_mesh, save=True)

    #### Defining impedances, experiments and currents
    background_value = FEIT.msd.BACK_COND
    z = FEIT.msd.Z_IMP
    ####

    ## Iteration limits
    step_limit = 20
    inner_step_limit = 300

    exp_cases = ["2_3", "4_1", "4_4"]
    methods = ["Lp|p=2", "TV1|p=2", "TV1|p=1.1"]

    startALL = perf_counter()
    for exp_case in exp_cases:
        ## Load experimental data
        U_delta, I_all = FEIT.msd.getdata_from_experiment(exp_case)

        noise_level = FEIT.msd.calc_noise_level(U_delta, I_all)
        print(f"Noise level (%): {noise_level * 100:.6f}")

        gamma_recs = []
        residual_list = []
        for method in methods:
            inner_method, pp = method.split("|")
            pp = float(pp.split("=")[1])

            #### EIT
            IP = FEIT.IP.InverseProblem(elec_mesh, U_delta, I_all, z)

            ## Solver Parameters
            IP.set_solverconfig(step_limit=step_limit)
            IP.set_inner_solverconfig(
                inner_method=inner_method, inner_step_limit=inner_step_limit
            )
            IP.set_Lebesgue(pp, 2)
            IP.set_penalty_beta(5)

            ## First step
            gamma_background = np.full(
                elec_mesh.num_cells(), background_value, dtype=float
            )
            IP.set_initial_guess(gamma_background)

            ## Noise Parameters
            tau = 1.5
            IP.set_noise_parameters(tau, noise_level)

            ## Solver
            space_name = "hilbert" if pp == 2 else "banach"
            filename = f"logs/log_{exp_case}_{inner_method}_{space_name}.txt"

            solve_IP_with_log(IP, filename)

            gamma_rec = np.copy(IP.gamma_all[-1])
            gamma_recs.append(gamma_rec)
            residual_list.append(IP.res_list)

        fplot.plot_reconstructions(
            gamma_recs,
            elec_mesh,
            exp_case,
            save=True,
            filename=f"reconstructions/recs_{exp_case}_all.pdf",
            cmap="turbo",
            mesh_display="nil",
            colorbar_display="aio",
        )

        fplot.plot_reconstructions(
            gamma_recs,
            elec_mesh,
            exp_case,
            save=True,
            filename=f"reconstructions/recs_{exp_case}_all_ind.pdf",
            cmap="turbo",
            mesh_display="nil",
            colorbar_display="individual",
        )

        fplot.plot_residuals(
            residual_list,
            save=True,
            filename=f"residuals/residuals_{exp_case}.pdf",
        )

    endALL = perf_counter()
    print(f"Elapsed time (ALL): {endALL - startALL:.4f}")

    return


def solve_IP_with_log(IP, filename):
    IP.solve_inverse = logger(filename)(IP.solve_inverse)
    IP.solve_inverse()

    msg = f"Sum of all inner steps: {np.sum(IP.inner_step_list)}"
    print(msg)
    with open(filename, "a") as f:
        f.write(msg)
        f.write("\n")


class Tee:
    """A class that duplicates output to both stdout and a file."""

    def __init__(self, filename, *, mode="w"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout  # Store original stdout

    def write(self, message):
        self.stdout.write(message)  # Write to console
        self.file.write(message)  # Write to file

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def logger(filename="log.txt", *, mode="w"):
    """Decorator to log print output to a file while still printing to console."""

    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            original_stdout = sys.stdout  # Save the original stdout
            tee = Tee(filename, mode=mode)

            sys.stdout = tee  # Redirect stdout to our Tee class
            try:
                start = perf_counter()
                result = func(*args, **kwargs)
                end = perf_counter()
                print(f"Elapsed time: {end - start:.4f}")
            finally:
                sys.stdout = original_stdout  # Restore original stdout
                tee.close()  # Close the file

            return result

        return inner

    return decorator


if __name__ == "__main__":
    main()
