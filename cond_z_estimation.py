import FEIT


def main() -> None:
    ## Mesh Parameters
    resolution = 26  # Mesh resolution
    N_in = 20  # Number of vertex on electrodes
    N_out = 8  # Number of vertex on gaps

    ## Defining mesh
    elec_mesh = FEIT.msd.get_tank_mesh(resolution, N_in, N_out)

    U0_bg, I_all = FEIT.msd.getdata_from_experiment("1_0")

    print("Estimating...")
    cond, z = FEIT.msd.estimate_cond_iter(U0_bg, I_all, elec_mesh)

    print(f"cond estimation: {cond:.15f}")
    print(f"z estimation: {z:.15f}")

    return


if __name__ == "__main__":
    main()
