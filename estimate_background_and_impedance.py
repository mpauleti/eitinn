import feit


def main() -> None:
    # Mesh parameters
    resolution = 26
    num_elec_vertices = 20
    num_gap_vertices = 8

    # Defining mesh
    elec_mesh = feit.msd.get_tank_mesh(resolution, num_elec_vertices, num_gap_vertices)

    U0_bg, I_all = feit.msd.get_data_from_experiment("1_0")

    print("Estimating...")
    cond, z = feit.msd.estimate_cond_iter(U0_bg, I_all, elec_mesh)

    print(f"Background conductivity estimation: {cond:.15f}")
    print(f"Contact impedance estimation: {z:.15f}")


if __name__ == "__main__":
    main()
