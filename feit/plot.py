import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Arc
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator


from fenics import FunctionSpace, Function
from fenics import plot as fxplot


# Define possible markers, colors and linestyles
markers_options = ["o", "s", "P", "v", "^", "<", ">", "d", "h", "*"]
colors_options = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
linestyles_options = ["dashed", "dashdot", "dotted", "solid"]


def plot_function(
    function,
    elec_mesh,
    *,
    title="",
    fontsize=16,
    linewidth_mesh=0.1,
    figsize=(5, 5),
    cmap="turbo",
    bar_axes=[0.9015, 0.2205, 0.0175, 0.5605],
    axis=False,
    save=False,
    filename="figure.pdf",
):
    fig, ax = plt.subplots(figsize=figsize)

    p = fxplot(function)
    fxplot(elec_mesh, linewidth=linewidth_mesh)

    p.set_cmap(cmap)
    fig.colorbar(p, cax=fig.add_axes(bar_axes), orientation="vertical")

    ax.set_title(title, fontsize=fontsize)

    if not axis:
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        ax.set_frame_on(False)  # Disable the frame

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return


def plot_function_with_phantom(
    function,
    elec_mesh,
    exp_case,
    *,
    title="",
    fontsize=14,
    linewidth_mesh=0.1,
    single_figsize=5,
    cmap="turbo",
    bar_axes=[0.9015, 0.2205, 0.0175, 0.5605],
    axis=False,
    save=False,
    filename="figure.pdf",
):
    nr, nc = 1, 2
    figsize = (nc * single_figsize, single_figsize)

    ax: tuple[Axes, ...]
    fig, ax = plt.subplots(nrows=nr, ncols=nc, figsize=figsize)
    ax0, ax1 = ax

    ## Phantom first
    photo_name = get_file_path(f"eit_data/target_photos/fantom_{exp_case}.jpg")
    img = mpimg.imread(photo_name)
    ax0.imshow(img)
    ax0.set_title("Inclusions", fontsize=fontsize)

    ## Reconstruction
    plt.sca(ax1)  # Set 'ax1' as the active axis

    p = fxplot(function)
    fxplot(elec_mesh, linewidth=linewidth_mesh)

    p.set_cmap(cmap)
    fig.colorbar(p, cax=fig.add_axes(bar_axes), orientation="vertical")

    ax1.set_title(title, fontsize=fontsize)

    if not axis:
        ax0.set_xticks([])  # Remove x ticks
        ax1.set_xticks([])

        ax0.set_yticks([])  # Remove y ticks
        ax1.set_yticks([])

        ax0.set_frame_on(False)  # Disable the frame
        ax1.set_frame_on(False)

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return


def plot_reconstructions(
    reconstructions,
    elec_mesh,
    exp_case,
    *,
    mesh_display="thin",  # "nil", "thin" or "thick"
    linewidth_mesh=0.1,  # only used for "thick" display
    fontsize=14,
    colorbar_display="individual",  # "individual" or "aio" (all in one)
    colorbar_fontsize=10,
    single_figsize=5,
    cmap="turbo",
    axis=False,
    save=False,
    filename="figure.pdf",
):
    if colorbar_display == "aio":
        clim_min = np.min([np.min(rec) for rec in reconstructions])
        clim_max = np.max([np.max(rec) for rec in reconstructions])

    nr, nc = 1, len(reconstructions) + 1
    figsize = (nc * single_figsize, single_figsize)

    ax: tuple[Axes, ...]
    fig, ax = plt.subplots(nrows=nr, ncols=nc, figsize=figsize)

    ## Phantom first
    ax0 = ax[0]
    photo_name = get_file_path(f"eit_data/target_photos/fantom_{exp_case}.jpg")
    img = mpimg.imread(photo_name)
    ax0.imshow(img)
    ax0.set_title("Inclusions", fontsize=fontsize)

    if not axis:
        ax0.set_xticks([])  # Remove x ticks
        ax0.set_yticks([])  # Remove y ticks
        ax0.set_frame_on(False)  # Disable the frame

    ## Reconstructions
    Q_DG = FunctionSpace(elec_mesh, "DG", 0)
    for i, reconstruction in enumerate(reconstructions, 1):
        axi = ax[i]
        plt.sca(axi)  # Set 'axi' as the active axis

        ## Transform vector to function
        func = Function(Q_DG)
        func.vector()[:] = reconstruction
        ##

        if mesh_display == "nil":
            p = fxplot(func)
            p.set_rasterized(True)
        if mesh_display == "thin":
            p = fxplot(func)
        if mesh_display == "thick":
            p = fxplot(func)
            fxplot(elec_mesh, linewidth=linewidth_mesh)

        if colorbar_display == "individual":
            p.set_cmap(cmap)

            # Create a new colorbar for each subplot
            cbar = fig.colorbar(
                p, ax=axi, orientation="vertical", fraction=0.046, pad=0.04
            )

            cbar.outline.set_visible(True)  # Ensure the outline is visible
            cbar.ax.yaxis.set_ticks_position("right")  # Set tick position
            cbar.ax.yaxis.set_visible(True)  # Ensure axis is visible
            cbar.ax.tick_params(labelsize=colorbar_fontsize)  # Adjust tick label size
            # cbar.update_ticks()  # Refresh the ticks

        if colorbar_display == "aio":
            p.set_cmap(cmap)
            p.set_clim(clim_min, clim_max)

            if i == len(reconstructions):
                # Create a colorbar only for the last subplot
                cbar = fig.colorbar(
                    p, ax=axi, orientation="vertical", fraction=0.046, pad=0.04
                )

                cbar.outline.set_visible(True)  # Ensure the outline is visible
                cbar.ax.yaxis.set_ticks_position("right")  # Set tick position
                cbar.ax.yaxis.set_visible(True)  # Ensure axis is visible
                cbar.ax.tick_params(
                    labelsize=colorbar_fontsize
                )  # Adjust tick label size
                # cbar.update_ticks()  # Refresh the ticks

        axi.set_title(f"Method {i}", fontsize=fontsize)

        if not axis:
            axi.set_xticks([])  # Remove x ticks
            axi.set_yticks([])  # Remove y ticks
            axi.set_frame_on(False)  # Disable the frame

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return


def plot_residuals(
    residual_list,
    *,
    title="",
    figsize=(5, 5),
    fontsize=14,
    tick_fontsize=12,
    legend_fontsize=12,
    linewidth=2,
    save=False,
    filename="residuals.pdf",
):
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_title(title, fontsize=fontsize)
        ax.set_ylabel(r"Nonlinear Residual $(\%)$", fontsize=fontsize)
        ax.set_xlabel("Iteration", fontsize=fontsize)

        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

        # min_len = min(map(len, residual_list))
        max_len = max(map(len, residual_list))

        # xtick_positions = np.linspace(0, max_len, num_xticks, dtype=int)
        # xtick_positions = np.unique(xtick_positions)

        # ax.set_xticks(xtick_positions)

        iter_mult = _calc_iter_mult(max_len)
        fig.gca().xaxis.set_major_locator(MultipleLocator(iter_mult))

        if len(residual_list) == 3:  # Hardcoded for 3 methods...
            ax.plot(
                residual_list[0],
                linestyle="dashed",
                marker="P",
                color="blue",
                linewidth=linewidth,
                label="Method 1",
            )
            ax.plot(
                residual_list[1],
                linestyle="dashdot",
                marker="o",
                color="orange",
                linewidth=linewidth,
                label="Method 2",
            )
            ax.plot(
                residual_list[2],
                linestyle="dotted",
                marker="v",
                color="purple",
                linewidth=linewidth,
                label="Method 3",
            )
        else:
            for i, residual in enumerate(residual_list, 1):
                ax.plot(
                    residual,
                    linestyle=np.random.choice(linestyles_options),
                    marker=np.random.choice(markers_options),
                    color=np.random.choice(colors_options),
                    linewidth=linewidth,
                    label=f"Method {i}",
                )

        ax.legend(fontsize=legend_fontsize, frameon=True, facecolor="white")

        if save:
            fig.savefig(filename, dpi=300, bbox_inches="tight")

    return


def _calc_iter_mult(max_len):
    iter_mult = 1

    if 5 < max_len <= 10:
        iter_mult = 2
    if 10 < max_len <= 20:
        iter_mult = 3
    if max_len > 20:
        iter_mult = 5

    return iter_mult


def plot_electrodes_mesh(
    elec_mesh,
    *,
    linewidth_mesh=0.5,
    linewidth_elec=5,
    figsize=(5, 5),
    fontsize=10,
    elec_num=True,
    axis=False,
    save=False,
    filename="electrodes_mesh.pdf",
):
    fig, ax = plt.subplots(figsize=figsize)

    radius = elec_mesh.radius
    theta_vec = np.degrees(
        np.array(elec_mesh.electrodes.position)
    )  # Convert angles from radians to degrees.

    for index, theta in enumerate(theta_vec):
        theta_start, theta_end = theta[0], theta[1]
        theta_center = (
            (np.abs(theta_start - theta_end) / 2 + theta_start) / 360 * (2 * np.pi)
        )

        # Plotting arc
        arc = Arc(
            (0, 0),
            2 * radius * 1.01,
            2 * radius * 1.01,
            angle=0,
            theta1=theta_start,
            theta2=theta_end,
            linewidth=linewidth_elec,
            color="black",
        )
        ax.add_artist(arc)

        # Plotting electrode number
        if elec_num:
            x, y = (
                radius * np.cos(theta_center) * 1.1,
                radius * np.sin(theta_center) * 1.1,
            )
            ax.annotate(
                index + 1,
                (x, y),
                color="black",
                weight="bold",
                fontsize=fontsize,
                ha="center",
                va="center",
            )

    ax.set_aspect("equal")  # Enforce equal aspect ratio
    fxplot(elec_mesh, linewidth=linewidth_mesh)

    if not axis:
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        ax.set_frame_on(False)  # Disable the frame

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return


def plot_electrodes_mesh_with_tank(
    elec_mesh,
    *,
    linewidth_mesh=0.5,
    linewidth_elec=3,
    single_figsize=5,
    fontsize=6,
    elec_num=True,
    axis=False,
    save=False,
    filename="electrodes_mesh_tank.pdf",
):
    nr, nc = 1, 2
    figsize = (nc * single_figsize, single_figsize)

    ax: tuple[Axes, ...]
    fig, ax = plt.subplots(nrows=nr, ncols=nc, figsize=figsize)
    ax0, ax1 = ax

    ## Tank first
    photo_name = get_file_path(f"eit_data/target_photos/fantom_1_0.jpg")
    img = mpimg.imread(photo_name)
    ax0.imshow(img)

    ## Mesh
    plt.sca(ax1)  # Set 'ax1' as the active axis

    radius = elec_mesh.radius
    theta_vec = np.degrees(
        np.array(elec_mesh.electrodes.position)
    )  # Convert angles from radians to degrees.

    for index, theta in enumerate(theta_vec):
        theta_start, theta_end = theta[0], theta[1]
        theta_center = (
            (np.abs(theta_start - theta_end) / 2 + theta_start) / 360 * (2 * np.pi)
        )

        # Plotting arc
        arc = Arc(
            (0, 0),
            2 * radius * 1.01,
            2 * radius * 1.01,
            angle=0,
            theta1=theta_start,
            theta2=theta_end,
            linewidth=linewidth_elec,
            color="black",
        )
        ax1.add_artist(arc)

        # Plotting electrode number
        if elec_num:
            x, y = (
                radius * np.cos(theta_center) * 1.1,
                radius * np.sin(theta_center) * 1.1,
            )
            ax1.annotate(
                index + 1,
                (x, y),
                color="black",
                weight="bold",
                fontsize=fontsize,
                ha="center",
                va="center",
            )

    ax1.set_aspect("equal")  # Enforce equal aspect ratio
    fxplot(elec_mesh, linewidth=linewidth_mesh)

    if not axis:
        ax0.set_xticks([])  # Remove x ticks
        ax1.set_xticks([])

        ax0.set_yticks([])  # Remove y ticks
        ax1.set_yticks([])

        ax0.set_frame_on(False)  # Disable the frame
        ax1.set_frame_on(False)

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return


def get_file_path(path):
    # Get the directory where this function is defined
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to a file relative to this directory
    file_path = os.path.join(current_dir, path)

    return file_path
