import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Arc
from matplotlib.ticker import MultipleLocator

from typing import Literal

from fenics import FunctionSpace, Function
from fenics import plot as fxplot


def plot_functions(
    funcs,
    *,
    fmesh=None,
    titles=None,
    fontsize=14,
    single_figsize=(4.8, 4.8),
    fig_max_cols=5,
    mesh_display: Literal["nil", "thin", "thick"] = "nil",
    linewidth_mesh=0.5,  # only used when mesh_display == "thick"
    with_colorbar=True,
    colorbar_fontsize=10,
    colorbar_display: Literal["ind", "all"] = "ind",
    cmap="turbo",
    with_phantom=False,
    exp_case="",
    with_axis=False,
    save=False,
    filename="functions.pdf",
    close_fig=True,
):
    try:
        NFUNC = len(funcs)
    except:
        NFUNC = 1
        funcs = [funcs]

    if isinstance(titles, str):
        titles = [titles]

    if titles is None:
        NT = NFUNC + 1 if with_phantom else NFUNC
        titles = ["" for _ in range(NT)]

    nr, nc = (1, NFUNC + 1) if with_phantom else (1, NFUNC)
    nr, nc = _adjust_dim((nr, nc), fig_max_cols)

    w, h = single_figsize
    figsize = (nc * w, nr * h)

    fig, ax = plt.subplots(nrows=nr, ncols=nc, figsize=figsize)
    fig.tight_layout()

    axs: tuple[plt.Axes, ...]
    axs = (ax,) if isinstance(ax, plt.Axes) else tuple(ax.flatten())

    if with_phantom:
        ax0 = axs[0]  ## Phantom first

        photo_name = _get_filepath(f"eit_data/target_photos/fantom_{exp_case}.jpg")
        img = mpimg.imread(photo_name)
        ax0.imshow(img)
        ax0.set_title(titles[0], fontsize=fontsize)

    if with_colorbar and colorbar_display == "all":
        clim_min = np.min([func.vector().get_local().min() for func in funcs])
        clim_max = np.max([func.vector().get_local().max() for func in funcs])

    istart = 1 if with_phantom else 0
    for i, func in enumerate(funcs, istart):
        axi = axs[i]
        plt.sca(axi)  # Set 'axi' as the active axis

        func_mesh = fmesh if fmesh is not None else func.function_space().mesh()

        p = _plot_mesh_handler(
            func,
            func_mesh,
            mesh_display,
            linewidth_mesh,
        )
        p.set_cmap(cmap)

        if not with_colorbar:
            pass  # Nothing to do, skip all colorbar-related logic
        elif colorbar_display == "ind":
            _create_colorbar(p, fig, axi, colorbar_fontsize=colorbar_fontsize)
        elif colorbar_display == "all":
            p.set_clim(clim_min, clim_max)

            last_subplot = i == NFUNC + istart - 1
            if last_subplot:
                # Create a common colorbar
                _create_colorbar(
                    p, fig, ax=axs, colorbar_fontsize=colorbar_fontsize, pad=0.01
                )

        axi.set_title(titles[i], fontsize=fontsize)

    if not with_axis:
        _disable_axis(axs)

    if save:
        _save_fig(fig, filename)

    if close_fig:
        plt.close(fig)

    return


def _adjust_dim(dim, max_cols):
    r, c = dim
    if c > max_cols:
        total = r * c
        c = max_cols
        while r * c < total:
            r += 1
    return r, c


def plot_reconstructions(
    reconstructions,
    elec_mesh,
    *,
    titles=None,
    fontsize=14,
    single_figsize=(4.8, 4.8),
    fig_max_cols=5,
    mesh_display: Literal["nil", "thin", "thick"] = "nil",
    linewidth_mesh=0.5,  # only used when mesh_display == "thick"
    with_colorbar=True,
    colorbar_fontsize=10,
    colorbar_display: Literal["ind", "all"] = "ind",
    cmap="turbo",
    with_phantom=False,
    exp_case="",
    with_axis=False,
    save=False,
    filename="reconstructions.pdf",
    close_fig=True,
):
    ## Transform vectors to functions
    funcs = []
    Q_DG = FunctionSpace(elec_mesh, "DG", 0)
    for reconstruction in reconstructions:
        func = Function(Q_DG)
        func.vector()[:] = reconstruction

        funcs.append(func)

    plot_functions(
        funcs,
        fmesh=elec_mesh,
        titles=titles,
        fontsize=fontsize,
        single_figsize=single_figsize,
        fig_max_cols=fig_max_cols,
        mesh_display=mesh_display,
        linewidth_mesh=linewidth_mesh,
        with_colorbar=with_colorbar,
        colorbar_fontsize=colorbar_fontsize,
        colorbar_display=colorbar_display,
        cmap=cmap,
        with_phantom=with_phantom,
        exp_case=exp_case,
        with_axis=with_axis,
        save=save,
        filename=filename,
        close_fig=close_fig,
    )
    return


# Define possible markers, colors and linestyles
_markers = ["P", "o", "v", "s"]
_colors = ["blue", "orange", "purple", "green"]
_linestyles = ["dashed", "dashdot", "dotted", "solid"]

_styles = list(zip(_linestyles, _markers, _colors))


def _calc_iter_mult(max_len):
    iter_mult = 1

    if 5 < max_len <= 10:
        iter_mult = 2
    if 10 < max_len <= 20:
        iter_mult = 3
    if max_len > 20:
        iter_mult = 5

    return iter_mult


def plot_residuals(
    residuals,
    *,
    title="",
    fontsize=14,
    tick_fontsize=12,
    legend_fontsize=12,
    figsize=(6.4, 4.8),
    linewidth=2,
    save=False,
    filename="residuals.pdf",
    close_fig=True,
):
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()

        ax.set_title(title, fontsize=fontsize)
        ax.set_ylabel(r"Nonlinear Residual $(\%)$", fontsize=fontsize)
        ax.set_xlabel("Iteration", fontsize=fontsize)

        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

        max_len = max(map(len, residuals))

        iter_mult = _calc_iter_mult(max_len)
        fig.gca().xaxis.set_major_locator(MultipleLocator(iter_mult))

        if len(residuals) <= 4:
            for i, residual in enumerate(residuals, 1):
                linestyle, marker, color = _styles[i - 1]
                ax.plot(
                    residual,
                    linestyle=linestyle,
                    marker=marker,
                    color=color,
                    linewidth=linewidth,
                    label=f"Method {i}",
                )
        else:
            for i, residual in enumerate(residuals, 1):
                ax.plot(
                    residual,
                    linewidth=linewidth,
                    label=f"Method {i}",
                )

        ax.legend(fontsize=legend_fontsize, frameon=True, facecolor="white")

    if save:
        _save_fig(fig, filename)

    if close_fig:
        plt.close(fig)

    return


def plot_electrodes_mesh(
    elec_mesh,
    *,
    title="",
    fontsize=14,
    single_figsize=(4.8, 4.8),
    linewidth_mesh=0.5,
    linewidth_elec=3,
    elec_num=True,
    elec_numsize=6,
    with_tank=False,
    with_axis=False,
    save=False,
    filename="electrodes_mesh.pdf",
    close_fig=True,
):
    nr, nc = (1, 2) if with_tank else (1, 1)

    w, h = single_figsize
    figsize = (nc * w, h)

    fig, ax = plt.subplots(nrows=nr, ncols=nc, figsize=figsize)
    fig.tight_layout()

    fig.suptitle(title, fontsize=fontsize)

    axs: tuple[plt.Axes, ...]
    axs = (ax,) if isinstance(ax, plt.Axes) else tuple(ax)

    if with_tank:
        ax0 = axs[0]  ## Tank first

        photo_name = _get_filepath(f"eit_data/target_photos/fantom_1_0.jpg")
        img = mpimg.imread(photo_name)
        ax0.imshow(img)

    ax1 = axs[1] if with_tank else axs[0]
    plt.sca(ax1)  # Set 'ax1' as the active axis

    ## Mesh
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
                fontsize=elec_numsize,
                ha="center",
                va="center",
            )

    ax1.set_aspect("equal")  # Enforce equal aspect ratio
    fxplot(elec_mesh, linewidth=linewidth_mesh)

    if not with_axis:
        _disable_axis(axs)

    if save:
        _save_fig(fig, filename)

    if close_fig:
        plt.close(fig)

    return


def _plot_mesh_handler(func, elec_mesh, mesh_display, linewidth_mesh):
    if mesh_display == "nil":
        p = fxplot(func)
        p.set_rasterized(True)

    if mesh_display == "thin":
        p = fxplot(func)

    if mesh_display == "thick":
        p = fxplot(func)
        fxplot(elec_mesh, linewidth=linewidth_mesh)

    return p


def _create_colorbar(
    p,
    fig,
    ax,
    *,
    orientation="vertical",
    colorbar_fontsize=10,
    fraction=0.046,
    pad=0.04,
    shrink=0.8,
):
    cbar = fig.colorbar(
        p, ax=ax, orientation=orientation, fraction=fraction, pad=pad, shrink=shrink
    )
    cbar.outline.set_visible(True)  # Ensure the outline is visible
    cbar.ax.yaxis.set_ticks_position("right")  # Set tick position
    cbar.ax.yaxis.set_visible(True)  # Ensure axis is visible
    cbar.ax.tick_params(labelsize=colorbar_fontsize)  # Adjust tick label size
    return


def _disable_axis(axs: tuple[plt.Axes, ...]):
    for ax in axs:
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        ax.set_frame_on(False)  # Disable the frame
    return


def _save_fig(fig, filename, *, dpi=300):
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    return


def _get_filepath(path):
    # Get the directory where this function is defined
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to a file relative to this directory
    filepath = os.path.join(current_dir, path)

    return filepath
