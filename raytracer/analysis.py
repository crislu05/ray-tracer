"""Analysis module."""
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from raytracer._utils.decorators import SaveOutput
from raytracer.elements import OutputPlane, SphericalReflection, SphericalRefraction
from raytracer.lenses import BiConvex, ConvexPlano, PlanoConvex
from raytracer.physics import DispersiveMaterial
from raytracer.rays import Ray, RayBundle


def task8():
    """
    Task 8.

    In this function you should check your propagate_ray function properly
    finds the correct intercept and correctly refracts a ray. Don't forget
    to check that the correct values are appended to your Ray object.
    """
    surface = SphericalRefraction(z_0=10, curvature=0.02, n_1=1.0, n_2=1.5, aperture=50.0)

    # --- intercept checks (range of starting positions along x) ---
    intercept_cases = [
        (Ray(pos=[0.0, 0.0, 0.0]), [0.0, 0.0, 10.0]),
        (Ray(pos=[1.0, 0.0, 0.0]), [1.0, 0.0, 10.010001]),
        (Ray(pos=[-1.0, 0.0, 0.0]), [-1.0, 0.0, 10.010001]),
        (Ray(pos=[7.0, 0.0, 0.0]), [7.0, 0.0, 10.49242482]),
    ]
    for ray, expected_q in intercept_cases:
        q = surface.intercept(ray)
        assert q is not None, f"No intercept for ray at {ray.pos}"
        assert np.allclose(q, expected_q), (
            f"Intercept mismatch at {ray.pos}: got {q}, expected {expected_q}"
        )

    # Ray outside aperture should not intercept
    blocked = Ray(pos=[10.0, 0.0, 0.0])
    narrow_surface = SphericalRefraction(
        z_0=10, curvature=0.02, n_1=1.0, n_2=1.5, aperture=9.0
    )
    assert narrow_surface.intercept(blocked) is None

    # --- propagate_ray: position, direction, and vertices after refraction ---
    propagate_cases = [
        (
            Ray(pos=[0.0, 0.0, 0.0], direc=[0.0, 0.0, 1.0]),
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 1.0],
        ),
        (
            Ray(pos=[1.0, 2.0, 0.0], direc=[0.0, 0.0, 1.0]),
            [1.0, 2.0, 10.05002503],
            [-0.00667112, -0.01334223, 0.99988873],
        ),
    ]
    for ray, expected_pos, expected_direc in propagate_cases:
        start_pos = ray.pos.copy()
        surface.propagate_ray(ray)

        assert np.allclose(ray.pos, expected_pos), (
            f"Position after propagate from {start_pos}: "
            f"got {ray.pos}, expected {expected_pos}"
        )
        assert np.allclose(ray.direc, expected_direc) or np.allclose(
            ray.direc / np.linalg.norm(ray.direc),
            np.array(expected_direc) / np.linalg.norm(expected_direc),
        ), (
            f"Direction after propagate from {start_pos}: "
            f"got {ray.direc}, expected {expected_direc}"
        )
        verts = ray.vertices
        assert len(verts) == 2, f"Expected 2 vertices, got {len(verts)}"
        assert np.allclose(verts[0], start_pos)
        assert np.allclose(verts[1], expected_pos)

    print("task8: all propagate_ray / intercept checks passed.")


@SaveOutput("task10")
def task10():
    """
    Task 10.

    In this function you should create Ray objects with the given initial positions.
    These rays should be propagated through the surface, up to the output plane.
    You should then plot the tracks of these rays.
    This function should return the matplotlib figure of the ray paths.

    Returns:
        Figure: the ray path plot.
    """
    surface = SphericalRefraction(
        z_0=100, curvature=0.03, n_1=1.0, n_2=1.5, aperture=34.0
    )
    output_plane = OutputPlane(z_0=250.0)

    initial_positions = [
        [0, 4, 0],
        [0, 1, 0],
        [0, 0.2, 0],
        [0, 0, 0],
        [0, -0.2, 0],
        [0, -1, 0],
        [0, -4, 0],
    ]

    rays = [Ray(pos=pos) for pos in initial_positions]

    fig, ax = plt.subplots()
    for ray in rays:
        surface.propagate_ray(ray)
        output_plane.propagate_ray(ray)
        ray.track_plot(axes=ax)

    surface.plot_spherical_surface(axes=ax)
    ax.axvline(output_plane.z_0, color="gray", linestyle="--", linewidth=1, label="Output plane")

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")

    ax.set_ylim(-10, 10)

    ax.set_title("Ray track plot")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    return fig


@SaveOutput("task11", plot_output_indices=itemgetter(0))
def task11():
    """
    Task 11.

    In this function you should propagate the three given paraxial rays through the system
    to the output plane and the tracks of these rays should then be plotted.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for ray paths
    2. the calculated focal point.

    Returns:
        tuple[Figure, float]: the ray path plot and the focal point
    """
    surface = SphericalRefraction(
        z_0=100, curvature=0.03, n_1=1.0, n_2=1.5, aperture=34.0
    )
    focal_point = surface.focal_point()
    output_plane = OutputPlane(z_0 = focal_point)

    initial_positions = [
        [0.1, 0.1, 0],
        [0, 0, 0],
        [-0.1, -0.1, 0],
    ]

    rays = [Ray(pos=pos) for pos in initial_positions]

    fig, ax = plt.subplots()
    for ray in rays:
        surface.propagate_ray(ray)
        output_plane.propagate_ray(ray)
        ray.track_plot(axes=ax)

    surface.plot_spherical_surface(axes=ax)
    ax.axvline(focal_point, color="gray", linestyle="--", linewidth=1, label="Focal plane")

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Paraxial ray track plot")
    ax.set_ylim(-0.5, 0.5)
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    return fig, focal_point


@SaveOutput("task12")
def task12():
    """
    Task 12.

    In this function you should create a RayBunble and propagate it to the output plane
    before plotting the tracks of the rays.
    This function should return the matplotlib figure of the track plot.

    Returns:
        Figure: the track plot.
    """
    surface = SphericalRefraction(
        z_0=100, curvature=0.03, n_1=1.0, n_2=1.5, aperture=34.0
    )
    focal_point = surface.focal_point()
    output_plane = OutputPlane(z_0=focal_point)

    bundle = RayBundle()
    bundle.propagate_bundle([surface, output_plane])
    fig = bundle.track_plot()
    surface.plot_spherical_surface(axes=fig.axes[0])
    fig.axes[0].set_ylim(-10, 10)
    fig.axes[0].axvline(focal_point, color="gray", linestyle="--", linewidth=1, label="Focal plane")
    fig.axes[0].legend(loc="upper right")
    return fig


@SaveOutput("task13", plot_output_indices=itemgetter(0))
def task13():
    """
    Task 13.

    In this function you should again create and propagate a RayBundle to the output plane
    before plotting the spot plot.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the spot plot
    2. the simulation RMS

    Returns:
        tuple[Figure, float]: the spot plot and rms
    """
    surface = SphericalRefraction(
        z_0=100, curvature=0.03, n_1=1.0, n_2=1.5, aperture=34.0
    )
    output_plane = OutputPlane(z_0=surface.focal_point())

    bundle = RayBundle()
    bundle.propagate_bundle([surface, output_plane])
    return bundle.spot_plot(), bundle.rms


@SaveOutput("task14", plot_output_indices=itemgetter(0))
def task14():
    """
    Task 14.

    In this function you will trace a number of RayBundles through the optical system and
    plot the RMS and diffraction scale dependence on input beam radii.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the diffraction scale plot
    2. the simulation RMS for input beam radius 2.5
    3. the diffraction scale for input beam radius 2.5

    Returns:
        tuple[Figure, float, float]: the plot, the simulation RMS value, the diffraction scale.
    """
    lam = 588e-6
    surface = SphericalRefraction(
        z_0=100, curvature=0.03, n_1=1.0, n_2=1.5, aperture=34.0
    )
    focal_point = surface.focal_point()
    focal_length = focal_point - surface.z_0
    output_plane = OutputPlane(z_0=focal_point)
    radii = np.linspace(0.1, 10.0, 100)

    rms_values = []
    diff_values = []

    for rmax in radii:
        bundle = RayBundle(rmax=rmax)
        bundle.propagate_bundle([surface, output_plane])
        rms_values.append(bundle.rms)
        diff_values.append(lam * focal_length / (2.0 * rmax))

    reference_radius = 2.5
    reference_bundle = RayBundle(rmax=reference_radius)
    reference_bundle.propagate_bundle([surface, output_plane])
    rms_at_2p5 = reference_bundle.rms
    diff_at_2p5 = lam * focal_length / (2.0 * reference_radius)

    fig, ax = plt.subplots()
    ax.plot(radii, rms_values, label="Simulation RMS")
    ax.plot(radii, diff_values, label="Diffraction scale")
    ax.set_xlabel("Bundle radius (mm)")
    ax.set_ylabel("Spot size (mm)")
    ax.set_title("RMS and diffraction scale vs bundle radius")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig, rms_at_2p5, diff_at_2p5


@SaveOutput(["task15a", "task15b"], plot_output_indices=itemgetter(0, 2))
def task15():
    """
    Task 15.

    In this function you will create plano-convex lenses in each orientation and propagate a
    RayBundle through each to their respective focal point. You should then plot the spot plot
    for each orientation.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the spot plot for the plano-convex system
    2. the focal point for the plano-convex lens
    3. the matplotlib figure object for the spot plot for the convex-plano system
    4  the focal point for the convex-plano lens

    Returns:
        tuple[Figure, float, Figure, float]: the spot plots and focal points for
        plano-convex and convex-plano.
    """
    pc_lens = PlanoConvex(
        z_0=100,
        curvature=-0.02,
        n_inside=1.5168,
        n_outside=1.0,
        thickness=5.0,
        aperture=50.0,
    )
    cp_lens = ConvexPlano(
        z_0=100,
        curvature=0.02,
        n_inside=1.5168,
        n_outside=1.0,
        thickness=5.0,
        aperture=50.0,
    )

    pc_focal_point = pc_lens.focal_point()
    cp_focal_point = cp_lens.focal_point()

    pc_output = OutputPlane(z_0=pc_focal_point)
    cp_output = OutputPlane(z_0=cp_focal_point)

    pc_bundle = RayBundle(rmax=5.0)
    pc_bundle.propagate_bundle([pc_lens, pc_output])

    cp_bundle = RayBundle(rmax=5.0)
    cp_bundle.propagate_bundle([cp_lens, cp_output])

    pc_fig = pc_bundle.spot_plot()
    pc_fig.axes[0].set_title("Plano-convex (PC) spot plot at focal point")

    cp_fig = cp_bundle.spot_plot()
    cp_fig.axes[0].set_title("Convex-plano (CP) spot plot at focal point")

    return pc_fig, pc_focal_point, cp_fig, cp_focal_point


@SaveOutput("task16", plot_output_indices=itemgetter(0))
def task16():
    """
    Task 16.

    In this function you will be again plotting the radial dependence of the RMS and
    diffraction values for each orientation of your lens.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the diffraction scale plot
    2. the RMS for input beam radius 3.5 for the plano-convex system
    3. the RMS for input beam radius 3.5 for the convex-plano system
    4  the diffraction scale for input beam radius 3.5

    Returns:
        tuple[Figure, float, float, float]: the plot, RMS for plano-convex, RMS for
        convex-plano, diffraction scale.
    """
    lam = 588e-6
    pc_lens = PlanoConvex(
        z_0=100,
        curvature=-0.02,
        n_inside=1.5168,
        n_outside=1.0,
        thickness=5.0,
        aperture=50.0,
    )
    cp_lens = ConvexPlano(
        z_0=100,
        curvature=0.02,
        n_inside=1.5168,
        n_outside=1.0,
        thickness=5.0,
        aperture=50.0,
    )

    pc_focal_point = pc_lens.focal_point()
    cp_focal_point = cp_lens.focal_point()
    pc_output = OutputPlane(z_0=pc_focal_point)
    cp_output = OutputPlane(z_0=cp_focal_point)

    focal_length = 1.0 / ((1.5168 - 1.0) * 0.02)
    radii = np.linspace(0.1, 10.0, 100)

    pc_rms_values = []
    cp_rms_values = []
    diff_values = []

    for rmax in radii:
        pc_bundle = RayBundle(rmax=rmax)
        pc_bundle.propagate_bundle([pc_lens, pc_output])
        pc_rms_values.append(pc_bundle.rms)

        cp_bundle = RayBundle(rmax=rmax)
        cp_bundle.propagate_bundle([cp_lens, cp_output])
        cp_rms_values.append(cp_bundle.rms)

        diff_values.append(lam * focal_length / (2.0 * rmax))

    reference_radius = 3.5
    pc_reference = RayBundle(rmax=reference_radius)
    pc_reference.propagate_bundle([pc_lens, pc_output])
    pc_rms_at_3p5 = pc_reference.rms

    cp_reference = RayBundle(rmax=reference_radius)
    cp_reference.propagate_bundle([cp_lens, cp_output])
    cp_rms_at_3p5 = cp_reference.rms

    diff_at_3p5 = lam * focal_length / (2.0 * reference_radius)

    fig, ax = plt.subplots()
    ax.plot(radii, pc_rms_values, label="Plano-convex RMS")
    ax.plot(radii, cp_rms_values, label="Convex-plano RMS")
    ax.plot(radii, diff_values, label="Diffraction scale")
    ax.set_xlabel("Bundle radius (mm)")
    ax.set_ylabel("Spot size (mm)")
    ax.set_title("RMS and diffraction scale vs radius of bundle")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig, pc_rms_at_3p5, cp_rms_at_3p5, diff_at_3p5


@SaveOutput("task17", plot_output_indices=itemgetter(0))
def task17():
    """
    Task 17.

    In this function you will be first plotting the spot plot for your PlanoConvex lens with
    the curved side first (at the focal point). You will then be optimising the curvatures of
    a BiConvex lens in order to minimise the RMS spot size at the same focal point.
    This function should return the following items as a tuple in the following order:
    1. The comparison spot plot for both PlanoConvex (curved side first) and BiConvex lenses
       at PlanoConvex focal point.
    2. The RMS spot size for the PlanoConvex lens at focal point
    3. the RMS spot size for the BiConvex lens at PlanoConvex focal point

    Returns:
        tuple[Figure, float, float]: The combined spot plot, RMS for the PC lens,
        RMS for the BiConvex lens
    """
    cp_lens = ConvexPlano(
        z_0=100,
        curvature=0.02,
        n_inside=1.5168,
        n_outside=1.0,
        thickness=5.0,
        aperture=50.0,
    )
    cp_focal_point = cp_lens.focal_point()
    cp_output = OutputPlane(z_0=cp_focal_point)
    cp_bundle = RayBundle(rmax=5.0)
    cp_bundle.propagate_bundle([cp_lens, cp_output])
    cp_rms = cp_bundle.rms

    def objective_function(x):
        curvature1, curvature2 = x
        bc_lens = BiConvex(
            z_0=100,
            curvature1=curvature1,
            curvature2=curvature2,
            n_inside=1.5168,
            n_outside=1.0,
            thickness=5.0,
            aperture=50.0,
        )
        bc_bundle = RayBundle(rmax=5.0)
        bc_bundle.propagate_bundle([bc_lens, cp_output])
        return bc_bundle.rms

    x0 = [0.01, -0.01]
    result = minimize(objective_function, x0=x0, method="Nelder-Mead")

    bc_lens = BiConvex(
        z_0=100,
        curvature1=result.x[0],
        curvature2=result.x[1],
        n_inside=1.5168,
        n_outside=1.0,
        thickness=5.0,
        aperture=50.0,
    )
    bc_bundle = RayBundle(rmax=5.0)
    bc_bundle.propagate_bundle([bc_lens, cp_output])
    bc_rms = bc_bundle.rms


    fig, ax = plt.subplots(figsize=(6, 5))
    cp_bundle.spot_plot(axes=ax, color="b")
    bc_bundle.spot_plot(axes=ax, color="r")

    # Phantom entries supply RMS values in the legend without re-plotting data.
    ax.plot([], [], "b.", markersize=6, label=f"Convex-plano (RMS = {cp_rms:.4f} mm)")
    ax.plot([], [], "r.", markersize=6, label=f"BiConvex (RMS = {bc_rms:.4f} mm)")
    ax.legend()
    ax.set_title("Combined spot plot")
    ax.grid(True)

    fig.tight_layout()
    return fig, cp_rms, bc_rms




@SaveOutput("task18", plot_output_indices=itemgetter(0))
def task18():
    """
    Task 18.

    In this function you will be testing your reflection modelling. Create a new
    SphericalReflection surface and trace a RayBundle through it to the OutputPlane.
    This function should return
    the following items as a tuple in the following order:
    1. The track plot showing reflecting ray bundle off SphericalReflection surface.
    2. The focal point of the SphericalReflection surface.

    Returns:
        tuple[Figure, float]: The track plot, the focal point
    """
    reflector = SphericalReflection(z_0=100, aperture=50.0, curvature= -0.02)
    focal_point = reflector.focal_point()
    output_plane = OutputPlane(z_0=50.0)

    bundle = RayBundle(rmax=5.0)
    bundle.propagate_bundle([reflector, output_plane])

    fig = bundle.track_plot()
    reflector.plot_spherical_surface(axes=fig.axes[0])
    fig.axes[0].set_ylim(-10, 10)
    fig.axes[0].axvline(50.0, color="gray", linestyle="--", linewidth=1, label="Focal plane")
    fig.axes[0].legend(loc="upper left")
    return fig, focal_point


@SaveOutput(["task19a", "task19b"])
def task19():
    """
    Task 19.

    In this function you will be quantifing the amount of spherical aberration.
    Create a new ConvexPlano lens and trace a RayBundle through it to the OutputPlane
    located at the focal point.
    This function should return two matplotlib figures as a tuple in the following order:
    1. The plot showing the transverse spherical aberration i.e. rms as a function of beam
       radius at focal point.
    2. The plot showing the longitudinal spherical aberration i.e. z intercept of a ray as
       a function of transverse distance from optical axis.

    Returns:
        tuple[Figure, Figure]: Transverse spherical aberration plot, longitudinal spherical
        aberration plot
    """
    lens = ConvexPlano(
        z_0=100,
        curvature=0.02,
        n_inside=1.5168,
        n_outside=1.0,
        thickness=5.0,
        aperture=50.0,
    )
    focal_point = lens.focal_point()
    output_plane = OutputPlane(z_0=focal_point)

    radii = np.linspace(0.1, 10.0, 100)
    rms_values = []

    for rmax in radii:
        bundle = RayBundle(rmax=rmax)
        bundle.propagate_bundle([lens, output_plane])
        rms_values.append(bundle.rms)

    fig_tvsa, ax_tvsa = plt.subplots()
    ax_tvsa.plot(radii, rms_values)
    ax_tvsa.set_xlabel("Beam radius (mm)")
    ax_tvsa.set_ylabel("RMS spot size (mm)")
    ax_tvsa.set_title("Transverse spherical aberration")
    ax_tvsa.grid(True)
    fig_tvsa.tight_layout()

    z_intercepts = []

    for rmax in radii:
        ray = Ray(pos=[0.0, rmax, 0.0], direc=[0.0, 0.0, 1.0])
        lens.propagate_ray(ray)
        z_intercepts.append(ray.z_int)

    fig_lgsa, ax_lgsa = plt.subplots()
    ax_lgsa.plot(radii, z_intercepts)
    ax_lgsa.axhline(
        focal_point, color="k", linewidth=0.8, linestyle="--", label="Paraxial focal point"
    )
    ax_lgsa.set_xlabel("Beam radius (mm)")
    ax_lgsa.set_ylabel("z intercept (mm)")
    ax_lgsa.set_title("Longitudinal spherical aberration")
    ax_lgsa.legend()
    ax_lgsa.grid(True)
    fig_lgsa.tight_layout()

    return fig_tvsa, fig_lgsa


@SaveOutput("task20")
def task20():
    """
    Task 20.

    In this function you will investigate dispersion. Create several Rays with different
    wavelengths. Plot the paths of these rays through a BiConvex lens made of BK7 glass
    to an OutputPlane at z=200.
    This function should return the track plot of the path of your Rays through the glass lens.

    Returns:
        Figure: The ray track plot.
    """
    z_0 = 100
    curvature1 = 0.02
    curvature2 = -0.02
    thickness = 5
    aperture = 50.0

    bk7 = DispersiveMaterial(
        [1.03961212, 0.231792344, 1.01046945],
        [6.00069867e-3, 2.00179144e-2, 1.03560653e2],
    )
    lens = BiConvex(
        z_0=z_0,
        curvature1=curvature1,
        curvature2=curvature2,
        n_inside=bk7,
        n_outside=1.0,
        thickness=thickness,
        aperture=aperture,
    )
    output_plane = OutputPlane(z_0=200.0)

    wavelengths = [0.400, 0.480, 0.588, 0.656, 0.700]
    colors = plt.colormaps["rainbow"](np.linspace(0, 1, len(wavelengths)))
    start_pos = [0, 2, 0]
    start_direc = [0, 0, 1]

    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5))

    for wl, color in zip(wavelengths, colors):
        ray = Ray(pos=start_pos, direc=start_direc, wavelength=wl)
        lens.propagate_ray(ray)
        output_plane.propagate_ray(ray)
        label = f"{wl * 1e3:.0f} nm"
        ray.track_plot(axes=ax,      color=color, label=label)
        ray.track_plot(axes=ax_zoom, color=color, label=label)

    lens.plot_lens(ax)
    lens.plot_lens(ax_zoom)

    ax.set_xlim(75, 200)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title("Dispersion through BK7 BiConvex lens")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.legend(loc="upper right", fontsize=8)

    ax_zoom.set_xlim(z_0 - 1, z_0 + thickness + 1)
    ax_zoom.set_ylim(1.92, 2.02)
    ax_zoom.set_title("Dispersion within BK7 BiConvex lens (Zoomed in)")
    ax_zoom.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def main():
    """Run all analysis tasks (8–20) and save marking plots where applicable."""
    print("Running task 8...")
    task8()

    print("Running task 10...")
    task10()

    print("Running task 11...")
    _fig11, focal_point = task11()
    print(f"  Focal point: {focal_point} mm")

    print("Running task 12...")
    task12()

    print("Running task 13...")
    _fig13, rms = task13()
    print(f"  RMS at focal point: {rms} mm")

    print("Running task 14...")
    _fig14, rms_2p5, diff_2p5 = task14()
    print(f"  RMS at r=2.5: {rms_2p5} mm, diffraction scale: {diff_2p5} mm")

    print("Running task 15...")
    _fig_pc, focal_pc, _fig_cp, focal_cp = task15()
    print(f"  Plano-convex focal point: {focal_pc} mm")
    print(f"  Convex-plano focal point: {focal_cp} mm")

    print("Running task 16...")
    _fig16, pc_rms, cp_rms, diff = task16()
    print(f"  Plano-convex RMS: {pc_rms} mm, convex-plano RMS: {cp_rms} mm, diffraction: {diff} mm")

    print("Running task 17...")
    _fig17, cp_rms, bc_rms = task17()
    print(f"  Convex-plano RMS: {cp_rms} mm, BiConvex RMS: {bc_rms} mm")


    print("Running task 18...")
    _fig18, refl_focal_point = task18()
    print(f"  Spherical reflection focal point: {refl_focal_point} mm")

    print("Running task 19...")
    task19()

    print("Running task 20...")
    task20()

    print("All tasks complete.")


if __name__ == "__main__":
    main()
