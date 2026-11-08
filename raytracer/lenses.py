"""Lens module for the ray tracer.

Defines the class hierarchy:
    Lens        - Thick lens composed of two ``SphericalRefraction`` surfaces.
    BiConvex    - Both surfaces are convex.
    PlanoConvex - The front surface is flat, the rear surface is curved.
    ConvexPlano - The front surface is curved, the rear surface is flat.
"""
import numpy as np
from raytracer.elements import OpticalElement, SphericalRefraction
from raytracer.rays import Ray


class Lens(OpticalElement):
    """A thick lens composed of two coaxial refracting surfaces.

    The lens is modelled as two ``SphericalRefraction`` surfaces separated by
    a physical thickness along the z-axis.  

    Subclasses (``BiConvex``, ``PlanoConvex``, ``ConvexPlano``) fix one or
    both curvatures to specialise the geometry without repeating constructor
    logic.

    Args:
        z_0 (float): z-position of the front surface vertex (mm).
        curvature1 (float): Curvature of the front surface (mm⁻¹).
        curvature2 (float): Curvature of the rear surface (mm⁻¹).
        n_inside (float or DispersiveMaterial): Refractive index of the lens medium.
        n_outside (float or DispersiveMaterial): Refractive index of the surrounding medium.
        thickness (float): Axial separation between the two surface vertices (mm).
        aperture (float): Radial aperture of both surfaces (mm).
    """

    def __init__(
            self, z_0, curvature1, curvature2, n_inside, n_outside, thickness, aperture):
        self.__surface1 = SphericalRefraction(
            z_0=z_0,
            aperture=aperture,
            curvature=curvature1,
            n_1=n_outside,
            n_2=n_inside,
        )

        self.__surface2 = SphericalRefraction(
            z_0=z_0 + thickness,
            aperture=aperture,
            curvature=curvature2,
            n_1=n_inside,
            n_2=n_outside,
        )

    @property
    def z_0(self):
        """float: z-position of the front surface vertex (mm)."""
        return self.__surface1.z_0

    @property
    def curvature1(self):
        """float: Curvature of the front surface (mm⁻¹)."""
        return self.__surface1.curvature

    @property
    def curvature2(self):
        """float: Curvature of the rear surface (mm⁻¹)."""
        return self.__surface2.curvature

    @property
    def aperture(self):
        """float: Radial aperture shared by both surfaces (mm)."""
        return self.__surface1.aperture

    @property
    def thickness(self):
        """float: Axial separation between the two surface vertices (mm)."""
        return self.__surface2.z_0 - self.__surface1.z_0

    @property
    def surface1(self):
        """SphericalRefraction: The front refracting surface."""
        return self.__surface1

    @property
    def surface2(self):
        """SphericalRefraction: The rear refracting surface."""
        return self.__surface2

    def n_inside(self, wavelength=0.588):
        """Return the refractive index of the lens medium.

        Args:
            wavelength (float): Wavelength in µm. Used only for dispersive materials.

        Returns:
            float: Refractive index of the lens medium.
        """
        return self.__surface1.n_2(wavelength)

    def n_outside(self, wavelength=0.588):
        """Return the refractive index of the surrounding medium.

        Args:
            wavelength (float): Wavelength in µm. Used only for dispersive materials.

        Returns:
            float: Refractive index of the surrounding medium.
        """
        return self.__surface1.n_1(wavelength)

    def intercept(self, ray):
        """Find the intersection of a ray with the front surface of the lens.

        Args:
            ray (Ray): The incoming ray.

        Returns:
            np.ndarray or None: 3D intersection point, or None if no valid intercept.
        """
        return self.__surface1.intercept(ray)

    def propagate_ray(self, ray):
        """Propagate a ray through both surfaces of the lens in sequence.

        Args:
            ray (Ray): The incoming ray. Modified in place.
        """
        self.__surface1.propagate_ray(ray)
        self.__surface2.propagate_ray(ray)

    def focal_point(self, wavelength=0.588):
        """Find the paraxial focal point by numerical ray tracing.

        Traces a ray at a small transverse offset (0.001 mm) through both
        surfaces and extrapolates its exit path to the optical axis.

        Args:
            wavelength (float): Wavelength in µm for dispersive materials.

        Returns:
            float: z-coordinate of the paraxial focal point (mm).
        """
        ray = Ray(pos=[0.0, 0.001, 0.0], direc=[0.0, 0.0, 1.0], wavelength=wavelength)
        self.propagate_ray(ray)
        return ray.z_int

    def plot_lens(self, axes):
        """Plot the lens cross-section in the y-z plane.

        Args:
            axes (matplotlib.axes.Axes): Axes to plot on.
        """
        y_coords = np.linspace(-self.aperture, self.aperture, 300)

        if self.curvature1 != 0:
            radius1 = 1.0 / self.curvature1
            origin1_z = self.z_0 + radius1
            arc1 = np.sqrt(np.maximum(radius1**2 - y_coords**2, 0.0)) * np.sign(radius1)
            z_coords1 = origin1_z - arc1
        else:
            z_coords1 = np.full_like(y_coords, self.z_0)

        if self.curvature2 != 0:
            radius2 = 1.0 / self.curvature2
            origin2_z = self.surface2.z_0 + radius2
            arc2 = np.sqrt(np.maximum(radius2**2 - y_coords**2, 0.0)) * np.sign(radius2)
            z_coords2 = origin2_z - arc2
        else:
            z_coords2 = np.full_like(y_coords, self.surface2.z_0)

        axes.plot(z_coords1, y_coords, "b-", linewidth=1.5)
        axes.plot(z_coords2, y_coords, "b-", linewidth=1.5)
        axes.plot([z_coords1[0], z_coords2[0]], [y_coords[0], y_coords[0]], "b-", linewidth=1.5)
        axes.plot([z_coords1[-1], z_coords2[-1]], [y_coords[-1], y_coords[-1]], "b-", linewidth=1.5)


class BiConvex(Lens):
    """A biconvex lens with independent curvatures on both surfaces."""


class PlanoConvex(Lens):
    """A plano-convex lens with a flat front surface and a curved rear surface.

    The front curvature is fixed to zero; only the rear curvature is required.

    Args:
        z_0 (float): z-position of the flat front surface vertex (mm).
        curvature (float): Curvature of the rear surface (mm⁻¹).
        n_inside (float or DispersiveMaterial): Refractive index of the lens medium.
        n_outside (float or DispersiveMaterial): Refractive index of the surrounding medium.
        thickness (float): Axial separation between the two surface vertices (mm).
        aperture (float): Radial aperture (mm).
    """

    def __init__(
            self, z_0, curvature, n_inside, n_outside, thickness, aperture):
        super().__init__(z_0, 0, curvature, n_inside, n_outside, thickness, aperture)


class ConvexPlano(Lens):
    """A convex-plano lens with a curved front surface and a flat rear surface.

    The rear curvature is fixed to zero; only the front curvature is required.

    Args:
        z_0 (float): z-position of the curved front surface vertex (mm).
        curvature (float): Curvature of the front surface (mm⁻¹).
        n_inside (float or DispersiveMaterial): Refractive index of the lens medium.
        n_outside (float or DispersiveMaterial): Refractive index of the surrounding medium.
        thickness (float): Axial separation between the two surface vertices (mm).
        aperture (float): Radial aperture (mm).
    """

    def __init__(
            self, z_0, curvature, n_inside, n_outside, thickness, aperture):
        super().__init__(z_0, curvature, 0, n_inside, n_outside, thickness, aperture)
