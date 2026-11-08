"""Optical elements module for the ray tracer.

Defines the class hierarchy for optical surfaces and elements:
    OpticalElement  - abstract base class
    SphericalSurface - spherical surface geometry
    SphericalRefraction - refracting surface (Snell's law)
    SphericalReflection - reflecting surface (law of reflection)
    PlanarSurface - flat surface (zero curvature)
    OutputPlane - detector plane
"""

from abc import ABC, abstractmethod
import numpy as np
import raytracer.rays
from raytracer.physics import reflect, refract, DispersiveMaterial


class OpticalElement(ABC):
    """Abstract base class for all optical elements.
    """

    @abstractmethod
    def intercept(self, _ray):
        """Find the intersection point of a ray with this element.

        Args:
            _ray (Ray): The incoming ray.

        Returns:
            np.ndarray or None: 3D intersection point, or None if no valid intercept.
        """
        raise NotImplementedError('intercept() needs to be implemented in derived classes')

    @abstractmethod
    def propagate_ray(self, _ray):
        """Propagate a ray through this element, updating its position and direction.

        Args:
            _ray (Ray): The incoming ray. Modified in place.
        """
        raise NotImplementedError('propagate_ray() needs to be implemented in derived classes')


class SphericalSurface(OpticalElement, ABC):
    """A spherical optical surface centred on the z-axis.

    Handles both curved (curvature != 0) and flat (curvature == 0) surfaces.
    Subclasses supply the optical physics through ``propagate_ray``.

    Args:
        z_0 (float): z-position of the surface vertex (mm).
        aperture (float): Maximum radial extent of the surface (mm).
        curvature (float): Reciprocal of the radius of curvature (mm⁻¹).
            Positive curves toward +z; zero gives a flat surface.
    """

    def __init__(self, z_0, aperture, curvature):
        self.__z_0 = z_0
        self.__aperture = aperture
        self.__curvature = curvature

    @property
    def z_0(self):
        """float: z-coordinate of the surface vertex (mm)."""
        return self.__z_0

    @property
    def aperture(self):
        """float: Radial aperture of the surface (mm)."""
        return self.__aperture

    @property
    def curvature(self):
        """float: Curvature of the surface (mm⁻¹); zero for a flat surface."""
        return self.__curvature

    def intercept(self, ray):
        """Find the first valid intersection of a ray with the surface.

        Dispatches to ``_intercept_flat`` or ``_intercept_curved`` depending
        on whether the surface curvature is zero.

        Args:
            ray (Ray): The incoming ray.

        Returns:
            np.ndarray or None: 3D intersection point, or None if no valid intercept.
        """
        if self.curvature == 0:
            return self._intercept_flat(ray)
        return self._intercept_curved(ray)

    def _intercept_flat(self, ray):
        """Intersect the ray with a flat surface.

        Args:
            ray (Ray): The incoming ray.

        Returns:
            np.ndarray or None: 3D intersection point, or None if no valid intercept.
        """
        pos = ray.pos
        direc = ray.direc
        k = direc / np.linalg.norm(direc)

        if np.isclose(k[2], 0.0):
            return None

        ray_length = (self.z_0 - pos[2]) / k[2]
        if ray_length < 0:
            return None

        intercept_pt = pos + ray_length * k

        if np.sqrt(intercept_pt[0]**2 + intercept_pt[1]**2) > self.aperture:
            return None
        return intercept_pt

    def _intercept_curved(self, ray):
        """Intersect the ray with a curved surface.

        Solves the ray-surface quadratic and selects the physically correct root.

        Args:
            ray (Ray): The incoming ray.

        Returns:
            np.ndarray or None: 3D intersection point, or None if no valid intercept.
        """
        pos = ray.pos
        direc = ray.direc
        k = direc / np.linalg.norm(direc)
        radius = 1 / self.curvature
        origin = np.array([0, 0, self.z_0 + radius])
        offset_vec = pos - origin
        offset_dot_k = np.dot(offset_vec, k)
        discriminant = offset_dot_k**2 - (np.dot(offset_vec, offset_vec) - radius**2)

        if discriminant < 0:
            return None

        ray_length_1 = -offset_dot_k + np.sqrt(discriminant)
        ray_length_2 = -offset_dot_k - np.sqrt(discriminant)

        # For a convex surface (radius > 0) take the shorter path; for concave the longer.
        if ray_length_1 > 0 and ray_length_2 > 0:
            if radius > 0:
                ray_length = min(ray_length_1, ray_length_2)
            else:
                ray_length = max(ray_length_1, ray_length_2)
        elif ray_length_2 < 0 < ray_length_1:
            ray_length = ray_length_1
        elif ray_length_1 < 0 < ray_length_2:
            ray_length = ray_length_2
        else:
            return None

        intercept_pt = pos + ray_length * k

        if np.sqrt(intercept_pt[0]**2 + intercept_pt[1]**2) > self.aperture:
            return None
        return intercept_pt

    def surface_normal(self, intercept_pt):
        """Return the outward unit normal to the surface at the intercept point.

        For a flat surface the normal is always [0, 0, 1].  For a curved
        surface it points radially away from the centre of curvature.

        Args:
            intercept_pt (np.ndarray): Intersection point on the surface.

        Returns:
            np.ndarray: Unit normal vector at the intercept point.
        """
        if self.curvature == 0:
            return np.array([0, 0, 1])
        radius = 1 / self.curvature
        origin = np.array([0, 0, self.z_0 + radius])
        return (intercept_pt - origin) / np.linalg.norm(intercept_pt - origin)

    def focal_point(self, wavelength=0.588):
        """Find the paraxial focal point by numerical ray tracing.

        Traces a paraxial ray at a small transverse offset (0.001 mm), propagates
        it through this surface via ``propagate_ray``, and extrapolates its exit
        path to the optical axis.  Works for both refracting and reflecting
        surfaces because ``propagate_ray`` is resolved polymorphically.

        Args:
            wavelength (float): Wavelength in µm. Relevant only for dispersive
                refracting surfaces; ignored by reflecting surfaces.

        Returns:
            float or None: z-coordinate of the paraxial focal point (mm), or
            None if the surface is flat (curvature == 0).
        """
        if self.curvature == 0:
            return None
        ray = raytracer.rays.Ray(
            pos=[0.0, 0.001, 0.0], direc=[0.0, 0.0, 1.0], wavelength=wavelength
        )
        self.propagate_ray(ray)
        return ray.z_int

    def plot_spherical_surface(self, axes):
        """Draw the surface profile in the y-z plane on a matplotlib axes.

        Args: axes: Axes to draw onto.
        """
        y_coords = np.linspace(-self.aperture, self.aperture, 300)
        if self.curvature != 0:
            radius = 1.0 / self.curvature
            origin_z = self.z_0 + radius
            arc = np.sqrt(np.maximum(radius**2 - y_coords**2, 0.0)) * np.sign(radius)
            z_coords = origin_z - arc
        else:
            z_coords = np.full_like(y_coords, self.z_0)
        axes.plot(z_coords, y_coords, "k-", linewidth=1.5)


class PlanarSurface(SphericalSurface):
    """A flat surface perpendicular to the z-axis.

    Implemented as a ``SphericalSurface`` with zero curvature.

    Args:
        z_0 (float): z-position of the plane (mm).
        aperture (float): Radial aperture (mm). Defaults to infinity.
    """

    def __init__(self, z_0, aperture=float("inf")):
        super().__init__(z_0, aperture, curvature=0)


class SphericalRefraction(SphericalSurface):
    """A refracting spherical surface separating two optical media.

    Applies Snell's law at the surface to compute the refracted ray direction.
    Both ``n_1`` and ``n_2`` may be plain floats or ``DispersiveMaterial``
    objects for wavelength-dependent behaviour.

    Args:
        z_0 (float): z-position of the surface vertex (mm).
        curvature (float): Surface curvature (mm⁻¹).
        n_1 (float or DispersiveMaterial): Refractive index of the incident medium.
        n_2 (float or DispersiveMaterial): Refractive index of the transmitted medium.
        aperture (float): Radial aperture (mm).
    """

    def __init__(self, z_0, curvature, n_1, n_2, aperture):
        super().__init__(z_0, aperture, curvature)
        self.__n_1_material = n_1
        self.__n_2_material = n_2

    def n_1(self, wavelength=0.588):
        """Return the refractive index of the incident medium.

        Args:
            wavelength (float): Wavelength in µm. Used only for dispersive materials.

        Returns:
            float: Refractive index of the incident medium.
        """
        if isinstance(self.__n_1_material, DispersiveMaterial):
            return self.__n_1_material.ref_index(wavelength)
        return self.__n_1_material

    def n_2(self, wavelength=0.588):
        """Return the refractive index of the transmitted medium.

        Args:
            wavelength (float): Wavelength in µm. Used only for dispersive materials.

        Returns:
            float: Refractive index of the transmitted medium.
        """
        if isinstance(self.__n_2_material, DispersiveMaterial):
            return self.__n_2_material.ref_index(wavelength)
        return self.__n_2_material

    def propagate_ray(self, ray):
        """Refract a ray through this surface using Snell's law.

        Args:
            ray (Ray): The incoming ray. Modified in place on success.
        """
        intercept_pt = self.intercept(ray)
        if intercept_pt is None:
            return
        normal = self.surface_normal(intercept_pt)
        propagated_direc = refract(
            ray.direc, normal, self.n_1(ray.wavelength), self.n_2(ray.wavelength)
        )
        if propagated_direc is None:
            return
        ray.append(intercept_pt, propagated_direc)


class OutputPlane(PlanarSurface):
    """A detector plane that records the ray position without altering its direction.

    Args:
        z_0 (float): z-position of the output plane (mm).
    """

    def propagate_ray(self, ray):
        """Advance the ray to the output plane, preserving its direction.

        Args:
            ray (Ray): The incoming ray. Modified in place.
        """
        intercept_pt = self.intercept(ray)
        if intercept_pt is None:
            return
        ray.append(intercept_pt, ray.direc)


class SphericalReflection(SphericalSurface):
    """A reflecting spherical mirror surface.

    Applies the law of reflection at the surface normal.

    Args:
        z_0 (float): z-position of the mirror vertex (mm).
        aperture (float): Radial aperture (mm).
        curvature (float): Surface curvature (mm⁻¹).
            Negative for a concave mirror, positive for a convex mirror.
    """

    def propagate_ray(self, ray):
        """Reflect a ray off this mirror surface.

        Args:
            ray (Ray): The incoming ray. Modified in place.
        """
        intercept_pt = self.intercept(ray)
        if intercept_pt is None:
            return
        normal = self.surface_normal(intercept_pt)
        propagated_direc = reflect(ray.direc, normal)
        ray.append(intercept_pt, propagated_direc)
