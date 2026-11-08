"""Ray and RayBundle modules for the ray tracer."""

import numpy as np
import matplotlib.pyplot as plt

from raytracer.genpolar import rtrings


def _validate_vec3(value, name):
    """Validate and convert a value to a 3-element float array.

    Args:
        value (list, tuple, or np.ndarray): The value to validate.
        name (str): Parameter name used in error messages.

    Returns:
        np.ndarray: A 1-D float array of shape (3,).

    Raises:
        TypeError: If ``value`` is not a list, tuple, or ndarray.
        ValueError: If ``value`` does not have exactly three elements.
    """
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f"{name} must be a list or array")
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-element vector")
    return arr


class Ray:
    """A single optical ray defined by its position, direction, and wavelength.

    The ray stores every point it passes through as a vertex.

    Args:
        pos (list, tuple, or np.ndarray, optional): Initial 3D position [x, y, z] in mm.
            Defaults to [0, 0, 0].
        direc (list, tuple, or np.ndarray, optional): Initial 3D direction vector [dx, dy, dz].
            Defaults to [0, 0, 1].
        wavelength (float, optional): Wavelength in µm for dispersive media.
            Defaults to 0.588 µm.
    """

    def __init__(self, pos=None, direc=None, wavelength=None):

        if pos is None:
            pos = [0.0, 0.0, 0.0]
        else:
            pos = _validate_vec3(pos, "pos")
        if direc is None:
            direc = [0.0, 0.0, 1.0]
        else:
            direc = _validate_vec3(direc, "direc")
        if wavelength is None:
            wavelength = 0.588

        self.__pos = np.array(pos, dtype=float)
        self.__direc = np.array(direc, dtype=float)
        self.__direc /= np.linalg.norm(self.__direc)
        self.__vertices = [self.__pos.copy()]
        self.__wavelength = wavelength

    @property
    def pos(self):
        """np.ndarray: Current 3D position of the ray (mm)."""
        return self.__pos

    @property
    def direc(self):
        """np.ndarray: Current unit direction vector of the ray."""
        return self.__direc

    @property
    def vertices(self):
        """list[np.ndarray]: Ordered list of all positions visited by the ray (mm)."""
        return self.__vertices

    @property
    def wavelength(self):
        """float: Wavelength of the ray (µm)."""
        return self.__wavelength

    def append(self, pos, direc):
        """Advance the ray to a new position with a new direction.

        Both the position and direction are validated and stored.  The new
        position is also appended to the vertex history.

        Args:
            pos (list, tuple, or numpy array): New 3D position [x, y, z] in mm.
            direc (list, tuple, or numpy array): New 3D direction vector [dx, dy, dz].
        """
        pos = _validate_vec3(pos, "pos")
        direc = _validate_vec3(direc, "direc")
        self.__pos = np.array(pos, dtype=float)
        self.__direc = np.array(direc, dtype=float)
        self.__direc /= np.linalg.norm(self.__direc)
        self.__vertices.append(self.__pos.copy())

    @property
    def z_int(self):
        """float: z-coordinate at which the ray crosses the optical axis (mm).

        The axis intercept is computed by extrapolating the current direction
        until y (or x if the ray lies in the x–z plane) reaches zero. Returns None
        if the ray travels parallel to the optical axis.
        """
        pos = self.__pos
        direc = self.__direc


        if abs(direc[1]) > 1e-12:
            ray_length = -pos[1] / direc[1]
        elif abs(direc[0]) > 1e-12:
            ray_length = -pos[0] / direc[0]
        else:
            return None

        return pos[2] + ray_length * direc[2]

    def track_plot(self, axes=None, color=None, label=None):
        """Plot the ray path in the y–z plane.

        Args:
            axes (matplotlib.axes.Axes, optional): Axes to draw onto. If None,
                a new figure and axes are created.
            color (str or tuple, optional): Matplotlib colour for the line.
                Defaults to None (uses the axes colour cycle).
            label (str, optional): Legend label for this ray's path.
                Defaults to None (no label added).

        Returns:
            matplotlib.figure.Figure: The figure containing the plot.
        """
        vertices = np.array(self.vertices)
        y_coords = vertices[:, 1]
        z_coords = vertices[:, 2]

        if axes is None:
            fig, axes = plt.subplots()
            axes.set_title("Ray track plot")
            axes.axhline(0, color="k", linewidth=0.5, linestyle="--")
        else:
            fig = axes.get_figure()

        axes.set_xlabel("z (mm)")
        axes.set_ylabel("y (mm)")
        axes.plot(z_coords, y_coords, "-o", markersize=3, color=color, label=label)
        return fig


class RayBundle:
    """A collection of rays arranged on concentric polar rings.

    Rays are generated using ``genpolar.rtrings``.

    Args:
        rmax (float): Radius of the outermost ring (mm). Defaults to 5.0.
        nrings (int): Number of concentric rings. Defaults to 5.
        multi (int): Points-per-ring multiplier; ring k has (multi * k)
            equally spaced rays. Defaults to 6.
    """

    def __init__(self, rmax=5., nrings=5., multi=6):
        self.__rays = []

        self.__rays.append(Ray())

        for x_pos, y_pos in rtrings(rmax=rmax, nrings=nrings, multi=multi):
            self.__rays.append(Ray(pos=[x_pos, y_pos, 0], direc=[0, 0, 1]))

    @property
    def rays(self):
        """list of Rays: All rays in the bundle, including the central ray."""
        return self.__rays

    @property
    def rms(self):
        """RMS spot radius of all rays at their current positions (mm).
        """
        positions = np.array([ray.pos for ray in self.__rays])
        rms_spot_radius = np.sqrt(np.mean((positions[:, 0] ** 2) + (positions[:, 1] ** 2)))
        return rms_spot_radius

    def propagate_bundle(self, elements):
        """Propagate the ray bundle through a sequence of optical elements.

        Args:
            elements (iterable of OpticalElement): Ordered sequence of optical elements
            to propagate through.
        """
        for ray in self.__rays:
            for element in elements:
                element.propagate_ray(ray)

    def track_plot(self):
        """Plot the path of the ray bundle in the y–z plane.

        Returns:
            fig (matplotlib.figure.Figure): Figure containing the bundle track plot.
        """
        fig, axes = plt.subplots()

        for ray in self.__rays:
            vertices = np.array(ray.vertices)
            y_coords = vertices[:, 1]
            z_coords = vertices[:, 2]
            axes.plot(z_coords, y_coords, 'b-', linewidth=0.5)
        axes.set_xlabel('z (mm)')
        axes.set_ylabel('y (mm)')
        axes.set_title('Ray bundle track plot')
        axes.axhline(0, color='k', linewidth=0.5, linestyle='--')
        axes.grid(True)

        return fig

    def spot_plot(self, axes=None, color="b"):
        """Plot the transverse positions of all rays at their current location.

        Args:
            axes (matplotlib.axes.Axes, optional): Axes to draw onto. If None,
                a new figure and axes are created.
            color (str, optional): Matplotlib colour for the ray dots.
                Defaults to ``"b"`` (blue).

        Returns:
            matplotlib.figure.Figure: Figure containing the spot diagram.
        """
        if axes is None:
            fig, axes = plt.subplots()
            axes.set_title("Spot plot")
            axes.set_aspect("equal")
            axes.axhline(0, color="k", linewidth=0.5, linestyle="--")
            axes.axvline(0, color="k", linewidth=0.5, linestyle="--")
        else:
            fig = axes.get_figure()

        axes.set_xlabel("x (mm)")
        axes.set_ylabel("y (mm)")
        axes.grid(True)

        for ray in self.__rays:
            pos = ray.pos
            axes.plot(pos[0], pos[1], ".", color=color, markersize=3)

        fig.tight_layout()
        return fig
