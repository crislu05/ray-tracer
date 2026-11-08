"""Physics module for ray tracing: refraction, reflection, and optical dispersion."""
import numpy as np


def refract(direc, normal, n_1, n_2):
    """Compute the refracted ray direction using Snell's law.

    Args:
        direc (np.ndarray): Incident ray direction.
        normal (np.ndarray): Surface normal at the point of incidence.
        n_1 (float): Refractive index of the incident medium.
        n_2 (float): Refractive index of the transmitted medium.

    Returns:
        np.ndarray or None: Refracted direction as a unit vector, or None if the
        incidence angle exceeds the critical angle (total internal reflection).
    """
    k = direc / np.linalg.norm(direc)
    unit_normal = normal / np.linalg.norm(normal)

    # Ensure the normal is on the same side of the incident ray.
    if np.dot(k, unit_normal) < 0:
        unit_normal = -unit_normal

    cos_theta_1 = np.dot(k, unit_normal)
    sin_theta_1 = np.linalg.norm(np.cross(k, unit_normal))

    if np.isclose(sin_theta_1, 0.0):
        return k

    if sin_theta_1 > n_2 / n_1:
        return None

    sin_theta_2 = (n_1 / n_2) * sin_theta_1
    cos_theta_2 = np.sqrt(1 - sin_theta_2**2)

    # Unit vector perpendicular to the normal in the plane of incidence.
    tangent_vec = (k - cos_theta_1 * unit_normal) / sin_theta_1

    k_refracted = sin_theta_2 * tangent_vec + cos_theta_2 * unit_normal

    return k_refracted / np.linalg.norm(k_refracted)


def reflect(direc, normal):
    """Compute the reflected ray direction using the law of reflection.

    Args:
        direc (np.ndarray): Incident ray direction.
        normal (np.ndarray): Surface normal at the point of incidence.

    Returns:
        np.ndarray: Reflected direction as a unit vector.
    """
    k = direc / np.linalg.norm(direc)
    unit_normal = normal / np.linalg.norm(normal)
    k_reflected = k - 2 * np.dot(k, unit_normal) * unit_normal
    return k_reflected / np.linalg.norm(k_reflected)


class DispersiveMaterial:
    """An optical glass whose refractive index varies with wavelength (dispersion).

    The refractive index is modelled by the Sellmeier equation


    Args:
        b_coeff (np.ndarray): Sellmeier B coefficients (dimensionless).
        c_coeff (np.ndarray): Sellmeier C coefficients (µm²).
    """

    def __init__(self, b_coeff, c_coeff):
        self.__b_coeff = np.array(b_coeff, dtype=float)
        self.__c_coeff = np.array(c_coeff, dtype=float)

    @property
    def b_coeff(self):
        """np.ndarray: Sellmeier B coefficients (dimensionless)."""
        return self.__b_coeff

    @property
    def c_coeff(self):
        """np.ndarray: Sellmeier C coefficients (µm²)."""
        return self.__c_coeff

    def ref_index(self, wavelength):
        """Evaluate the refractive index at a given wavelength.

        Args:
            wavelength (float): Wavelength in µm.

        Returns:
            float: Refractive index for the given wavelength.
        """
        n_square = 1 + np.sum(self.__b_coeff * wavelength**2 / (wavelength**2 - self.__c_coeff))
        return np.sqrt(n_square)
