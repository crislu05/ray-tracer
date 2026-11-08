"""Concentric polar point generator for ray bundles."""

import numpy as np


def rtrings(rmax=5.0, nrings=5, multi=6):
    """Generate (x, y) positions arranged on concentric polar rings.

    Rays are distributed evenly around each ring.  The number of points on
    ring k is ``multi * k`` to keep a uniform angular distribution of ray bundles across radii.

    Args:
        rmax (float): Radius of the outermost ring (mm). Defaults to ``5.0``.
        nrings (int): Number of concentric rings. Defaults to ``5``.
        multi (int): Multiplier controlling points per ring. Ring k has
            ``multi * k`` equally spaced points. Defaults to ``6``.

    Returns:
        positions (list of tuples): Ordered list of ``(x, y)`` positions (mm).
    """
    positions = []

    for ring in range(1, int(nrings) + 1):
        ring_radius = (rmax / nrings) * ring
        n_rays = multi * ring

        for i in range(n_rays):
            angle = (2 * np.pi * i) / n_rays  # Equally spaced angles around the ring.
            x_pos = ring_radius * np.cos(angle)
            y_pos = ring_radius * np.sin(angle)
            positions.append((x_pos, y_pos))

    return positions
