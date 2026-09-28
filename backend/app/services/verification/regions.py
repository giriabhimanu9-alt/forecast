"""Coarse regional blocks over India for the first regional blend.

These are simple lat/lon rectangles assigned in priority order. They are NOT
IMD's meteorological subdivisions; they exist to test whether regionally
varying weights help before moving to grid-cell weights.

    Northeast : lon >= 89,  lat >= 21.5     (NE states, Sikkim, north Bengal)
    North     : lat >= 28                   (J&K, HP, Punjab, Haryana, UK, north UP)
    South     : lat < 18                    (peninsula)
    West      : lon < 77                    (Gujarat, Rajasthan, west MP/MH)
    East      : lon >= 83                   (Odisha, Jharkhand, Bihar, WB)
    Central   : everything else             (MP, Chhattisgarh, Vidarbha, south UP)
"""

from __future__ import annotations

import numpy as np
import xarray as xr

REGIONS = ("North", "West", "Central", "East", "South", "Northeast")


def region_map(latitudes: np.ndarray, longitudes: np.ndarray) -> xr.DataArray:
    lat2d, lon2d = np.meshgrid(latitudes, longitudes, indexing="ij")
    labels = np.full(lat2d.shape, "Central", dtype=object)
    assigned = np.zeros(lat2d.shape, dtype=bool)
    for name, mask in (
        ("Northeast", (lon2d >= 89) & (lat2d >= 21.5)),
        ("North", lat2d >= 28),
        ("South", lat2d < 18),
        ("West", lon2d < 77),
        ("East", lon2d >= 83),
    ):
        take = mask & ~assigned
        labels[take] = name
        assigned |= take
    return xr.DataArray(
        labels.astype(str), dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": longitudes}, name="region",
    )
