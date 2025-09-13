from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.datasets import fetch_california_housing


@dataclass(frozen=True)
class Data:
    """
    Simple container for regression datasets.

    Attributes
    ----------
    n_samples : int
        Number of samples in this split.
    random_state : int
        Seed used to generate/split this dataset.
    X : NDArray[np.float_]
        Feature matrix, shape (n_samples, n_features).
    y : NDArray[np.float_]
        Target vector, shape (n_samples,).
    """

    n_samples: int
    random_state: int
    X: NDArray[np.float32]
    y: NDArray[np.float32]


@dataclass(frozen=True)
class HousingGeoData:
    """
    Geographic view of the California housing dataset.

    Attributes
    ----------
    longitude : NDArray[np.float_]
        Longitudes for each block group.
    latitude : NDArray[np.float_]
        Latitudes for each block group.
    median_value_100k : NDArray[np.float_]
        Median house value in units of $100,000 (as provided by sklearn).
    """

    longitude: NDArray[np.float32]
    latitude: NDArray[np.float32]
    median_value_100k: NDArray[np.float32]


def load_california_housing_geo() -> HousingGeoData:
    """
    Load the California housing dataset and extract longitude, latitude, and target.

    Returns
    -------
    HousingGeoData
        Dataclass with geo-coordinates and median house values.
    """
    ds = fetch_california_housing()
    X: NDArray[np.float32] = ds.data # pyright: ignore 
    y: NDArray[np.float32] = ds.target # pyright: ignore 

    # Find feature indices robustly
    feature_names = list(ds.feature_names) # pyright: ignore 
    try:
        lon_idx = feature_names.index("Longitude")
        lat_idx = feature_names.index("Latitude")
    except ValueError as exc:
        raise RuntimeError(
            "Expected 'Longitude' and 'Latitude' in feature names."
        ) from exc

    longitude: NDArray[np.float32] = X[:, lon_idx]
    latitude: NDArray[np.float32] = X[:, lat_idx]

    return HousingGeoData(
        longitude=longitude,
        latitude=latitude,
        median_value_100k=y,
    )


def plot_california_housing_geo(
    data: HousingGeoData,
    *,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "viridis",
    alpha: float = 0.6,
    point_size: int = 20,
    grid_alpha: float = 0.3,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Create a scatter plot of California housing locations colored by median value.

    Parameters
    ----------
    data : HousingGeoData
        Geographic housing data (lon/lat + target).
    figsize : Tuple[int, int], optional
        Figure size in inches, by default (12, 8).
    cmap : str, optional
        Matplotlib colormap name, by default "viridis".
    alpha : float, optional
        Point transparency in [0, 1], by default 0.6.
    point_size : int, optional
        Marker size, by default 20.
    grid_alpha : float, optional
        Grid line transparency, by default 0.3.
    save_path : Optional[str], optional
        If provided, save the figure to this path.
    show : bool, optional
        If True, display the plot window (useful in notebooks/scripts).

    Returns
    -------
    (Figure, Axes)
        The created matplotlib Figure and Axes.
    """
    fig: Figure = plt.figure(figsize=figsize)
    ax: Axes = fig.add_subplot(111)

    sc = ax.scatter(
        data.longitude,
        data.latitude,
        c=data.median_value_100k,
        cmap=cmap,
        alpha=alpha,
        s=point_size,
        edgecolors="none",
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Median House Value ($100,000s)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "California Housing Market – Geographic Distribution of Median House Values"
    )
    ax.grid(True, alpha=grid_alpha)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def main() -> None:
    """Entry point for running this module as a script."""
    data = load_california_housing_geo()
    plot_california_housing_geo(data)


if __name__ == "__main__":
    main()
