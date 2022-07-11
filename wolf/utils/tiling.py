"""Special package that provides tiling service."""
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from intelinair_data.functional import read_imagery, create_tiles_bounds, calculate_tiles_transform
from rasterio.crs import CRS
from rasterio.enums import Resampling


class TilingService:
    """Special class for yielding tiles coordinates."""

    def __init__(self,
                 geo_reference: Dict,
                 tile_resolution: float,
                 tile_size: int,
                 stride_size: int,
                 offset: int,
                 ):
        """
        Args:
            geo_reference: the dict with geo-reference information about given imagery
            tile_resolution:  resolution that tile should be read at, in terms of imagery CRS
            tile_size: image size of tile that should be read out
            stride_size: the stride to tile the image in terms of pixels, defaults ot tile_size
            offset: the pixel padding to add the the top and left of the image
        """
        self._geo_reference = geo_reference

        self._resolution = tile_resolution
        self._tile_size = tile_size
        self._stride_size = stride_size
        self._offset = offset

        self._tiles_bounds = self._get_tiles_coordinates()
        self._tiles_profile = self._get_tiles_profile(self._tiles_bounds, self._resolution)

    @property
    def tile_size(self) -> int:
        return self._tile_size

    @property
    def stride_size(self) -> int:
        return self._stride_size

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def tile_resolution(self) -> float:
        return self._resolution

    @property
    def tiles_bounds(self) -> List[Tuple]:
        return self._tiles_bounds

    @property
    def tiles_profile(self) -> Dict:
        return self._tiles_profile

    def _get_tiles_coordinates(self):
        """Creates the tiles coordinates which will be used for tiling the images."""
        return create_tiles_bounds(
            bounds=self._geo_reference['bounds'],
            bounds_crs=self._geo_reference['crs'],
            resolution=self._resolution,
            tile_size=self._tile_size,
            stride=self._stride_size,
            offset=self._offset
        )

    def _get_tiles_profile(self, tiles_bounds: List[Tuple], tiles_resolution: float) -> Dict:
        """Sets tiles geo-reference information that will be used in the feature for combining them."""
        crs = self._geo_reference['crs']
        tiles_transform, tiles_width, tiles_height = calculate_tiles_transform(tiles_bounds, crs, tiles_resolution)
        tiles_profile = deepcopy(self._geo_reference['profile'])
        tiles_profile.update({'transform': tiles_transform, 'width': tiles_width, 'height': tiles_height})
        return tiles_profile

    @staticmethod
    def read_tile(band_path: str or Path, tile_bound: Tuple, tile_crs: CRS, tile_size: int,
                  return_profile: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Reads a tile from given band and returns its array."""
        return read_imagery(
            str(band_path), resolution=None, bounds=tile_bound, return_profile=return_profile, dtype=None,
            bounds_crs=tile_crs, out_shape=(tile_size, tile_size), resampling=Resampling.cubic,
        )

    def __len__(self) -> int:
        return len(self._tiles_bounds)
