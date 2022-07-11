import warnings
from pathlib import Path
from typing import Callable, List, Union, Tuple, Dict

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from intelinair_data.functional import read_imagery
from rasterio import DatasetReader
from rasterio.enums import Resampling
from rasterio.warp import reproject

from yield_forecasting.utils.logger import logger


# TODO: move to intelinair utils
def reproject_raster(input_raster_path: str or Path,
                     reference_raster_path: str or Path,
                     output_raster_path: str or Path,
                     interpolation: int = Resampling.cubic_spline):
    """
    Re-projects the given raster to match the reference raster. Reprojected raster saves in the output raster path.
    Args:
        input_raster_path: the input raster path to be reprojected
        reference_raster_path: the reference raster to be used for re-projection
        output_raster_path: the output raster path to save the reprojected result
        interpolation: the interpolation type to use during re-projection
    """
    with rasterio.open(input_raster_path) as src:
        kwargs = src.meta.copy()
        with rasterio.open(reference_raster_path) as match_ds:
            kwargs.update({
                'crs': match_ds.crs,
                'transform': match_ds.transform,
                'width': match_ds.width,
                'height': match_ds.height})

        with rasterio.open(output_raster_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=match_ds.transform,
                    dst_crs=match_ds.crs,
                    resampling=interpolation)


def get_border_geometry(border_file_path: str, crs: str):
    """Reads the border file, converts its CRS to given CRS and returns the border geometry as a Shapely BaseGeometry
    object - mainly Polygon or MultiPolygon."""
    gdf = gpd.read_file(border_file_path)
    if gdf.shape[0] != 1 or 'geometry' not in gdf.columns:
        raise ValueError(f'Boundary file {border_file_path} is wrong.')
    return gdf.to_crs(crs=crs)['geometry'].iloc[0]


def get_border_mask(border_file_path, raster_dataset) -> np.ndarray:
    """Returns boolean mask, where True means that pixel belongs to the field, aka inside the border geometry."""
    border_geometry = get_border_geometry(border_file_path, raster_dataset.crs)
    mask, _, _ = rasterio.mask.raster_geometry_mask(raster_dataset, [border_geometry], crop=False, invert=True)
    return mask


def change_raster_resolution(input_path: str, output_path: str, resolution: float, dtype: str = None) -> str:
    """
    Converts given raster file to the given raster resolution and saves it in the given output_path.
    Args:
        input_path: the input raster path
        output_path: the output raster path
        resolution: the target resolution in meters
        dtype: final dtype of the raster, if not given will use the same as the input one

    Returns:
        the output path of final raster image
    """
    image, profile = read_imagery(
        input_path, resolution=resolution, bounds=None, return_profile=True, dtype=dtype,
        bounds_crs=None, out_shape=None, resampling=Resampling.cubic,
    )
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(image)

    return output_path


# Adapted from here https://github.com/mapbox/rasterio/blob/master/rasterio/merge.py.
# The adoption forces that all input rasters have the same resolution and shape.
def merge_rasters(datasets: List[Union[DatasetReader, str, Path]],
                  nodata: int or float = None,
                  indexes: int or List[int] = None,
                  method: str or Callable = 'first'):
    """
    Copy valid pixels from input files to an output file.

    All files must have the same number of bands, data type, and
    coordinate reference system, resolution and shape.

    Input files are merged in their listed order using the reverse
    painter's algorithm (default) or another method. If the output file exists,
    its values will be overwritten by input values.

    Geospatial bounds and resolution of a new output file in the
    units of the input file coordinate reference system may be provided
    and are otherwise taken from the first input file.

    Parameters
    ----------
    datasets: list of dataset objects opened in 'r' mode, or paths of datasets
        source datasets to be merged.
    nodata: float, optional
        nodata value to use in output file. If not set, uses the nodata value
        in the first input raster.
    indexes : list of ints or a single int, optional
        bands to read and merge
    method : str or callable
        pre-defined method:
            first: reverse painting
            last: paint valid new on top of existing
            min: pixel-wise min of existing and new
            max: pixel-wise max of existing and new
        or custom callable with signature:

        def function(old_data, new_data, old_nodata, new_nodata):

            Parameters
            ----------
            old_data : array_like
                array to update with new_data
            new_data : array_like
                data to merge
                same shape as old_data
            old_nodata, new_data : array_like
                boolean masks where old/new data is nodata
                same shape as old_data

    Returns
    -------
    tuple

        Two elements:

            dest: numpy ndarray
                Contents of all input rasters in single array

            out_transform: affine.Affine()
                Information for mapping pixel coordinates in `dest` to another
                coordinate system
    """

    def copy_first(merged_data, new_data, merged_mask, new_mask, **kwargs):
        mask = np.empty_like(merged_mask, dtype="bool")
        np.logical_not(new_mask, out=mask)
        np.logical_and(merged_mask, mask, out=mask)
        np.copyto(merged_data, new_data, where=mask, casting="unsafe")

    def copy_last(merged_data, new_data, merged_mask, new_mask, **kwargs):
        mask = np.empty_like(merged_mask, dtype="bool")
        np.logical_not(new_mask, out=mask)
        np.copyto(merged_data, new_data, where=mask, casting="unsafe")

    def copy_min(merged_data, new_data, merged_mask, new_mask, **kwargs):
        mask = np.empty_like(merged_mask, dtype="bool")
        np.logical_or(merged_mask, new_mask, out=mask)
        np.logical_not(mask, out=mask)
        np.minimum(merged_data, new_data, out=merged_data, where=mask)
        np.logical_not(new_mask, out=mask)
        np.logical_and(merged_mask, mask, out=mask)
        np.copyto(merged_data, new_data, where=mask, casting="unsafe")

    def copy_max(merged_data, new_data, merged_mask, new_mask, **kwargs):
        mask = np.empty_like(merged_mask, dtype="bool")
        np.logical_or(merged_mask, new_mask, out=mask)
        np.logical_not(mask, out=mask)
        np.maximum(merged_data, new_data, out=merged_data, where=mask)
        np.logical_not(new_mask, out=mask)
        np.logical_and(merged_mask, mask, out=mask)
        np.copyto(merged_data, new_data, where=mask, casting="unsafe")

    merge_methods = {
        'first': copy_first,
        'last': copy_last,
        'min': copy_min,
        'max': copy_max
    }

    if method in merge_methods:
        copyto = merge_methods[method]
    elif callable(method):
        copyto = method
    else:
        raise ValueError(f'Unknown method {method}, must be one of {list(merge_methods.keys())} or callable')

    first = datasets[0]
    if isinstance(first, (str, Path)):
        first = rasterio.open(first)

    nodataval = first.nodatavals[0]
    dtype = first.dtypes[0]

    # Determine output band count
    if indexes is None:
        output_count = first.count
    elif isinstance(indexes, int):
        output_count = 1
    else:
        output_count = len(indexes)

    # Compute output array shape. We guarantee it will cover the output bounds completely
    output_width, output_height = first.profile['width'], first.profile['height']
    logger.debug(f"Output width: {output_width}, height: {output_height}")

    # create destination array
    dest = np.zeros((output_count, output_height, output_width), dtype=dtype)

    if nodata is not None:
        nodataval = nodata
        logger.debug(f"Set nodataval: {nodataval}")

    if nodataval is not None:
        # Only fill if the nodataval is within dtype's range
        inrange = False
        if np.dtype(dtype).kind in ('i', 'u'):
            info = np.iinfo(dtype)
            inrange = (info.min <= nodataval <= info.max)
        elif np.dtype(dtype).kind == 'f':
            info = np.finfo(dtype)
            if np.isnan(nodataval):
                inrange = True
            else:
                inrange = (info.min <= nodataval <= info.max)
        if inrange:
            dest.fill(nodataval)
        else:
            warnings.warn(
                f"Input file's nodata value, {nodataval}, is beyond the valid "
                f"range of its data type, {dtype}. Consider overriding it "
                "using the --nodata option for better results.")
    else:
        nodataval = 0

    for src in datasets:
        if isinstance(src, (str, Path)):
            src = rasterio.open(src)
        elif not isinstance(src, DatasetReader):
            raise TypeError(f"All datasets must be DatasetReader or path, got: {type(src)}")

        if src.profile['width'] != output_width or src.profile['height'] != output_height:
            raise ValueError('All rasters must have the same shape.')

        temp = src.read(boundless=False, masked=True, indexes=indexes)
        assert temp.shape == dest.shape

        if np.isnan(nodataval):
            dst_nodata = np.isnan(dest)
            temp_nodata = np.isnan(temp)
        else:
            dst_nodata = dest == nodataval
            temp_nodata = temp.mask

        copyto(dest, temp, dst_nodata, temp_nodata)

    return dest, first.transform


def sum_multiple_rasters(raster_files: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Sums a list of raster files that have the same geo data."""
    final_image, no_data_mask, profile = None, None, None
    for raster_path in raster_files:

        with rasterio.open(raster_path) as dataset:
            image = np.squeeze(dataset.read(), axis=0)
            mask = image == dataset.meta['nodata']
            image[mask] = 0

            if profile is None:
                profile = dataset.profile.copy()

        final_image = final_image + image if final_image is not None else image
        no_data_mask = np.logical_and(no_data_mask, mask) if no_data_mask is not None else mask

    return final_image, no_data_mask, profile
