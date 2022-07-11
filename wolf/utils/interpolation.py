from typing import List, Tuple

import cv2
import numpy as np
import rasterio
from scipy.interpolate import griddata

IMAGE_PADDING_SIZE = 20


class Interpolation:

    @staticmethod
    def read_image(image_path: str):
        """

        Args:
            image_path: path to image for reading

        Returns:

        """
        with rasterio.open(image_path) as src_dataset:
            img = np.squeeze(src_dataset.read())
            profile = src_dataset.profile
        return img, profile

    @staticmethod
    def make_interpolated_image(image: np.ndarray, x_index_input: np.ndarray, y_index_input: np.ndarray,
                                x_index_predict: np.ndarray, y_index_predict: np.ndarray):
        """Make an interpolated image from a valid selection of pixels with scipy.interpolate.griddata.
            Args:
            img: image for interpolation
            x_index_input: x index coordinates for valid pixels in image
            y_index_input: y index coordinates for valid pixels in image
            x_index_predict: x index coordinates for pixels to predict
            y_index_predict: y index coordinates for pixels to predict

        Returns:
            predicted values after interpolation
        """
        samples = image[x_index_input, y_index_input]
        predicted_values = griddata((x_index_input, y_index_input), samples, (x_index_predict, y_index_predict),
                                    method='nearest')
        return predicted_values

    @staticmethod
    def get_padded_coordinates(bbox: List, image_shape: Tuple):
        """
            This function expands the window  by IMAGE_PADDING_SIZE pixels
        Args:
            bbox: bounding box of tiles with nan values
            image_shape: image shape for expansion

        Returns:
                expanded coordinates of image
        """
        start_index_x = (bbox[1] - IMAGE_PADDING_SIZE) if (bbox[1] - IMAGE_PADDING_SIZE) >= 0 else 0
        start_index_y = (bbox[0] - IMAGE_PADDING_SIZE) if (bbox[0] - IMAGE_PADDING_SIZE) >= 0 else 0
        if (bbox[1] + bbox[3] + IMAGE_PADDING_SIZE) < image_shape[0]:
            end_index_x = (bbox[1] + bbox[3] + IMAGE_PADDING_SIZE)
        else:
            end_index_x = image_shape[0]

        if (bbox[0] + bbox[2] + IMAGE_PADDING_SIZE) < image_shape[1]:
            end_index_y = (bbox[0] + bbox[2] + IMAGE_PADDING_SIZE)
        else:
            end_index_y = image_shape[1]
        index_coordinates = np.s_[start_index_x:end_index_x, start_index_y:end_index_y]
        return index_coordinates

    @classmethod
    def interpolate(cls, image: np.ndarray, mask: np.ndarray, missing_data_value: float) -> np.ndarray:
        """
        The algorithm sets Nan values to those parts in the image where the pixel value is equal to nodata and is in
        image boundary. Because  interpolating whole image is slow, it calls connected Components from opencv to process
        them separately. Griddata algorithm can't fill some pixel values which have no valid neighbours, so after
        interpolation the remaining NaN values are filled by the mean of the interpolated pixels.

        Args:
            image: the image array to interpolate
            mask: the mask to exclude values
            missing_data_value: the no data value in the original image to be interpolated

        Returns:
                interpolated_image
        """
        assert mask.dtype.name == 'bool'
        image_shape = image.shape
        interpolated_img = image.copy()
        image[mask & (image == missing_data_value)] = np.nan
        num_labels, labels_im = cv2.connectedComponents(np.isnan(image).astype(np.uint8))
        later_processing_indexes = []
        for index in range(1, num_labels):
            bbox = cv2.boundingRect((labels_im == index).astype(np.uint8))
            strided_bbox = cls.get_padded_coordinates(bbox, image_shape)
            cropped_img, cropped_mask = image[strided_bbox], mask[strided_bbox]
            x_new_predicted, y_new_predicted = np.where(np.isnan(cropped_img) > 0)
            x_input, y_input = np.where((cropped_mask > 0) & (~np.isnan(cropped_img)))
            if not len(x_input):
                later_processing_indexes.append(index)
                continue
            predicted_values = cls.make_interpolated_image(
                cropped_img, x_input, y_input, x_new_predicted, y_new_predicted
            )
            cropped_img[x_new_predicted, y_new_predicted] = predicted_values
            cropped_img[np.isnan(cropped_img)] = predicted_values[~np.isnan(predicted_values)].mean()
            interpolated_img[strided_bbox] = cropped_img

        mean_value_field = interpolated_img[mask & (~np.isnan(interpolated_img))].mean()
        for index in later_processing_indexes:
            interpolated_img[labels_im == index] = mean_value_field
        return interpolated_img

    @classmethod
    def run_interpolation(cls, image_path: str, boundary_path: str, save_path: str = None) -> None:
        """
            interpolates image
        """

        image, image_profile = cls.read_image(image_path)
        b_mask, _ = cls.read_image(boundary_path)
        b_mask = b_mask.astype(bool)
        interpolated_img = cls.interpolate(image, b_mask, image_profile['nodata'])
        if save_path:
            with rasterio.open(save_path, 'w', **image_profile) as dst:
                dst.write(interpolated_img, 1)
