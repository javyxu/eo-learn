"""
Module for cloud masking
"""

import logging
from pathlib import Path

import numpy as np
import scipy.ndimage
import rasterio
from sentinelhub import DataSource, MimeType, ServiceType
from s2cloudless import S2PixelCloudDetector, MODEL_EVALSCRIPT

from eolearn.core import EOTask, get_common_timestamps


INTERP_METHODS = ['nearest', 'linear']

LOGGER = logging.getLogger(__name__)


class AddLocalCloudMaskTask(EOTask):
    """ Task to add a cloud mask and cloud probability map to an EOPatch

    This task computes a cloud probability map and corresponding cloud binary mask for the input EOPatch. The classifier
    to be used to compute such maps must be provided at declaration. The `data_feature` to be used as input to the
    classifier is also a mandatory argument. If `data_feature` exists already, downscaling to the given (lower) cloud
    mask resolution is performed, the classifier is run, and upsampling returns the cloud maps to the original
    resolution.
    Otherwise, if `data_feature` does not exist, a new OGC request at the given cloud mask resolution is made, the
    classifier is run, and upsampling returns the cloud masks to original resolution. This design should allow faster
    execution of the classifier, and reduce the number of requests. `linear` interpolation is used for resampling of
    the `data_feature` and cloud probability map, while `nearest` interpolation is used to upsample the binary cloud
    mask.

    This implementation should allow usage with any cloud detector implemented for different data sources (S2, L8, ..).
    """
    def __init__(self, classifier, data_feature, cm_size_x=None, cm_size_y=None, cmask_feature='CLM',
                 cprobs_feature=None, data_source=DataSource.SENTINEL2_L1C,
                 image_format=MimeType.TIFF_d32f, model_evalscript=MODEL_EVALSCRIPT):
        """ Constructor

        If both `cm_size_x` and `cm_size_y` are `None` and `data_feature` exists, cloud detection is computed at same
        resolution of `data_feature`.

        :param classifier: Cloud detector classifier. This object implements a `get_cloud_probability_map` and
                            `get_cloud_masks` functions to generate probability maps and binary masks
        :param data_feature: Name of key in eopatch.data dictionary to be used as input to the classifier. If the
                           `data_feature` does not exist, a new OGC request at the given cloud mask resolution is made
                           with layer name set to `data_feature` parameter.
        :param cm_size_x: Resolution to be used for computation of cloud mask. Allowed values are number of column
                            pixels (WMS-request) or spatial resolution (WCS-request, e.g. '10m'). Default is `None`
        :param cm_size_y: Resolution to be used for computation of cloud mask. Allowed values are number of row
                            pixels (WMS-request) or spatial resolution (WCS-request, e.g. '10m'). Default is `None`
        :param cmask_feature: Name of key to be used for the cloud mask to add. The cloud binary mask is added to the
                            `eopatch.mask` attribute dictionary. Default is `'clm'`.
        :param cprobs_feature: Name of key to be used for the cloud probability map to add. The cloud probability map is
                            added to the `eopatch.data` attribute dictionary. Default is `None`, so no cloud
                            probability map will be computed.
        :param data_source: Data source to be requested by OGC service request. Default is `DataSource.SENTINEL2_L1C`
        :param image_format: Image format to be requested by OGC service request. Default is `MimeType.TIFF_d32f`
        :param model_evalscript: CustomUrlParam defining the EVALSCRIPT to be used by OGC request. Should reflect the
                            request necessary for the correct functioning of the classifier. For instance, for the
                            `S2PixelCloudDetector` classifier, `MODEL_EVALSCRIPT` is used as it requests the required 10
                            bands. Default is `MODEL_EVALSCRIPT`
        """
        self.classifier = classifier
        # self.datafolder = datafolder
        self.data_feature = data_feature
        self.cm_feature = cmask_feature
        self.cm_size_x = cm_size_x
        self.cm_size_y = cm_size_y
        self.cprobs_feature = cprobs_feature
        self.data_source = data_source
        self.image_format = image_format
        self.model_evalscript = model_evalscript

    def _get_rescale_factors(self, reference_shape, meta_info):
        """ Compute the resampling factor for height and width of the input array

        :param reference_shape: Tuple specifying height and width in pixels of high-resolution array
        :type reference_shape: tuple of ints
        :param meta_info: Meta-info dictionary of input eopatch. Defines OGC request and parameters used to create the
                            eopatch
        :return: Rescale factor for rows and columns
        :rtype: tuple of floats
        """
        # Figure out resampling size
        height, width = reference_shape

        service_type = ServiceType(meta_info['service_type'])
        rescale = None
        if service_type == ServiceType.WMS:

            if (self.cm_size_x is None) and (self.cm_size_y is not None):
                rescale = (self.cm_size_y / height, self.cm_size_y / height)
            elif (self.cm_size_x is not None) and (self.cm_size_y is None):
                rescale = (self.cm_size_x / width, self.cm_size_x / width)
            else:
                rescale = (self.cm_size_y / height, self.cm_size_x / width)

        elif service_type == ServiceType.WCS:
            # Case where only one resolution for cloud masks is specified in WCS
            if self.cm_size_y is None:
                self.cm_size_y = self.cm_size_x
            elif self.cm_size_x is None:
                self.cm_size_x = self.cm_size_y

            hr_res_x, hr_res_y = int(meta_info['size_x'].strip('m')), int(meta_info['size_y'].strip('m'))
            lr_res_x, lr_res_y = int(self.cm_size_x.strip('m')), int(self.cm_size_y.strip('m'))
            rescale = (hr_res_y / lr_res_y, hr_res_x / lr_res_x)

        return rescale

    def _downscaling(self, hr_array, meta_info, interp='linear', smooth=True):
        """ Downscale existing array to resolution requested by cloud detector

        :param hr_array: High-resolution data array to be downscaled
        :param meta_info: Meta-info of eopatch
        :param interp: Interpolation method to be used in downscaling. Default is `'linear'`
        :param smooth: Apply Gaussian smoothing in spatial directions before downscaling. Sigma of kernel is estimated
                        by rescaling factor. Default is `True`
        :return: Down-scaled array
        """
        # Run cloud mask on full resolution
        if (self.cm_size_y is None) and (self.cm_size_x is None):
            return hr_array, None

        # Rescaling factor in spatial (height, width) dimensions
        rescale = self._get_rescale_factors(hr_array.shape[1:3], meta_info)

        if smooth:
            sigma = (0,) + tuple(int(1/x) for x in rescale) + (0,)
            hr_array = scipy.ndimage.gaussian_filter(hr_array, sigma)

        lr_array = scipy.ndimage.interpolation.zoom(hr_array, (1.0,) + rescale + (1.0,),
                                                    order=INTERP_METHODS.index(interp), mode='nearest')

        return lr_array, rescale

    @staticmethod
    def _upsampling(lr_array, rescale, reference_shape, interp='linear'):
        """ Upsample the low-resolution array to the original high-resolution grid

        :param lr_array: Low-resolution array to be upsampled
        :param rescale: Rescale factor for rows/columns
        :param reference_shape: Original size of high-resolution eopatch. Tuple with dimension for time, height and
                                width
        :param interp: Interpolation method ot be used in upsampling. Default is `'linear'`
        :return: Upsampled array. The array has 4 dimensions, the last one being of size 1
        """
        hr_shape = reference_shape + (1,)
        lr_shape = lr_array.shape + (1,)

        if rescale is None:
            return lr_array.reshape(lr_shape)

        out_array = scipy.ndimage.interpolation.zoom(lr_array.reshape(lr_shape),
                                                     (1.0,) + tuple(1 / x for x in rescale) + (1.0,),
                                                     output=lr_array.dtype, order=INTERP_METHODS.index(interp),
                                                     mode='nearest')

        # Padding and cropping might be needed to get to the reference shape
        out_shape = out_array.shape
        padding = tuple((0, np.max((h-o, 0))) for h, o in zip(hr_shape, out_shape))
        hr_array = np.pad(out_array, padding, 'edge')
        hr_array = hr_array[:, :hr_shape[1], :hr_shape[2], :]

        return hr_array

    def execute(self, eopatch, counti, countj, datafolder=None, hsplit=10, vsplit=10):
        """ Add cloud binary mask and (optionally) cloud probability map to input eopatch

        :param eopatch: Input `EOPatch` instance
        :return: `EOPatch` with additional cloud maps
        """
        # Downsample or make request
        if not eopatch.data:
            raise ValueError('EOPatch must contain some data feature')
        
        bands = {}
        bands[0] = (2, 0)
        bands[1] = (0, 0)
        bands[2] = (0, 2)
        bands[3] = (1, 0)
        bands[4] = (0, 3)
        bands[5] = (1, 3)
        bands[6] = (2, 1)
        bands[7] = (2, 2)
        bands[8] = (1, 4)
        bands[9] = (1, 5)

        filenames = list(Path(datafolder).resolve().glob('*.zip'))
        res = dict()
        for filename in filenames:
            with rasterio.open(filename.as_posix()) as ds:
                subdatasets = ds.subdatasets
                i = 0
                for v in bands.values():
                    with rasterio.open(subdatasets[v[0]]) as subds:
                        tmpdata = subds.read(v[1] + 1)
                        minval = tmpdata.min()
                        maxval = tmpdata.max()
                        tmpdata = (tmpdata - [minval]) / (maxval - minval)
                        if v[0] is 1:
                            tmpdata = scipy.ndimage.interpolation.zoom(tmpdata, 2, order=3, mode='nearest')
                        elif v[0] is 2:
                            tmpdata = scipy.ndimage.interpolation.zoom(tmpdata, 6, order=3, mode='nearest')

                        # 进行数据拆分
                        hsplit_data = np.hsplit(tmpdata, hsplit)
                        vsplit_datas = []
                        for j in range(len(hsplit_data)):
                            vsplit_data = np.vsplit(hsplit_data[j], vsplit)
                            vsplit_datas.append(vsplit_data)
                        res[i] = vsplit_datas
                        del tmpdata
                    i = i + 1

        attr_data = np.asarray([res[k][countj][counti] for k in range(len(res))])
        new_data = np.transpose(attr_data, (1, 2, 0))[np.newaxis, :]
        del attr_data

        # clf_probs_lr = self.classifier.get_cloud_probability_maps(new_data)
        # clf_mask_lr = self.classifier.get_mask_from_prob(clf_probs_lr)

        # Add cloud mask as a feature to EOPatch
        # reference_shape = next(iter(eopatch.data.values())).shape[:3]
        # rescale = self._get_rescale_factors(reference_shape[1:3], eopatch.meta_info)
        # clf_mask_hr = self._upsampling(clf_mask_lr, rescale, reference_shape, interp='nearest')
        clf_probs = self.classifier.get_cloud_probability_maps(new_data)
        clf_mask = self.classifier.get_mask_from_prob(clf_probs)
        eopatch.mask[self.cm_feature] = np.transpose(clf_mask, (1, 2, 0))[np.newaxis, :]

        # If the feature name for cloud probability maps is specified, add as feature
        if self.cprobs_feature is not None:
            # clf_probs_hr = self._upsampling(clf_probs_lr, rescale, reference_shape, interp='linear')
            eopatch.data[self.cprobs_feature] = (np.transpose(clf_probs, (1, 2, 0))[np.newaxis, :]).astype(np.float32)
        
        del new_data
        del res
        return eopatch


    # def execute(self, eopatch):
    #     """ Add cloud binary mask and (optionally) cloud probability map to input eopatch

    #     :param eopatch: Input `EOPatch` instance
    #     :return: `EOPatch` with additional cloud maps
    #     """
    #     # Downsample or make request
    #     if not eopatch.data:
    #         raise ValueError('EOPatch must contain some data feature')
        
    #     bands = {}
    #     bands[0] = (2, 0)
    #     bands[1] = (0, 0)
    #     bands[2] = (0, 2)
    #     bands[3] = (1, 0)
    #     bands[4] = (0, 3)
    #     bands[5] = (1, 3)
    #     bands[6] = (2, 1)
    #     bands[7] = (2, 2)
    #     bands[8] = (1, 4)
    #     bands[9] = (1, 5)

    #     filenames = list(Path(self.datafolder).resolve().glob('*.zip'))
    #     for filename in filenames:
    #         with rasterio.open(filename.as_posix()) as ds:
    #             subdatasets = ds.subdatasets
    #             new_data = None
    #             for v in bands.values():
    #                 with rasterio.open(subdatasets[v[0]]) as subds:
    #                     tmpdata = subds.read(v[1] + 1)
    #                     minval = tmpdata.min()
    #                     maxval = tmpdata.max()
    #                     tmpdata = (tmpdata - [minval]) / (maxval - minval)
    #                     if v[0] is 1:
    #                         tmpdata = scipy.ndimage.interpolation.zoom(tmpdata, 2, order=3, mode='nearest')
    #                     elif v[0] is 2:
    #                         tmpdata = scipy.ndimage.interpolation.zoom(tmpdata, 6, order=3, mode='nearest')
    #                     if new_data is None:
    #                         new_data = tmpdata[np.newaxis, :]
    #                     else:
    #                         new_data = np.vstack((new_data, tmpdata[np.newaxis, :]))
    #                     del tmpdata
        
    #     new_data = np.transpose(new_data, (1, 2, 0))[np.newaxis, :]

    #     # clf_probs_lr = self.classifier.get_cloud_probability_maps(new_data)
    #     # clf_mask_lr = self.classifier.get_mask_from_prob(clf_probs_lr)

    #     # Add cloud mask as a feature to EOPatch
    #     # reference_shape = next(iter(eopatch.data.values())).shape[:3]
    #     # rescale = self._get_rescale_factors(reference_shape[1:3], eopatch.meta_info)
    #     # clf_mask_hr = self._upsampling(clf_mask_lr, rescale, reference_shape, interp='nearest')
    #     clf_probs = self.classifier.get_cloud_probability_maps(new_data)
    #     clf_mask = self.classifier.get_mask_from_prob(clf_probs)
    #     eopatch.mask[self.cm_feature] = np.transpose(clf_mask, (1, 2, 0))[np.newaxis, :]

    #     # If the feature name for cloud probability maps is specified, add as feature
    #     if self.cprobs_feature is not None:
    #         # clf_probs_hr = self._upsampling(clf_probs_lr, rescale, reference_shape, interp='linear')
    #         eopatch.data[self.cprobs_feature] = (np.transpose(clf_probs, (1, 2, 0))[np.newaxis, :]).astype(np.float32)

    #     return eopatch


def get_s2_pixel_cloud_detector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True):
    """ Wrapper function for pixel-based S2 cloud detector `S2PixelCloudDetector`
    """
    return S2PixelCloudDetector(threshold=threshold,
                                average_over=average_over,
                                dilation_size=dilation_size,
                                all_bands=all_bands)
