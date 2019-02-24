"""
Module for creating new EOPatches with data obtained from local files
"""

import logging
import datetime as dt
from pathlib import Path

import numpy as np
from sentinelhub import MimeType
import rasterio

from eolearn.core import EOPatch, EOTask, FeatureType, get_common_timestamps

LOGGER = logging.getLogger(__name__)


class LocalFilesInput(EOTask):
    """
    Task for creating EOPatch and filling it with data using Local Files.

    :param layer: the preconfigured layer to be added to EOPatch's DATA feature.
    :type layer: str
    :param feature: Feature to which the data will be added. By default the name will be the same as the name of the
        layer
    :type feature: str or (FeatureType, str) or None
    :param valid_data_mask_feature: A feature to which valid data mask will be stored. Default is `'IS_DATA'`.
    :type valid_data_mask_feature: str or (FeatureType, str)
    :param image_format: format of the returned image by the Sentinel Hub's WMS getMap service. Default is 32-bit TIFF.
    :type image_format: constants.MimeType
    """

    def __init__(self, layer, datafolder, datatype=0, feature=None, valid_data_mask_feature='IS_DATA', 
                image_format=MimeType.TIFF_d32f):
        # pylint: disable=too-many-arguments
        self.layer = layer
        self.datafolder = datafolder
        self.datatype = datatype
        self.feature_type, self.feature_name = next(self._parse_features(layer if feature is None else feature,
                                                                         default_feature_type=FeatureType.DATA)())

        # self.valid_data_mask_feature = self._parse_features(valid_data_mask_feature,
        #                                                     default_feature_type=FeatureType.MASK)
        self.image_format = image_format

    def _add_data(self, eopatch, data):
        """ Adds downloaded data to EOPatch """
        # valid_mask = data[..., -1]
        # data = data[..., :-1]

        # if data.ndim == 3:
        #     data = data.reshape(data.shape + (1,))
        # if not self.feature_type.is_time_dependent():
        #     if data.shape[0] > 1:
        #         raise ValueError('Cannot save time dependent data to time independent feature')
        #     data = data.squeeze(axis=0)
        # if self.feature_type.is_discrete():
        #     data = data.astype(np.int32)

        eopatch[self.feature_type][self.feature_name] = data

        # mask_feature_type, mask_feature_name = next(self.valid_data_mask_feature())

        # max_value = self.image_format.get_expected_max_value()
        # valid_data = (valid_mask == max_value).astype(np.bool).reshape(valid_mask.shape + (1,))

        # if mask_feature_name not in eopatch[mask_feature_type]:
        #     eopatch[mask_feature_type][mask_feature_name] = valid_data

    def _add_meta_info(self, eopatch, bbox):
        """ Adds any missing metadata info to EOPatch """

        if eopatch.bbox is None:
            eopatch.bbox = bbox

    def execute(self, eopatch=None):
        """
        
        """
        if eopatch is None:
            eopatch = EOPatch()

        # filename = '/Users/xujavy/Documents/Work/data/jupyter_data/sentinel/yunnan/S2B_MSIL1C_20180606T033629_N0206_R061_T48RUQ_20180606T085923.zip'
        filenames = list(Path(self.datafolder).resolve().glob('*.zip'))
        for filename in filenames:
            with rasterio.open(filename.as_posix()) as ds:
                subdatasets = ds.subdatasets
                with rasterio.open(subdatasets[self.datatype]) as subds:
                    tmp_bands = np.transpose(subds.read(), (1, 2, 0))
                    images = tmp_bands[np.newaxis, :]
                    del tmp_bands
                    break
            

        self._add_data(eopatch, np.asarray(images))
        # self._add_meta_info(eopatch, '')
        return eopatch


class LocalSenDataInput(LocalFilesInput):
    """
    Adds Sentinel Data to DATA_TIMELESS EOPatch feature.
    """
    def __init__(self, layer, feature=None, **kwargs):
        if feature is None:
            feature = (FeatureType.DATA_TIMELESS, layer)
        elif isinstance(feature, str):
            feature = (FeatureType.DATA_TIMELESS, feature)
        super().__init__(layer=layer, feature=feature, **kwargs)