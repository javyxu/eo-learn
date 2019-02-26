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
from sentinelhub import BBox, CRS, ServiceType

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

    def __init__(self, layer, datafolder, bbox, datatype=0, feature=None, valid_data_mask_feature='IS_DATA', 
                image_format=MimeType.TIFF_d32f):
        # pylint: disable=too-many-arguments
        self.layer = layer
        self.datafolder = datafolder
        self.bbox = bbox
        self.datatype = datatype
        self.feature_type, self.feature_name = next(self._parse_features(layer if feature is None else feature,
                                                                         default_feature_type=FeatureType.DATA)())

        self.valid_data_mask_feature = self._parse_features(valid_data_mask_feature,
                                                            default_feature_type=FeatureType.MASK)
        self.image_format = image_format

    def _add_data(self, eopatch, data):
        """ Adds downloaded data to EOPatch """
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

        if self.datatype is 3:
            valid_mask = data[..., -1]
            mask_feature_type, mask_feature_name = next(self.valid_data_mask_feature())

            max_value = self.image_format.get_expected_max_value()
            # valid_data = (valid_mask == max_value).astype(np.bool).reshape(valid_mask.shape)
            valid_data = (valid_mask == max_value).astype(np.bool).reshape(valid_mask.shape + (1,))

            if mask_feature_name not in eopatch[mask_feature_type]:
                eopatch[mask_feature_type][mask_feature_name] = valid_data


    def _add_meta_info(self, eopatch, maxcc, service_type, size_x, size_y, bbox):
        """ Adds any missing metadata info to EOPatch """
        
        if 'maxcc' not in eopatch.meta_info:
            eopatch.meta_info['maxcc'] = maxcc

        if 'time_interval' not in eopatch.meta_info:
            eopatch.meta_info['time_interval'] = [dt.datetime(2017, 1, 1, 0, 0), dt.datetime(2017, 12, 31, 0, 0)]
        
        if 'time_difference' not in eopatch.meta_info:
            eopatch.meta_info['time_difference'] = dt.timedelta(-1, 86399)
        
        if 'service_type' not in eopatch.meta_info:
            eopatch.meta_info['service_type'] = service_type.value # ServiceType.IMAGE

        if 'size_x' not in eopatch.meta_info:
            eopatch.meta_info['size_x'] = size_x
        
        if 'size_y' not in eopatch.meta_info:
            eopatch.meta_info['size_y'] = size_y

        if eopatch.bbox is None:
            # bbox: BBox(((510157.61722214246, 5122327.229129893), (513489.214628833, 5125693.036780571)), crs=EPSG:32633)
            eopatch.bbox = bbox

    def execute(self, eopatch=None):
        """
        
        """
        if eopatch is None:
            eopatch = EOPatch()

        # filename = '/Users/xujavy/Documents/Work/data/jupyter_data/sentinel/yunnan/S2B_MSIL1C_20180606T033629_N0206_R061_T48RUQ_20180606T085923.zip'
        filenames = list(Path(self.datafolder).resolve().glob('*.zip'))
        images = None
        for filename in filenames:
            with rasterio.open(filename.as_posix()) as ds:
                subdatasets = ds.subdatasets
                with rasterio.open(subdatasets[self.datatype]) as subds:
                    if self.datatype is 0:
                        datas = []
                        for i in range(subds.count):
                            tmpdata = subds.read(i + 1)
                            minval = tmpdata.min()
                            maxval = tmpdata.max()
                            tmpdata = (tmpdata - [minval]) / (maxval - minval)
                            datas.append(tmpdata)
                            del tmpdata
                        images = ((np.transpose(np.asarray(datas), (1, 2, 0)))[np.newaxis, :]).astype(np.float32)
                        del datas
                    else:
                        images = np.transpose(subds.read(), (1, 2, 0))
                        images = images[np.newaxis, :]

                    # TODO: 计算与Shape文件一样的box，和参考系
                    # BBox(bbox=(510157.61722214246, 5122327.229129893, 513489.214628833, 5125693.036780571), crs=CRS.WGS84)
                    # bbox = BBox(bbox=(subds.bounds[0], subds.bounds[1], subds.bounds[2], subds.bounds[3]), crs=CRS.POP_WEB)
                    # bbox =  BBox(((11375052.80128679, 2787505.5475038067), (11489096.847488275, 2924480.702190843)), crs=CRS.POP_WEB)
                    tar_bbox = BBox(((self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3])), crs=CRS.POP_WEB)
                    break
            

        self._add_data(eopatch, np.asarray(images))
        self._add_meta_info(eopatch, 0.8, ServiceType.WCS, '10m', '10m', tar_bbox)
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