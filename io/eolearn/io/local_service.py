"""
Module for creating new EOPatches with data obtained from local files
"""

import logging
import datetime as dt
from pathlib import Path

import numpy as np
from sentinelhub import MimeType
from rasterio.warp import calculate_default_transform
from rasterio.transform import Affine
import rasterio
from rasterio import crs

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

    def __init__(self, layer, datatype=0, feature=None, valid_data_mask_feature='IS_DATA', 
                image_format=MimeType.TIFF_d32f):
        # pylint: disable=too-many-arguments
        self.layer = layer
        # self.datafolder = datafolder
        # self.hsplit = hsplit
        # self.vsplit = vsplit
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

        valid_mask = data[..., -1]
        mask_feature_type, mask_feature_name = next(self.valid_data_mask_feature())

        max_value = self.image_format.get_expected_max_value()
        # valid_data = (valid_mask == max_value).astype(np.bool).reshape(valid_mask.shape)
        valid_data = (valid_mask == max_value).astype(np.bool).reshape(valid_mask.shape + (1,))

        if mask_feature_name not in eopatch[mask_feature_type]:
            eopatch[mask_feature_type][mask_feature_name] = valid_data

        # if self.datatype is 3:
        #     valid_mask = data[..., -1]
        #     mask_feature_type, mask_feature_name = next(self.valid_data_mask_feature())

        #     max_value = self.image_format.get_expected_max_value()
        #     # valid_data = (valid_mask == max_value).astype(np.bool).reshape(valid_mask.shape)
        #     valid_data = (valid_mask == max_value).astype(np.bool).reshape(valid_mask.shape + (1,))

        #     if mask_feature_name not in eopatch[mask_feature_type]:
        #         eopatch[mask_feature_type][mask_feature_name] = valid_data


    def _add_meta_info(self, eopatch, bbox):
        """ Adds any missing metadata info to EOPatch """
        
        # if 'maxcc' not in eopatch.meta_info:
        #     eopatch.meta_info['maxcc'] = maxcc

        # if 'time_interval' not in eopatch.meta_info:
        #     eopatch.meta_info['time_interval'] = [dt.datetime(2017, 1, 1, 0, 0), dt.datetime(2017, 12, 31, 0, 0)]
        
        # if 'time_difference' not in eopatch.meta_info:
        #     eopatch.meta_info['time_difference'] = dt.timedelta(-1, 86399)
        
        # if 'service_type' not in eopatch.meta_info:
        #     eopatch.meta_info['service_type'] = service_type.value # ServiceType.IMAGE

        # if 'size_x' not in eopatch.meta_info:
        #     eopatch.meta_info['size_x'] = size_x
        
        # if 'size_y' not in eopatch.meta_info:
        #     eopatch.meta_info['size_y'] = size_y

        if eopatch.bbox is None:
            # bbox: BBox(((510157.61722214246, 5122327.229129893), (513489.214628833, 5125693.036780571)), crs=EPSG:32633)
            # print(bbox)
            # dst_crs = crs.CRS.from_epsg('3857')
            # trans_bbox = calculate_default_transform(bbox[0], dst_crs, bbox[1], bbox[2], *bbox[3])
            # bbox = BBox(((trans_bbox[0][2], trans_bbox[0][5] + trans_bbox[2] * trans_bbox[0][4]), \
            #             (trans_bbox[0][2] + trans_bbox[1] * trans_bbox[0][0] ,trans_bbox[0][5])), \
            #             crs=CRS.POP_WEB)
            eopatch.bbox = bbox

    
    def _split_data(self, eopatch, datas, bbox, i, j, xdis=1098, ydis=1098):
        # 加载数据
        attr_data = np.asarray([datas[k][j][i] for k in range(len(datas))])
        del datas
        # print(attr_data.shape)

        if self.datatype is 0:
            images = ((np.transpose(np.asarray(attr_data), (1, 2, 0)))[np.newaxis, :]).astype(np.float32)
        else:
            images = ((np.transpose(np.asarray(attr_data), (1, 2, 0)))[np.newaxis, :])
        self._add_data(eopatch, images)

        # 设置数据box
        dst_crs = crs.CRS.from_epsg('3857')
        trans_bbox = calculate_default_transform(bbox[0], dst_crs, bbox[1], bbox[2], *bbox[3])[0]
        # print(trans_bbox)

        x = trans_bbox[2] + ((i * xdis)  * trans_bbox[0])
        y = trans_bbox[5] + ((j * ydis)  * trans_bbox[4])
        new_trans = Affine(trans_bbox[0], trans_bbox[1], x, \
                        trans_bbox[3], trans_bbox[4], y)
        # print(new_trans)
        # bbox = BBox(((new_trans[2], new_trans[5] + new_trans[2] * new_trans[4]), \
        #             (new_trans[2] + new_trans[1] * new_trans[0], new_trans[5])), \
                    # crs=CRS.POP_WEB)
        bbox = BBox(((new_trans[2], new_trans[5] + (xdis * new_trans[4])), \
                    (new_trans[2] + (ydis * new_trans[0]), new_trans[5])), \
                    crs=CRS.POP_WEB)

        # print(bbox)
        self._add_meta_info(eopatch, bbox)
        del attr_data

        # for j in range(len(datas.keys())):
        #     x = trans_bbox[2] + ((j * 1098)  * trans_bbox[0])
        #     for i in range(len(datas[j][0])):
        #         attr_data = np.asarray([datas[j][0][i], datas[j][0][i], datas[j][0][i]])
        #         y = trans_bbox[5] + ((i * 1098)  * trans_bbox[4])
        #         new_trans = Affine(trans_bbox[0], trans_bbox[1], x, \
        #                         trans_bbox[3], trans_bbox[4], y)
        #         # 加载数据
        #         if self.datatype is 0:
        #             images = ((np.transpose(np.asarray(attr_data), (1, 2, 0)))[np.newaxis, :]).astype(np.float32)
        #         else:
        #             images = ((np.transpose(np.asarray(attr_data), (1, 2, 0)))[np.newaxis, :])
        #         self._add_data(eopatch, images)

        #         # 设置数据box
        #         bbox = BBox(((new_trans[0][2], new_trans[0][5] + new_trans[2] * new_trans[0][4]), \
        #                     (new_trans[0][2] + new_trans[1] * new_trans[0][0] ,new_trans[0][5])), \
        #                     crs=CRS.POP_WEB)
        #         self._add_meta_info(eopatch, bbox)
        #         del attr_data



    def execute(self, counti, countj, eopatch=None, datafolder=None, hsplit=10, vsplit=10):
        """
        
        """
        if datafolder is None:
            return

        if eopatch is None:
            eopatch = EOPatch()

        # filename = '/Users/xujavy/Documents/Work/data/jupyter_data/sentinel/yunnan/S2B_MSIL1C_20180606T033629_N0206_R061_T48RUQ_20180606T085923.zip'
        filenames = list(Path(datafolder).resolve().glob('*.zip'))
        # images = None
        res = dict()
        for filename in filenames:
            with rasterio.open(filename.as_posix()) as ds:
                subdatasets = ds.subdatasets
                with rasterio.open(subdatasets[self.datatype]) as subds:
                    if self.datatype is 0:
                        for i in range(subds.count):
                            tmpdata = subds.read(i + 1)
                            minval = tmpdata.min()
                            maxval = tmpdata.max()
                            tmpdata = (tmpdata - [minval]) / (maxval - minval)

                            # 进行数据拆分
                            hsplit_data = np.hsplit(tmpdata, hsplit)
                            vsplit_datas = []
                            for j in range(len(hsplit_data)):
                                vsplit_data = np.vsplit(hsplit_data[j], vsplit)
                                vsplit_datas.append(vsplit_data)
                            res[i] = vsplit_datas
                            del tmpdata
                        # images = ((np.transpose(np.asarray(datas), (1, 2, 0)))[np.newaxis, :]).astype(np.float32)
                        # del datas
                    else:
                        for i in range(subds.count):
                            tmpdata = subds.read(i + 1)
                            hsplit_data = np.hsplit(tmpdata, hsplit)
                            vsplit_datas = []
                            for j in range(len(hsplit_data)):
                                vsplit_data = np.vsplit(hsplit_data[j], vsplit)
                                vsplit_datas.append(vsplit_data)
                            res[i] = vsplit_data
                            del tmpdata
                        # images = np.transpose(subds.read(), (1, 2, 0))
                        # images = images[np.newaxis, :]

                    # 获取影像数据的Bounds和参考系
                    srcbound = subds.crs, subds.width, subds.height, subds.bounds
                    # print(srcbound)
                    break
        
        hdis = 10980 / hsplit
        vdis = 10980 / vsplit
        self._split_data(eopatch, res, srcbound, counti, countj, hdis, vdis)
        del res
        # self._add_data(eopatch, np.asarray(images))
        # self._add_meta_info(eopatch, srcbound)
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


class LocalTiffFilesInput(EOTask):
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

    def __init__(self, bands, feature=None, valid_data_mask_feature='IS_DATA', 
                image_format=MimeType.TIFF_d32f):
        # pylint: disable=too-many-arguments
        self.bands = bands
        self.feature_type, self.feature_name = next(self._parse_features(layer if feature is None else feature,
                                                                         default_feature_type=FeatureType.DATA)())

        self.valid_data_mask_feature = self._parse_features(valid_data_mask_feature,
                                                            default_feature_type=FeatureType.MASK)
        self.image_format = image_format

    def _add_data(self, eopatch, data):
        """ Adds data to EOPatch """
        eopatch[self.feature_type][self.feature_name] = data

        valid_mask = data[..., -1]
        mask_feature_type, mask_feature_name = next(self.valid_data_mask_feature())

        max_value = self.image_format.get_expected_max_value()
        # valid_data = (valid_mask == max_value).astype(np.bool).reshape(valid_mask.shape)
        valid_data = (valid_mask == max_value).astype(np.bool).reshape(valid_mask.shape + (1,))

        if mask_feature_name not in eopatch[mask_feature_type]:
            eopatch[mask_feature_type][mask_feature_name] = valid_data


    def _add_meta_info(self, eopatch, bbox):
        """ Adds any missing metadata info to EOPatch """
        
        # if 'maxcc' not in eopatch.meta_info:
        #     eopatch.meta_info['maxcc'] = maxcc

        # if 'time_interval' not in eopatch.meta_info:
        #     eopatch.meta_info['time_interval'] = [dt.datetime(2017, 1, 1, 0, 0), dt.datetime(2017, 12, 31, 0, 0)]
        
        # if 'time_difference' not in eopatch.meta_info:
        #     eopatch.meta_info['time_difference'] = dt.timedelta(-1, 86399)
        
        # if 'service_type' not in eopatch.meta_info:
        #     eopatch.meta_info['service_type'] = service_type.value # ServiceType.IMAGE

        # if 'size_x' not in eopatch.meta_info:
        #     eopatch.meta_info['size_x'] = size_x
        
        # if 'size_y' not in eopatch.meta_info:
        #     eopatch.meta_info['size_y'] = size_y

        if eopatch.bbox is None:
            # bbox: BBox(((510157.61722214246, 5122327.229129893), (513489.214628833, 5125693.036780571)), crs=EPSG:32633)
            # print(bbox)
            # dst_crs = crs.CRS.from_epsg('3857')
            # trans_bbox = calculate_default_transform(bbox[0], dst_crs, bbox[1], bbox[2], *bbox[3])
            # bbox = BBox(((trans_bbox[0][2], trans_bbox[0][5] + trans_bbox[2] * trans_bbox[0][4]), \
            #             (trans_bbox[0][2] + trans_bbox[1] * trans_bbox[0][0] ,trans_bbox[0][5])), \
            #             crs=CRS.POP_WEB)
            eopatch.bbox = bbox

    
    def _split_data(self, eopatch, datas, bbox, i, j, xdis=1098, ydis=1098):
        # 加载数据
        attr_data = np.asarray([datas[k][j][i] for k in range(len(datas))])
        del datas
        # print(attr_data.shape)

        # if self.datatype is 0:
        images = ((np.transpose(np.asarray(attr_data), (1, 2, 0)))[np.newaxis, :]).astype(np.float32)
        # else:
        #     images = ((np.transpose(np.asarray(attr_data), (1, 2, 0)))[np.newaxis, :])
        self._add_data(eopatch, images)

        # 设置数据box
        # dst_crs = crs.CRS.from_epsg('3857')
        # trans_bbox = calculate_default_transform(bbox[0], dst_crs, bbox[1], bbox[2], *bbox[3])[0]
        # # print(trans_bbox)

        trans_bbox = bbox[3]
        x = trans_bbox[2] + ((j * xdis)  * trans_bbox[0])
        y = trans_bbox[5] + ((i * ydis)  * trans_bbox[4])
        new_trans = Affine(trans_bbox[0], trans_bbox[1], x, \
                        trans_bbox[3], trans_bbox[4], y)
        # print(new_trans)
        # new_trans = bbox[3]
        # print(new_trans)
        bbox = BBox(((new_trans[2], new_trans[5] + (xdis * new_trans[4])), \
                    (new_trans[2] + (ydis * new_trans[0]), new_trans[5])), \
                    crs=CRS.POP_WEB)

        # print(bbox)
        self._add_meta_info(eopatch, bbox)
        del attr_data


    def execute(self, counti, countj, eopatch=None, datafolder=None, hsplit=10, vsplit=10):
        """
        
        """
        if datafolder is None:
            return

        if eopatch is None:
            eopatch = EOPatch()

        
        banddict = {'B01':1, 'B02':2, 'B03':3, 'B04':4, 'B05':5, 'B06':6, \
                    'B07':7, 'B08':8, 'B09':9, 'B10':10, 'B11':11, 'B12':12}

        # filename = '/Users/xujavy/Documents/Work/data/jupyter_data/sentinel/yunnan/S2B_MSIL1C_20180606T033629_N0206_R061_T48RUQ_20180606T085923.tif'
        filenames = list(Path(datafolder).resolve().glob('*.tif'))
        res = dict()
        for filename in filenames:
            with rasterio.open(filename.as_posix()) as ds:
                i = 0
                for band in self.bands:
                    tmpdata = ds.read(banddict[band])
                    minval = tmpdata.min()
                    maxval = tmpdata.max()
                    tmpdata = (tmpdata - [minval]) / (maxval - minval)

                    # 进行数据拆分
                    hsplit_data = np.hsplit(tmpdata, hsplit)
                    vsplit_datas = []
                    for j in range(len(hsplit_data)):
                        vsplit_data = np.vsplit(hsplit_data[j], vsplit)
                        vsplit_datas.append(vsplit_data)
                    res[i] = vsplit_datas
                    del tmpdata
                    i = i + 1

                # 获取影像数据的Bounds和参考系
                srcbound = ds.crs, ds.width, ds.height, ds.transform
                # print(srcbound)
        
        hdis = srcbound[2] / hsplit
        vdis = srcbound[1] / vsplit
        # print(hdis, vdis)
        # hdis = ds.profile['width']
        # vdis = ds.profile['height']
        self._split_data(eopatch, res, srcbound, counti, countj, hdis, vdis)
        del res
        return eopatch

class LocalTiffDataInput(LocalTiffFilesInput):
    """
    Adds Sentinel Data to DATA_TIMELESS EOPatch feature.
    """
    def __init__(self, bands, feature=None, **kwargs):
        # if feature is None:
        #     feature = (FeatureType.DATA_TIMELESS, layer)
        # elif isinstance(feature, str):
        #     feature = (FeatureType.DATA_TIMELESS, feature)
        super().__init__(bands=bands, feature=feature, **kwargs)