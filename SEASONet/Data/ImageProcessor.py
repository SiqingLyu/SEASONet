"""
Date: 2022.9.22
Author: Lv
this class aims to create a class which contains all the operation on Images
Last coded date: 2022.9.23
"""

import numpy as np
from numpy import ndarray
import os
import tiffile as tif
import cv2
import sys
from math import pi
sys.path.append('/home/dell/lsq/SEASONet_231030')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif'
]
city_dict = {
    'Aomen': 1, 'Baoding': 2, 'Beijing': 3, 'Changchun': 4, 'Changsha': 5,
    'Changzhou': 6, 'Chengdu': 7, 'Chongqing': 8, 'Dalian': 9, 'Dongguan': 10,
    'Eerduosi': 11, 'Foshan': 12, 'Fuzhou': 13, 'Guangzhou': 14, 'Guiyang': 15,
    'Haerbin': 16, 'Haikou': 17, 'Hangzhou': 18, 'Hefei': 19, 'Huhehaote': 20,
    'Huizhou': 21, 'Jinan': 22, 'Kunming': 23, 'Lanzhou': 24, 'Lasa': 25,
    'Luoyang': 26, 'Nanchang': 27, 'Nanjing': 28, 'Nanning': 29, 'Ningbo': 30,
    'Quanzhou': 31, 'Sanya': 32, 'Shanghai': 33, 'Shantou': 34, 'Shenyang': 35,
    'Shenzhen': 36, 'Shijiazhuang': 37, 'Suzhou': 38, 'Taiyuan': 39, 'Taizhou': 40,
    'Tangshan': 41, 'Tianjin': 42, 'Wenzhou': 43, 'Wuhan': 44, 'Xiamen': 45,
    'Xianggang': 46, 'Xian': 47, 'Xining': 48, 'Yangzhou': 49, 'Yinchuan': 50,
    'Zhengzhou': 51, 'Zhongshan': 52,
    'Jiaxing': 53, 'Jinhua': 54, 'Nantong': 55, 'Qingdao':56,
    'Shaoxing': 57,  'Wuxi': 58, 'Wuhu': 59,
    'Xuzhou': 60, 'Yantai': 61, 'Zhuhai': 62,
    'ZhangUkrineWAR2022': 1000,
    'ZhangUkrineWAR2023': 1001

}
City_latitudes = np.load('/home/dell/lsq/LSQNetModel_generalize/Data/City_latitudes.npy')  # [name, longtitude, latitude]
LATRANGE = np.max(City_latitudes[:,1].astype(float)) -  np.min(City_latitudes[:,1].astype(float))
City_latitudes = City_latitudes.tolist()
class Images:
    def __init__(self, img_data: ndarray = None, img_path: str = None):
        assert (img_data is not None) or (img_path is not None), 'No img data or img path input!'
        if img_path is not None:
            self.img_path = img_path
            self.get_info()

        self.img_data = img_data if img_data is not None else self.read_img()
        self.max_value, self.min_value = self.get_range()
        self.pre_process()

    def read_img(self):
        assert self.is_image_file(), 'The file extension is not read-able!'
        if os.path.splitext(self.img_path)[-1] is 'tif':
            data = tif.imread(self.img_path)
        else:
            data = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
        return data

    def pre_process(self):
        imgdata = self.img_data
        imgdata = np.nan_to_num(imgdata)
        imgdata[(imgdata >= 10000) | (imgdata <= -10000)] = 0.
        self.img_data = imgdata

    def is_image_file(self):
        filename = self.img_path
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def normalize(self,
                  method: str = 'max_min',
                  all_same_ignore: bool = False):
        """
        :param method: max_min or mean_std, the way to normalize
        :param path: file path of the data, will help to locate the image which has all-same data
        :param all_same_ignore: see the return explanation
        :return: if max=min, assert error if all_same_ignore is False, return input data if all_same_ignore is True
                else return normalized data
        """
        assert method in ['max_min', 'mean_std'], 'only mean_std or max_min method is available!'
        path = self.img_path
        array = np.copy(self.img_data)
        mx = np.max(array)
        mn = np.min(array)
        if mx == mn:
            if all_same_ignore is False:
                assert mx != mn, 'All-slice same data ===>' + path
        if method is 'max_min':
            array = (array - mn) / (mx - mn)
        elif method is 'mean_std':
            array_mean = np.mean(array)
            array_std = np.std(array, ddof=1)
            array = (array - array_mean) / array_std
        self.img_data = array

    def get_range(self):
        self.max_value = np.nanmax(self.img_data)
        self.min_value = np.nanmin(self.img_data)
        return self.max_value, self.min_value

    def cat_image(self, cat_data: ndarray = None, axis: int = 2, cat_pos: str = 'after'):
        assert cat_data is not None, 'No data to be concat!'
        assert cat_pos in ['after', 'before'], 'only cat in [ "after", "before" ] ways'
        if len(cat_data.shape) == 2 and axis == 2:
            cat_data = cat_data[:, :, np.newaxis]
        if cat_pos is 'after':
            self.img_data = np.concatenate((self.img_data, cat_data), axis=axis)
        if cat_data is 'before':
            self.img_data = np.concatenate((cat_data, self.img_data), axis=axis)

    def cat_images(self, cat_datas: list = None, **kwargs):
        assert cat_datas is not None, 'No datas to be concat!'
        for cat_data in cat_datas:
            self.cat_image(cat_data, **kwargs)

    def get_info(self):
        self.file_name = (self.img_path.split('/')[-1]).split('.')[0]
        # self.file_name = (self.file_name.split('\\')[-1])
        self.city_name = self.file_name.split('_')[0]
        # print('------------------', self.file_name, '-------------------')
        self.image_id = [int(city_dict[self.city_name]), int(self.file_name.split('_')[1]),
                         int(self.file_name.split('_')[2])]

    def get_latnseason(self, buffer_pixels = 20, latnseason=1):
        if latnseason == 1:
            storey_fatctor = buffer_pixels
            result = []  #[days to summer solstice, latitude]
            if 'spring' in self.img_path or 'fall' in self.img_path:
                result.append(90.0)
            elif 'winter' in self.img_path:
                result.append(180.0)
            elif 'summer' in self.img_path:
                result.append(0.0)
            # print(self.img_path)
            assert len(result) == 1
            for city_lat in City_latitudes:
                if city_lat[0] == self.city_name:
                    result.append(city_lat[1])
            result = np.array(result).astype('float')
            sun_angle = (1 - result[0]/90) * 23.5
            sun_elevation_noon = 90 - ((result[1] - sun_angle) if (result[1] - sun_angle) > 0 else 0)
            tan_factor = storey_fatctor/(np.tan(sun_elevation_noon))
            result[0] /= 180.0
            result[1] /= LATRANGE
            result = np.array(result).astype(np.float32)
        elif latnseason == 2:
            storey_fatctor = buffer_pixels
            result = []  # [latitude]
            for city_lat in City_latitudes:
                if city_lat[0] == self.city_name:
                    result.append(city_lat[2])
            result = np.array(result).astype('float')
            sun_angle = - 15
            sun_elevation_noon = 90 - ((result[0] - sun_angle) if (result[0] - sun_angle) > 0 else 0)
            tan_factor = (np.tan(pi*sun_elevation_noon / 180))
            tan_buffer = storey_fatctor / tan_factor
            result[0] /= LATRANGE
            result = np.array(result).astype(np.float32)
        return result, tan_factor, int(np.around(tan_buffer))

class Labels(Images):
    def __init__(self, lab_data: ndarray = None, lab_path: str = None):
        super(Labels, self).__init__(lab_data, lab_path)

    def pre_process(self):
        return

    def level_labels(self, levels: list = None):
        lab = np.copy(self.img_data)
        for i in range(len(levels) - 1):
            lab = np.where((lab > levels[i]) & (lab <= levels[i+1]), i, lab)
        lab = np.where((lab > levels[i+1]), i+1, lab)
        self.img_data = lab

    def get_footprint(self, background: int = 0):
        data = np.copy(self.img_data)
        fp = np.where(data != background, 1, 0)
        return fp

