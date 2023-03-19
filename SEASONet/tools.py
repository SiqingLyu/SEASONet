"""
Date: 2022.9.20
Author: Lv
this class aims to create a class which contains all the operation on Labels
Last coded date: 2022.12.20
"""
import numpy as np
import os
import tifffile as tif
import matplotlib.pyplot as plt

def get_range(x):
    ma = np.nanmax(x)
    mi = np.nanmin(x)
    return mi, ma


def read_tif(tif_path):
    dataset = tif.imread(tif_path)
    return dataset


def cal_rmse(y_true, ypred):
    diff=y_true.flatten()-ypred.flatten()
    return np.sqrt(np.mean(diff*diff))


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
    return path


def plot_figure(data, title = ''):
    plt.figure()
    plt.title(title, fontsize=12, fontweight='bold')
    plt.imshow(data)

def Normalize(array):
    '''
    Normalize the array
    '''
    data = np.copy(array)
    mx = np.nanmax(data)
    mn = np.nanmin(data)
    assert mx != mn, 'All-same value slice encountered!'
    t = (data-mn)/(mx-mn)
    return t


def file_name_tif(file_dir):
    '''
    eg: Listfile, allFilename = file_name(r'/www/lsq/optical')
    only for tif files
    :param file_dir: str
    :return: two List: a list of file absolute path & a list of file with no suffix
    '''
    if (os.path.isdir(file_dir)):
        L = []
        allFilename = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if file.split('.')[-1] != 'tif':
                    continue
                formatName = os.path.splitext(file)[1]
                fileName = os.path.splitext(file)[0]
                allFilename.append(fileName)
                if (formatName == '.tif'):
                    tempPath = os.path.join(root, file)
                    L.append(tempPath)
        return L, allFilename
    else:
        print('must be folder path')


def file_name_shp(file_dir):
    '''
    eg: Listfile, allFilename = file_name(r'/www/lsq/optical')
    only for shp files
    :param file_dir: str
    :return: two List: a list of file absolute path & a list of file with no suffix
    '''
    if (os.path.isdir(file_dir)):
        L = []
        allFilename = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if file.split('.')[-1] != 'shp':
                    continue
                formatName = os.path.splitext(file)[1]
                fileName = os.path.splitext(file)[0]
                allFilename.append(fileName)
                if (formatName == '.shp'):
                    tempPath = os.path.join(root, file)
                    L.append(tempPath)
        return L, allFilename
    else:
        print('must be folder path')


def file_name(file_dir, suffix):
    '''
    eg: Listfile, allFilename = file_name(r'/www/lsq/optical')
    only for shp files
    :param file_dir: str
    :return: two List: a list of file absolute path & a list of file with no suffix
    '''
    if (os.path.isdir(file_dir)):
        L = []
        allFilename = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if file.split('.')[-1] != suffix:
                    continue
                formatName = os.path.splitext(file)[1]
                fileName = os.path.splitext(file)[0]
                allFilename.append(fileName)
                if (formatName == ('.' + suffix)):
                    tempPath = os.path.join(root, file)
                    L.append(tempPath)
        return L, allFilename
    else:
        print('must be folder path')


def get_filelist_keywords(data_path, keywords):
    FileList, allFileid = file_name(data_path, suffix = 'shp')
    #keywords = ['after','upper']
    for keyword in keywords:
        FileList_keyword = []
        for file_path in FileList:
            if keyword in file_path:
                FileList_keyword.append(file_path)
        FileList = FileList_keyword
    return FileList


def transpose_list(list):
    return zip(*list)


def get_allzero_image_name(image_path):
    Listfile, allFilename = file_name_tif(image_path)
    names = []
    for ii in range(len(allFilename)):
        tif_filepath = Listfile[ii]
        tif_fileid = allFilename[ii]
        img_data = read_tif(tif_filepath)
        img_data = np.nan_to_num(img_data)
        mx = np.nanmax(img_data)
        mn = np.nanmin(img_data)
        if mx == mn:
            names.append(tif_fileid)
    return names


def get_different_tifname(refer_folder, compare_folder):
    _, allFilename_refer = file_name_tif(refer_folder)
    _, allFilename_compare = file_name_tif(compare_folder)
    dif_names = []
    for item in allFilename_refer:
        if item in allFilename_compare:
            pass
        else:
            dif_names.append(item)
    print('未出现在参考文件夹中的文件名：\n', dif_names)
    return dif_names


def label_round(tif_data):
    tif_data = np.round(tif_data)
    return tif_data

def label_round_batch(path, save_path):
    make_dir(save_path)
    filepath_list, filename_list = file_name_tif(path)
    for ii in range(len(filename_list)):
        filename = filename_list[ii]
        filepath = filepath_list[ii]
        data = tif.imread(filepath)
        data = np.round(data)
        tif.imsave(os.path.join(save_path, filename + '.tif'), data)


def pred_threshold(data, threshold):
    return np.where(data > threshold, data, 0)


def pred_threshold_batch(path, save_path, threshold, prefix = None):
    make_dir(save_path)
    filepath_list, filename_list = file_name_tif(path)
    for ii in range(len(filename_list)):
        filename = filename_list[ii]
        if prefix is not None and prefix not in filename:
            continue
        filepath = filepath_list[ii]
        data = tif.imread(filepath)
        data = np.where(data > threshold, data, 0)
        tif.imsave(os.path.join(save_path, filename + '.tif'), data)


def rewrite_tif(data, filename, save_path):
    make_dir(save_path)
    tif.imsave(os.path.join(save_path, filename + '.tif'), data)


def rewrite_tif_batch(path, save_path):
    make_dir(save_path)
    filepath_list, filename_list = file_name_tif(path)
    for ii in range(len(filename_list)):
        filename = filename_list[ii]
        filepath = filepath_list[ii]
        data = tif.imread(filepath)
        tif.imsave(os.path.join(save_path, filename + '.tif'), data)
