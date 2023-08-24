import torch.utils.data as data
import torch
from Data.LabelTargetProcessor import LabelTarget
import os
import numpy as np
from os.path import join
from Data.ImageProcessor import Images, Labels


def make_dataset(filepath, split=[0.7, 0.1, 0.2], test_only=False):
    '''
    :param filepath: the root dir of img, lab and ssn
    :return: img, lab
    '''
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        img = [join(filepath, 'image', 'optical', name) for name in os.listdir(join(filepath, 'image', 'optical'))]
        ssn = [join(filepath, 'image', 'season', 'all', name) for name in
               os.listdir(join(filepath, 'image', 'season', 'all'))]

        lab = [join(filepath, 'label', name) for name in os.listdir(join(filepath, 'label'))]

        spr = [join(filepath, 'image', 'season', 'spring', name) for name in
               os.listdir(join(filepath, 'image', 'season', 'spring'))]
        smr = [join(filepath, 'image', 'season', 'summer', name) for name in
               os.listdir(join(filepath, 'image', 'season', 'summer'))]
        fal = [join(filepath, 'image', 'season', 'fall', name) for name in
               os.listdir(join(filepath, 'image', 'season', 'fall'))]
        wnt = [join(filepath, 'image', 'season', 'winter', name) for name in
               os.listdir(join(filepath, 'image', 'season', 'winter'))]

    assert len(img) == len(lab)
    if not test_only:
        assert len(img) == len(ssn)

    assert len(img) == len(spr)
    assert len(img) == len(smr)
    assert len(img) == len(fal)
    assert len(img) == len(wnt)

    img.sort()
    ssn.sort()
    lab.sort()

    spr.sort()
    smr.sort()
    fal.sort()
    wnt.sort()

    num_samples = len(img)
    img = np.array(img)
    ssn = np.array(ssn)
    lab = np.array(lab)

    spr = np.array(spr)
    smr = np.array(smr)
    fal = np.array(fal)
    wnt = np.array(wnt)
    seasons = np.concatenate((spr, smr, fal, wnt), axis=0)
    lab_seasons = np.concatenate((lab, lab, lab, lab), axis=0)
    if test_only:
        ssn = img
    # generate sequence
    # load the path
    seqpath = join(filepath, 'seq.txt')
    if os.path.exists(seqpath):
        seq = np.loadtxt(seqpath, delimiter=',')
    else:
        seq = np.random.permutation(num_samples)
        np.savetxt(seqpath, seq, fmt='%d', delimiter=',')
    seq = np.array(seq, dtype='int32')




    num_train = int(num_samples * split[0])  # the same as floor
    num_val = int(num_samples * split[1])

    train = seq[0:num_train]
    val = seq[num_train:(num_train + num_val)]
    test = seq[num_train + num_val:]

    imgt = np.vstack((img[train], ssn[train],
                      spr[train], smr[train], fal[train], wnt[train])).T
    labt = lab[train]

    imgv = np.vstack((img[val], ssn[val],
                      spr[val], smr[val], fal[val], wnt[val])).T
    labv = lab[val]

    imgte = np.vstack((img[test], ssn[test],
                       spr[test], smr[test], fal[test], wnt[test])).T
    labte = lab[test]

    return imgt, labt, imgv, labv, imgte, labte


class MaskRcnnDataloader(data.Dataset):
    def __init__(self, imgpath, labpath, prepath=None, lab_sup_path= None,
                 augmentations=False, area_thd=4,
                 label_is_nos=True, seasons_mode: str = None,
                 if_buffer=False, buffer_pixels=0):  # data loader #params nrange
        assert seasons_mode in ['compress', 'intact', 'Spring', 'Summer', 'Autumn', 'Winter', 'seasons', None, 'None'],\
            'season mode must in [compress, intact] or None'
        super(MaskRcnnDataloader, self).__init__()
        self.imgpath = imgpath
        self.labpath = labpath
        self.augmentations = augmentations
        self.area_thd = area_thd
        self.label_is_nos = label_is_nos
        self.seasons_mode = seasons_mode
        self.if_buffer = if_buffer
        self.buffer_pixels = buffer_pixels

    def __getitem__(self, index):
        muxpath_ = self.imgpath[index, 0]
        mux_img = Images(img_path=muxpath_)
        mux_img.normalize(method='mean_std')
        img = Images(img_data=mux_img.img_data, img_path=muxpath_)

        lab_img = Labels(lab_path=self.labpath[index])
        lab = lab_img.img_data
        if self.seasons_mode == 'intact' or self.seasons_mode == 'seasons':
            spr_img = Images(img_path=self.imgpath[index, 2])
            spr_img.normalize(method='mean_std')
            smr_img = Images(img_path=self.imgpath[index, 3])
            smr_img.normalize(method='mean_std')
            fal_img = Images(img_path=self.imgpath[index, 4])
            fal_img.normalize(method='mean_std')
            wnt_img = Images(img_path=self.imgpath[index, 5])
            wnt_img.normalize(method='mean_std')
            cat_data_list = [smr_img.img_data, fal_img.img_data, wnt_img.img_data]
            spr_img.cat_images(cat_datas=cat_data_list)
            ssn = spr_img.img_data
            if self.seasons_mode == 'intact':
                img.cat_image(ssn)
            elif self.seasons_mode == 'seasons':
                img.img_data = ssn

        if self.seasons_mode == 'compress':
            ssn_img = Images(img_path=self.imgpath[index, 1])
            ssn_img.normalize(method='mean_std')
            ssn = ssn_img.img_data
            img.cat_image(ssn)

        if self.seasons_mode == 'Spring':
            spr_img = Images(img_path=self.imgpath[index, 2])
            spr_img.normalize(method='mean_std')
            img = spr_img
        if self.seasons_mode == 'Summer':
            smr_img = Images(img_path=self.imgpath[index, 3])
            smr_img.normalize(method='mean_std')
            img = smr_img
        if self.seasons_mode == 'Autumn':
            fal_img = Images(img_path=self.imgpath[index, 4])
            fal_img.normalize(method='mean_std')
            img = fal_img
        if self.seasons_mode == 'Winter':
            wnt_img = Images(img_path=self.imgpath[index, 5])
            wnt_img.normalize(method='mean_std')
            img = wnt_img

        label_class = LabelTarget(label_data=lab)

        target = label_class.to_target(image_id=lab_img.image_id,
                                       file_name=lab_img.file_name,
                                       if_buffer_proposal=self.if_buffer,
                                       area_thd=self.area_thd,
                                       mask_mode='value',
                                       background=0,
                                       label_is_value=self.label_is_nos,
                                       buffer_pixels=self.buffer_pixels,
                                       )

        img = img.img_data
        assert (np.isnan(img)).any() is not True, 'NaN in Data!'
        img = img.transpose((2, 0, 1))  # H W C => C H W
        img = torch.tensor(img.copy(), dtype=torch.float)

        return img, target

    def __len__(self):
        return len(self.imgpath)

    def collate_fn(self, batch):
        img_list, target_list, sup_list = [], [], []
        for img, target in batch:
            img_list.append(img)
            target_list.append(target)
        return img_list, target_list
