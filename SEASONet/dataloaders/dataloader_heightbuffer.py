import torch.utils.data as data
import torch
import tifffile as tif
from Data.LabelTargetProcessor import LabelTarget
from augmentations.diyaugmentation import my_segmentation_transforms, my_segmentation_transforms_crop
import os
import numpy as np
from os.path import join
from Data.ImageProcessor import Images, Labels
import gc
import types




def make_dataset(filepath, split=[0.7, 0.1, 0.2], test_only=False, buffer_assist=False, latnseason=0, supervision=False):
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
        if supervision:
            lab_sup = [join(filepath, 'label_sup', name) for name in os.listdir(join(filepath, 'label_sup'))]
        if buffer_assist:
            pre = [join(filepath, 'pred', name) for name in os.listdir(join(filepath, 'pred'))]

        vvh = [join(filepath, 'image', 'VVVH', name) for name in os.listdir(join(filepath, 'image', 'VVVH'))]

        spr = [join(filepath, 'image', 'season', 'spring', name) for name in
               os.listdir(join(filepath, 'image', 'season', 'spring'))]
        smr = [join(filepath, 'image', 'season', 'summer', name) for name in
               os.listdir(join(filepath, 'image', 'season', 'summer'))]
        fal = [join(filepath, 'image', 'season', 'fall', name) for name in
               os.listdir(join(filepath, 'image', 'season', 'fall'))]
        wnt = [join(filepath, 'image', 'season', 'winter', name) for name in
               os.listdir(join(filepath, 'image', 'season', 'winter'))]

        # img = [join(filepath, 'img', name) for name in os.listdir(join(filepath, 'img'))]
        # ssn = [join(filepath, 'ssn', name) for name in os.listdir(join(filepath, 'ssn'))]
        # lab = [join(filepath, 'Label_SR25', name) for name in os.listdir(join(filepath, 'Label_SR25'))]
        # vvh = [join(filepath, 'vvh', name) for name in os.listdir(join(filepath, 'vvh'))]

    assert len(img) == len(lab)
    if not test_only:

        assert len(img) == len(vvh)
        assert len(img) == len(ssn)
    if buffer_assist:
        assert len(pre) == len(img)
        pre.sort()
        pre = np.array(pre)
    if supervision:
        assert len(lab_sup) == len(lab)
        lab_sup.sort()
        lab_sup = np.array(lab_sup)
    assert len(img) == len(spr)
    assert len(img) == len(smr)
    assert len(img) == len(fal)
    assert len(img) == len(wnt)

    img.sort()
    ssn.sort()
    lab.sort()
    vvh.sort()

    spr.sort()
    smr.sort()
    fal.sort()
    wnt.sort()

    num_samples = len(img)
    img = np.array(img)
    ssn = np.array(ssn)
    lab = np.array(lab)
    vvh = np.array(vvh)

    spr = np.array(spr)
    smr = np.array(smr)
    fal = np.array(fal)
    wnt = np.array(wnt)
    seasons = np.concatenate((spr, smr, fal, wnt), axis=0)
    lab_seasons = np.concatenate((lab, lab, lab, lab), axis=0)
    if test_only:
        vvh = img
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

    imgt = np.vstack((img[train], ssn[train], vvh[train],
                      spr[train], smr[train], fal[train], wnt[train])).T
    labt = lab[train]

    imgv = np.vstack((img[val], ssn[val], vvh[val],
                      spr[val], smr[val], fal[val], wnt[val])).T
    labv = lab[val]


    imgte = np.vstack((img[test], ssn[test], vvh[test],
                       spr[test], smr[test], fal[test], wnt[test])).T
    labte = lab[test]

    if buffer_assist:
        pret = pre[train]
        prev = pre[val]
        prete = pre[test]
        return imgt, labt, imgv, labv, imgte, labte, pret, prev, prete


    if latnseason == 1:
        num_samples *= 4
        seqpath_seasons = join(filepath, 'seq_seasons.txt')
        if os.path.exists(seqpath_seasons):
            seq_seasons = np.loadtxt(seqpath_seasons, delimiter=',')
        else:
            seq_seasons = np.random.permutation(num_samples)
            np.savetxt(seqpath_seasons, seq_seasons, fmt='%d', delimiter=',')
        seq_seasons = np.array(seq_seasons, dtype='int32')
        num_train = int(num_samples * split[0])  # the same as floor
        num_val = int(num_samples * split[1])

        train = seq_seasons[0:num_train]
        val = seq_seasons[num_train:(num_train + num_val)]
        test = seq_seasons[num_train + num_val:]

        imgt = seasons[train]
        labt = lab_seasons[train]

        imgv = seasons[val]
        labv = lab_seasons[val]

        imgte = seasons[test]
        labte = lab_seasons[test]
        if supervision:
            sup_seasons = np.concatenate((lab_sup, lab_sup, lab_sup, lab_sup), axis=0)
            lab_supt = sup_seasons[train]
            lab_supv = sup_seasons[val]
            lab_supte = sup_seasons[test]
            return imgt, labt, imgv, labv, imgte, labte, lab_supt, lab_supv, lab_supte

    if supervision:
        lab_supt = lab_sup[train]
        lab_supv = lab_sup[val]
        lab_supte = lab_sup[test]
        return imgt, labt, imgv, labv, imgte, labte, lab_supt, lab_supv, lab_supte

    return imgt, labt, imgv, labv, imgte, labte


class MaskRcnnDataloader(data.Dataset):
    def __init__(self, imgpath, labpath, prepath=None, lab_sup_path= None,
                 augmentations=False, area_thd=4,
                 label_is_nos=True, footprint_mode=False, seasons_mode: str = None, sar_data=False, RGB_mode=False,
                 if_buffer=False, buffer_pixels=0, buffer_assist=False, latnseason=False, storey_fatctor=0, supervision=False):  # data loader #params nrange
        assert seasons_mode in ['compress', 'intact', 'Spring', 'Summer', 'Autumn', 'Winter', 'seasons', None, 'None'], 'season mode must in [compress, intact] or None'
        super(MaskRcnnDataloader, self).__init__()
        self.imgpath = imgpath
        self.labpath = labpath
        if buffer_assist:
            assert prepath is not None
            self.prepath = prepath
        if supervision:
            assert lab_sup_path is not None
            self.sup_path = lab_sup_path
        self.augmentations = augmentations
        self.area_thd = area_thd
        self.label_is_nos = label_is_nos
        self.footprint_mode = footprint_mode
        self.seasons_mode = seasons_mode
        self.sar_data = sar_data
        self.RGB_mode = RGB_mode
        self.if_buffer = if_buffer
        self.buffer_pixels = buffer_pixels
        self.buffer_assist = buffer_assist
        self.supervision = supervision
        self.latnseason = latnseason
        assert isinstance(storey_fatctor, int)
        self.storey_fatctor = storey_fatctor

    def __getitem__(self, index):
        muxpath_ = self.imgpath[index, 0]
        mux_img = Images(img_path=muxpath_)
        mux_img.normalize(method='mean_std')
        img = Images(img_data=mux_img.img_data, img_path=muxpath_)

        lab_img = Labels(lab_path=self.labpath[index])
        lab = lab_img.img_data
        if self.seasons_mode == 'intact' or self.seasons_mode == 'seasons':
            spr_img = Images(img_path=self.imgpath[index, 3])
            spr_img.normalize(method='mean_std')
            smr_img = Images(img_path=self.imgpath[index, 4])
            smr_img.normalize(method='mean_std')
            fal_img = Images(img_path=self.imgpath[index, 5])
            fal_img.normalize(method='mean_std')
            wnt_img = Images(img_path=self.imgpath[index, 6])
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
            spr_img = Images(img_path=self.imgpath[index, 3])
            spr_img.normalize(method='mean_std')
            img = spr_img
        if self.seasons_mode == 'Summer':
            smr_img = Images(img_path=self.imgpath[index, 4])
            smr_img.normalize(method='mean_std')
            img = smr_img
        if self.seasons_mode == 'Autumn':
            fal_img = Images(img_path=self.imgpath[index, 5])
            fal_img.normalize(method='mean_std')
            img = fal_img
        if self.seasons_mode == 'Winter':
            wnt_img = Images(img_path=self.imgpath[index, 6])
            wnt_img.normalize(method='mean_std')
            img = wnt_img

        if self.footprint_mode:
            footprint = lab_img.get_footprint(background=0)
            img.cat_image(footprint, cat_pos='before')

        if self.sar_data:
            vvh_img = Images(img_path=self.imgpath[index, 2])
            vvh_img.normalize(method='mean_std')
            img.cat_image(vvh_img.img_data)

        if self.RGB_mode:
            img.img_data = mux_img.img_data[:, :, 0:3]

        label_class = LabelTarget(label_data=lab)
        if self.supervision:
            sup = Labels(lab_path=self.sup_path[index])
            sup = sup.img_data
            sup = torch.tensor(sup.copy(), dtype=torch.float)
            sup = torch.unsqueeze(sup, 0)

        if self.buffer_assist == 'pre_assist':  # predata: nos(story)
            pre_img = Labels(lab_path=self.prepath[index])
            pre = pre_img.img_data
            label_class = LabelTarget(label_data=lab, height_data=pre)

        if self.buffer_assist == 'sup_assist':  # supdata: height(m)
            assert self.supervision is True
            label_class = LabelTarget(label_data=lab, height_data=sup)

        if self.latnseason != 0:
            latnseason, tan_factor, tan_buffer = img.get_latnseason(buffer_pixels=self.buffer_pixels, latnseason=self.latnseason)
            #  latnseason == 2: 使用label或者height数据的信息结合tan_factor得到bufferpixels的值
            #  latnseason == 1: 使用根据固定的buffer值与纬度、季节信息结合得到的tanbuffer得到bufferpixels的值，使用这个分支必须要将季节数据分批输入
            target = label_class.to_target(image_id=lab_img.image_id,
                                           file_name=lab_img.file_name,
                                           if_buffer_proposal=self.if_buffer,
                                           area_thd=self.area_thd,
                                           mask_mode='value',
                                           background=0,
                                           label_is_value=self.label_is_nos,
                                           buffer_pixels=self.buffer_pixels if self.latnseason == 2 else tan_buffer,
                                           buffer_assist=self.buffer_assist,
                                           tan_factor=tan_factor if self.latnseason == 2 else None
                                           )

            if self.latnseason == 1:
                building_sum = target["nos"].shape[0]
                latnseason = [latnseason for i in range(building_sum)]
                target['latnseason'] = torch.FloatTensor(latnseason.copy())

        else:
            target = label_class.to_target(image_id=lab_img.image_id,
                                           file_name=lab_img.file_name,
                                           if_buffer_proposal=self.if_buffer,
                                           area_thd=self.area_thd,
                                           mask_mode='value',
                                           background=0,
                                           label_is_value=self.label_is_nos,
                                           buffer_pixels=self.buffer_pixels,
                                           buffer_assist=self.buffer_assist,
                                           )

        img = img.img_data
        assert (np.isnan(img)).any() is not True, 'NaN in Data!'
        img = img.transpose((2, 0, 1))  # H W C => C H W
        img = torch.tensor(img.copy(), dtype=torch.float)
        if self.supervision:
            return img, target, sup
        else:
            return img, target

    def __len__(self):
        return len(self.imgpath)

    def collate_fn(self, batch):
        img_list, target_list, sup_list = [], [], []
        if self.supervision:
            for img, target, sup in batch:
                img_list.append(img)
                target_list.append(target)
                sup_list.append(sup)
            return img_list, target_list, sup_list
        else:
            for img, target in batch:
                img_list.append(img)
                target_list.append(target)
            return img_list, target_list
