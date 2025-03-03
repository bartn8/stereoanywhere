from .base_dataset import BaseDataset
from glob import glob
import os.path as osp
from .frame_utils import *
import numpy as np
import os
import cv2
import torch

class BoosterDataset(BaseDataset):
    def load_data(self, datapath):
        image2_list = sorted(glob(osp.join(datapath, 'balanced/*/camera_00/*.png')))
        image3_list = sorted(glob(osp.join(datapath, 'balanced/*/camera_02/*.png')))

        assert len(image2_list) == len(image3_list), 'Different number of images'

        for i in range(len(image2_list)):            
            _gt = os.path.join(osp.dirname(image2_list[i]).replace('camera_00', ''), 'disp_00.npy')
            _gtr = os.path.join(osp.dirname(image2_list[i]).replace('camera_00', ''), 'disp_02.npy')
            _maskocc = os.path.join(osp.dirname(image2_list[i]).replace('camera_00', ''), 'mask_00.png')
            _maskcat = os.path.join(osp.dirname(image2_list[i]).replace('camera_00', ''), 'mask_cat.png')

            _tmp = [image2_list[i], image3_list[i], _gt, _gtr, _maskocc, _maskcat]
            _tmp_mono = []

            if self.mono is not None:
                _tmp_mono += [image2_list[i].replace('camera_00', f'camera_00_{self.mono}'), image3_list[i].replace('camera_02', f'camera_02_{self.mono}')]

            self.image_list += [ _tmp + [self.scale_factor] + _tmp_mono ]
            self.extra_info += [ image2_list[i] ] # scene and frame_id
    
    def load_sample(self, index):
        data = {}

        data['im2'] = read_gen(self.image_list[index][0])
        data['im3'] = read_gen(self.image_list[index][1])
        
        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        if self.mono is not None:
            data['im2_mono'] = read_mono(self.image_list[index][7])
            data['im3_mono'] = read_mono(self.image_list[index][8])

            data['im2_mono'] = np.expand_dims( data['im2_mono'].astype(np.float32), -1)
            data['im3_mono'] = np.expand_dims( data['im3_mono'].astype(np.float32), -1)

        if self.is_test:
            data['im2'] = data['im2'] / 255.0
            data['im3'] = data['im3'] / 255.0

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        data['gt'] = np.load(self.image_list[index][2])
        data['validgt'] = (data['gt']>0).astype(np.uint8)

        data['gt'] = np.expand_dims( np.array(data['gt']).astype(np.float32), -1)
        data['validgt'] = np.expand_dims( np.array(data['validgt']).astype(np.uint8), -1)

        data['gt_right'] = np.load(self.image_list[index][3])
        data['validgt_right'] = (data['gt_right']>0).astype(np.uint8)

        data['gt_right'] = np.expand_dims( np.array(data['gt_right']).astype(np.float32), -1)
        data['validgt_right'] = np.expand_dims( np.array(data['validgt_right']).astype(np.uint8), -1)
        
        data['maskocc'] = np.expand_dims( np.array(read_gen(self.image_list[index][4])).astype(np.uint8), -1)
        data['maskocc'] = np.where(data['maskocc'] == 0, 1, 0)# 1 if occluded, 0 otherwise
        data['maskocc'] = np.array(data['maskocc']).astype(np.uint8)

        # data['maskcat'] = np.expand_dims( np.array(read_gen(self.image_list[index][5])).astype(np.uint8), -1)

        # resize_factor = self.image_list[index][6]

        for k in data:
            if data[k] is not None:
                if k not in ['gt', 'gt_right', 'validgt', 'validgt_right', 'maskocc', 'maskcat']:
                    data[k] = cv2.resize(data[k], (int(data[k].shape[1]/self.scale_factor), int(data[k].shape[0]/self.scale_factor)), interpolation=cv2.INTER_LINEAR)
                else:
                    data[k] = cv2.resize(data[k], (int(data[k].shape[1]/self.scale_factor), int(data[k].shape[0]/self.scale_factor)), interpolation=cv2.INTER_NEAREST)

                if len(data[k].shape) == 2:
                    data[k] = np.expand_dims(data[k], -1)
                
                if k in ['gt', 'gt_right']:
                    data[k] = data[k] / self.scale_factor
        
        if self.is_test or self.augmentor is None:
            data['im2_aug'] = data['im2']
            data['im3_aug'] = data['im3']
        else:
            im2_mono = data['im2_mono'] if self.mono is not None else None
            im3_mono = data['im3_mono'] if self.mono is not None else None
            augm_data = self.augmentor(data['im2'], data['im3'], im2_mono, im3_mono, gt2=data['gt'], validgt2=data['validgt'], gt3=data['gt_right'], validgt3=data['validgt_right'], maskocc=data['maskocc'])

            for key in augm_data:
                data[key] = augm_data[key]

        for k in data:
            if data[k] is not None:
                data[k] = torch.from_numpy(data[k]).permute(2, 0, 1).float() 
        
        data['extra_info'] = self.extra_info[index]

        return data