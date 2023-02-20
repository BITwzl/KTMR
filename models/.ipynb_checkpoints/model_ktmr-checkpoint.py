from models.model_plain import ModelPlain
from utils.utils_rec import consistency_layer
import h5py
import numpy as np
import torch
from collections import OrderedDict

class KTMR(ModelPlain):

    def __init__(self, opt):
        super().__init__(opt)

        mask_path=self.opt['mask_path']
        try:
            mask = scio.loadmat(mask_path)
            npmask = []
            for k in mask['mask_matrix']:
                npmask.append(k)
            mask = np.array(npmask)

            tenmask = torch.from_numpy(mask)
            tenmask = torch.unsqueeze(tenmask,0)
            self.mask=tenmask
        except:
            mask = h5py.File(mask_path,"r")
            mask = np.array(mask['mask_matrix'])

            tenmask = torch.from_numpy(mask)
            tenmask = torch.unsqueeze(tenmask,0)
            self.mask=tenmask
            
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = consistency_layer(self.L.detach()[0].float().cpu(),
                                          self.H.detach()[0].float().cpu(),
                                          self.mask.squeeze()).reshape(self.L.detach()[0].shape)
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict