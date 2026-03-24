############################# get preditions function ##############################
#### Author: Dr. Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Health, PolyU, Hong Kong
#### Attempt: get the outputs of model layer

import numpy as np
import os

class Get_Features:
    def __init__(self, layer = None, end_no = None,
                 save_path = None, out_or_in = 'in',
                 with_pe = True):
        '''
        Args:
            layer: nn.module
            end_no: int
            save_path: str
        '''
        self.end_no = end_no
        self.layer = layer
        self.feat_list = []
        self.grad_list = []
        self.save_path = save_path
        self.count = 0
        self.out_or_in = out_or_in
        self.with_pe = with_pe

    def save_feats(self, model, feats_in, feats_out):
        if self.out_or_in == 'in':
            feats_np = feats_in[0].detach().cpu().numpy()
        elif self.out_or_in == 'out':
            feats_np = feats_out[0].detach().cpu().numpy()
        else:
            assert print('Error!!!')
        self.count += 1
        print(self.count)
        if np.ndim(feats_np) == 4:
            feats_np = feats_np.reshape((feats_np.shape[1], feats_np.shape[2], feats_np.shape[3]))
        elif np.ndim(feats_np) == 3:
            feats_np = feats_np.reshape((feats_np.shape[1], feats_np.shape[2]))


        if self.with_pe == True:
            self.feat_list.append(feats_np[:self.end_no, :])
        elif self.with_pe == False:
            self.feat_list.append(feats_np)
        else:
            assert print('Error!!!')

    def save_grads(self, model, grads_in, grads_out):
        grads_np = grads_out.detach().cpu().numpy()
        print(grads_np.shape)
        if np.ndim(grads_np) == 4:
            grads_np = grads_np.reshape((grads_np.shape[1], grads_np.shape[2], grads_np.shape[3]))
        grads_np = np.mean(grads_np, axis=1)
        self.grad_list.append(grads_np[:self.end_no])

    def regis_layer(self):
        self.layer.register_forward_hook(self.save_feats)
        #self.layer.register_backward_hook(self.save_grads)

    def get_feats_grads(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        #grad_npy = np.concatenate(self.grad_list)

        from sklearn.decomposition import PCA
        pca_1 = PCA(n_components=1, svd_solver='full')
        pca_2 = PCA(n_components=2, svd_solver='full')
        pca_3 = PCA(n_components=3, svd_solver='full')
        feat_npy_1d = []
        feat_npy_2d = []
        feat_npy_3d = []

        for feat_i in self.feat_list:
            print(feat_i.shape)
            feat_npy_1d.append(pca_1.fit_transform(feat_i))
            feat_npy_2d.append(pca_2.fit_transform(feat_i))
            if feat_i.shape[0] < 3:
                feat_npy_3d.append(feat_i[:, :3])
            else:
                feat_npy_3d.append(pca_3.fit_transform(feat_i))

        import joblib

        joblib.dump(self.feat_list, os.path.join(self.save_path, 'feats_full.joblib'))
        joblib.dump(feat_npy_1d, os.path.join(self.save_path, 'feats_1d.joblib'))
        joblib.dump(feat_npy_2d, os.path.join(self.save_path, 'feats_2d.joblib'))
        joblib.dump(feat_npy_3d, os.path.join(self.save_path, 'feats_3d.joblib'))



if __name__ == '__main__':
    print('sucess!!!')