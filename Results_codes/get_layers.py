############################# get layers value function ##############################
#### Author: Dr. Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Health, PolyU, Hong Kong
#### Attempt: get layers value

import numpy as np

def get_layers(layer = None, save_path = None):
    layer_value = layer.weight.detach().cpu().numpy()
    layer_value = np.transpose(layer_value)
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, 'layers.npy'), layer_value)


if __name__ == '__main__':
    print('success!!')