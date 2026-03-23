#############################  visiualizing whole-slide image ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Health, PolyU, Hong Kong
#### Attempt: visiualizing whole-slide image
import json

import numpy as np
import cv2
from skimage import io, color
import PIL
import matplotlib.pyplot as plt
import os
import natsort
import re




def draw_grid(img, grid_size=(224, 224), color=(0, 255, 0), thickness=1):
    """
    在图像上画网格线
    img: BGR 图像 (OpenCV)
    grid_size: 每个格子多少像素
    color: 线条颜色 (B, G, R)
    thickness: 线条宽度
    """
    h, w = img.shape[:2]

    # 画竖线
    for x in range(0, w, grid_size[1]):
        cv2.line(img, (x, 0), (x, h), color, thickness)

    # 画横线
    for y in range(0, h, grid_size[0]):
        cv2.line(img, (0, y), (w, y), color, thickness)

    return img





'''
visual_recon_wsi函数的使用方法：
情况 1：target_map_mode = False，target_group_mode = False，add_map_color = 0，恢复WSI原图模样；
情况 2：target_map_mode = False，target_group_mode = False，add_map_color = 非0，WSI组织区域是原图，背景区域是各种颜色图；
情况 3：target_map_mode = True，target_group_mode = False，add_map_color = 非0，target_map_color = 非0，WSI组织和背景是具有对比的颜色图；
情况 4：target_map_mode = False，target_group_mode = True，target_group_wegt = nx1数组，WSI的组织按n-1类去上色，背景是第n类颜色；
情况 5：target_map_mode = False，target_group_mode = False, grad_mode = True，add_map_color = 0 颜色图和原图混合, add_map_color = 13
情况 6：color_grad = True, heat_map_mode = True
情况 7：
情况 8：
情况 9：target_map_mode = True，target_group_mode = True，我设置为assert，直接报错；
'''

def visual_recon_wsi(read_path = None, save_path = None, show_mode = False, add_map_color = 4, target_class = 1,
                     target_map_mode = False, target_map_color = 5, target_group_mode = False,
                     target_group_wegt = None, group_color = None, grad_mode = False,
                     grad_color = None, grad_class = 0, heat_map_mode = False, non_class_color = 2,
                     heat_wegt = None, heat_class = 1, map_color_str = 'HOT', cam_save_path = None,
                     grid_stat = False, grid_color = (0, 0, 0), grid_size = (224, 224),
                     grid_thick = 10, re_list_mode = '2', img_size = (224, 224, 3)):
    '''
    :param read_path:
    :param save_path:
    :param show_mode:
    :param add_map_color: 0: white, 1: [86, 152, 196], 2: [10, 130, 124], 3:[244, 163, 46], 4:[15 10 14], 5:[171, 130, 124]
    :param target_map_mode: boolean, True: 显示组织区域为原始图像，False: 组织区域是一种特殊颜色map;
    :param target_map_color: If target_map_mode is True, it is the effective.
    :param target_group_mode: target_map_mode和target_group_mode只能一个是True，或者两个都是False
    :param target_group_wegt: list
    :param group_color: list
    :param grad_mode:
    :param grad_class:
    :param heat_map_mode:
    :param color_grad:
    :param heat_wegt:
    :param
    :param
    :return:
    '''
    assert len(heat_wegt) == len(target_group_wegt)

    color_map_dict = {
        '0': [255, 255, 255], '1': [86, 152, 196], '2': [10, 130, 124], '3': [244, 163, 46],
        '4': [15, 10, 14], '5': [171, 130, 124], '6': [144, 238, 144], '7': [255, 183, 178],
        '8': [255, 214, 165], '9': [255, 229, 180], '10': [188, 226, 250], '11': [205, 180, 219],
        '12': [255, 0, 0], '13': [0, 0, 0]
    }

    patch_name_list = natsort.natsorted(os.listdir(read_path), alg=natsort.ns.PATH)
    if re_list_mode == '1':  ### 格式10_10_3_4
        new_patch_name = re.split('_', patch_name_list[0][:-4])
        w_step = int(new_patch_name[1])
        h_step = int(new_patch_name[0])
    elif re_list_mode == '2': ### 格式0_3_4_10x10
        new_patch_name = re.findall('\d+', patch_name_list[0])
        w_step = int(new_patch_name[4])
        h_step = int(new_patch_name[3])
    else:
        assert print('Errors!!!')



    heat_map = np.zeros((h_step, w_step))
    print(w_step, h_step)

    sub_img_list = []
    img_list = []
    count_target = 0
    for h_i in range(h_step):
        for w_i in range(w_step):
            if re_list_mode == '1':
                patch_i_name = str(h_step) + '_' + str(w_step) + '_' + str(h_i) + '_' + str(w_i) + '.jpg'
            elif re_list_mode == '2':
                patch_i_name = new_patch_name[0] + '_' + str(h_i) + '_' + str(w_i) + '_' + str(h_step) + 'x' + str(w_step) + '.jpg'
            if patch_i_name in patch_name_list:
                print(count_target)
                patch_read_path = os.path.join(read_path, patch_i_name)
                print(patch_read_path)
                ####
                if target_map_mode == True and target_group_mode == False and grad_mode == False and heat_map_mode == False:
                    patch_img = np.uint8(np.zeros(img_size, dtype=np.uint8) + color_map_dict[str(target_map_color)])
                    patch_img = patch_img[:, :, ::-1]
                ####
                elif target_map_mode == False and target_group_mode == True and grad_mode == False and heat_map_mode == False:
                    if target_class == target_group_wegt[count_target]:
                        patch_img = np.uint8(np.zeros(img_size, dtype=np.uint8)
                                          + color_map_dict[str(group_color[1])])
                    else:
                        patch_img = np.uint8(np.zeros(img_size, dtype=np.uint8)
                                             + color_map_dict[str(group_color[2])])
                    patch_img = patch_img[:, :, ::-1]
                ####
                elif target_map_mode == False and target_group_mode == False and grad_mode == True and heat_map_mode == False:
                    if grad_class == target_group_wegt[count_target]:
                        patch_img = np.uint8(
                            (np.zeros(img_size, dtype=np.uint8) + grad_color)
                            * heat_wegt[count_target]
                        )
                        patch_img = patch_img[:, :, ::-1]
                    else:
                        patch_img = np.uint8(np.zeros(img_size, dtype=np.uint8) + color_map_dict[str(non_class_color)])
                        patch_img = patch_img[:, :, ::-1]
                ####
                elif target_map_mode == False and target_group_mode == False and grad_mode == False and heat_map_mode == True:
                    patch_img = cv2.imread(patch_read_path)
                    if heat_class == target_group_wegt[count_target]:
                        heat_map[h_i, w_i] = heat_wegt[count_target]
                    else:
                        pass
                ####
                elif target_map_mode == False and target_group_mode == False and grad_mode == False and heat_map_mode == False:
                    patch_img = cv2.imread(patch_read_path)
                ####
                else:
                    assert print("error!!! The target_map_mode and target_group_mode are only one True，or are both False.")
                ####
                sub_img_list.append(patch_img)
                count_target += 1
            else:
                add_img = np.uint8(np.zeros(img_size, dtype=np.uint8) + color_map_dict[str(add_map_color)])
                sub_img_list.append(add_img)
        #k = sub_img_list[0]
        #io.imshow(k)
        #io.show()
        sub_img = cv2.hconcat(sub_img_list)
        img_list.append(sub_img)
        sub_img_list = []

    wsi_img = cv2.vconcat(img_list)

    if heat_map_mode == True:
        cam_resz = cv2.resize(heat_map, (wsi_img.shape[1], wsi_img.shape[0]))
        ## cv2.COLORMAP_HOT, cv2.COLORMAP_JET, cv2.COLORMAP_INFERNO, cv2.COLORMAP_MAGMA, cv2.COLORMAP_TURBO
        map_color_dict = {
            'HOT': cv2.COLORMAP_HOT, 'JET': cv2.COLORMAP_JET, 'INFERNO':cv2.COLORMAP_INFERNO,
            'MAGMA': cv2.COLORMAP_MAGMA, 'TURBO': cv2.COLORMAP_TURBO,
        }
        if map_color_str == 'MY_COLOR':
            new_img = np.stack((cam_resz, cam_resz, np.zeros_like(cam_resz)), axis=2)
            wsi_heatm = np.uint8(new_img * [255, 255, 0])
            #wsi_heatm = cv2.applyColorMap(np.uint8(255 * cam_resz), map_color_dict[map_color_str])
            wsi_img = wsi_heatm * 0.6 + wsi_img[:, :, ::-1] # BGR 转 RGB
            wsi_img = wsi_img.astype(np.uint8)
            wsi_img = wsi_img[:, :, ::-1].copy()
            white_heatm = wsi_heatm.copy()
            # mask = cv2.inRange(white_heatm, (0, 0, 0), (30, 30, 30))
            mask = cv2.cvtColor(white_heatm, cv2.COLOR_BGR2GRAY)
            white_heatm[mask == 0] = (255, 255, 255)
            if grid_stat == True:
                white_heatm = draw_grid(white_heatm, grid_size=grid_size, color=grid_color, thickness=grid_thick)
            else:
                pass
            cv2.imwrite(cam_save_path, white_heatm)
        else:
            wsi_heatm = cv2.applyColorMap(np.uint8(255 * cam_resz), map_color_dict[map_color_str])
            wsi_img = wsi_heatm * 0.6 + wsi_img[:, :, ::-1]
            wsi_img = wsi_img.astype(np.uint8)
            wsi_img = wsi_img[:, :, ::-1].copy()
            white_heatm = wsi_heatm.copy()
            #mask = cv2.inRange(white_heatm, (0, 0, 0), (30, 30, 30))
            mask = cv2.cvtColor(white_heatm, cv2.COLOR_BGR2GRAY)
            white_heatm[mask == 0] = (255, 255, 255)
            if grid_stat == True:
                white_heatm = draw_grid(white_heatm, grid_size=grid_size, color=grid_color, thickness=grid_thick)
            else:
                pass
            cv2.imwrite(cam_save_path, white_heatm)
    else:
        pass

    if grid_stat == True:
        wsi_img = draw_grid(wsi_img, grid_size=grid_size, color=grid_color, thickness=grid_thick)
    else:
        pass

    if show_mode == True:
        print(wsi_img.shape)
        plt.figure(dpi=300)
        ###因为cv2存储是BGR，而其他图像库是RGB，需要最后一阶翻转一下；
        io.imshow(wsi_img[:, :, ::-1])
        io.show()
    else:
        pass

    cv2.imwrite(save_path, wsi_img)




'''
visual_recon_wsi函数的使用方法：
情况 1：target_map_mode = False，target_group_mode = False，add_map_color = 0，恢复WSI原图模样；
情况 2：target_map_mode = False，target_group_mode = False，add_map_color = 非0，WSI组织区域是原图，背景区域是各种颜色图；
情况 3：target_map_mode = True，target_group_mode = False，add_map_color = 非0，target_map_color = 非0，WSI组织和背景是具有对比的颜色图；
情况 4：target_map_mode = False，target_group_mode = True，target_group_wegt = nx1数组，WSI的组织按n-1类去上色，背景是第n类颜色；
情况 5：target_map_mode = False，target_group_mode = True, grad_mode = True, 热力图模式, add_map_color = 13
情况 6：target_map_mode = True，target_group_mode = True，我设置为assert，直接报错；
'''

def visual_recon_wsi_folder(
        files_read_path = None, files_save_path = None, show_mode=True, add_map_color=0,
        target_map_mode=False, target_map_color=1, target_class_arr=None,  # 组织区域是颜色图的模式
        target_group_mode=False, target_group_wegt_arr=None, group_color=[1, 2, 3], # 组织区域是不同类别颜色
        grad_mode=False, grad_color=[255, 255, 0], grad_class_arr=None, non_class_color=6,  # 任务相关组织是渐变颜色区域
        heat_map_mode=False, heat_wegt_arr=None, heat_class_arr=None, map_color_str='HOT',cam_save_path=None,  # 热力图模式
        grid_stat=True, grid_size=(224, 224), grid_color=(255, 255, 255), grid_thick=10,  #画网格模式
        re_list_mode = '2', count_wegt = None, img_size = (224, 224, 3)
):
    '''
    :param files_read_path:
    :param files_save_path:
    :param visual_recon_wsi:
    :return:
    '''
    images_name_list = natsort.natsorted(os.listdir(files_read_path), alg=natsort.ns.PATH)
    print(images_name_list)

    for patch_name_i in images_name_list:
        print(count_wegt)

        if not os.path.exists(files_save_path):
            os.makedirs(files_save_path)

        if not os.path.exists(cam_save_path):
            os.makedirs(cam_save_path)

        read_path = os.path.join(files_read_path, patch_name_i)
        save_path = os.path.join(files_save_path, patch_name_i + '.jpg')
        cam_save =  os.path.join(cam_save_path, patch_name_i + '.jpg')
        visual_recon_wsi(
            read_path=read_path, save_path=save_path, show_mode=show_mode, add_map_color=add_map_color,
            target_map_mode=target_map_mode, target_map_color=target_map_color, img_size=img_size,
                            target_class=target_class_arr[count_wegt],  # 组织区域是颜色图的模式
            target_group_mode=target_group_mode, target_group_wegt=target_group_wegt_arr[count_wegt],
                              group_color=group_color, # 组织区域是不同类别颜色
            grad_mode= grad_mode, grad_color=grad_color, grad_class=grad_class_arr[count_wegt],
                          non_class_color=non_class_color,  # 任务相关组织是渐变颜色区域
            heat_map_mode=heat_map_mode, heat_wegt=heat_wegt_arr[count_wegt], heat_class=heat_class_arr[count_wegt],
                           map_color_str=map_color_str, cam_save_path=cam_save,  # 热力图模式
            grid_stat=grid_stat, grid_size=grid_size, grid_color=grid_color, grid_thick=grid_thick,  #画网格模式,
            re_list_mode = re_list_mode
        )
        count_wegt += 1




###
def get_all_path(files_read_path=None, father_name_order=[1, 0]):
    father_path_name = natsort.natsorted(os.listdir(files_read_path), alg=natsort.ns.PATH)
    father_name_order = [1, 0]
    father_path_new_name = [father_path_name[father_name_order[i]] for i in range(len(father_path_name))]

    son_path_name_list = []
    for father_i in father_path_new_name:
        new_path = os.path.join(files_read_path, father_i)
        new_path_list = natsort.natsorted(os.listdir(new_path), alg=natsort.ns.PATH)
        for new_j in new_path_list:
            son_path_name_list.append(os.path.join(new_path, new_j))

    print(father_path_new_name)
    print(son_path_name_list)
    return son_path_name_list





###
def get_inputs(dists_path = None, labels_path = None,
               relats_path = None, selec_order = None,
               model_mode = 'ours'):
    import joblib
    import numpy as np

    if model_mode == 'ours':
        ########## distances
        dists_list = joblib.load(dists_path)
        dists_selec_list = dists_list[selec_order['start']:selec_order['end']]
        for dists_i in range(len(dists_selec_list)):
            dists_selec_list[dists_i] = 1 - (np.array(dists_selec_list[dists_i]) / max(dists_selec_list[dists_i]))

        print(len(dists_selec_list))
        print(len(dists_selec_list[0]))

        ########## relations
        relats_list = joblib.load(relats_path)
        relats_new_list = []
        for relats_i in relats_list:
            relats_new_list.append(np.int8(relats_i[0]))

        relats_selec_list = relats_new_list[selec_order['start']:selec_order['end']]
        print(len(relats_selec_list))
        print(len(relats_selec_list[0]))

        ########## labels
        labes_list = joblib.load(labels_path)
        labes_selec_list =labes_list[selec_order['start']:selec_order['end']]
        for label_j in range(len(labes_selec_list)):
            labes_selec_list[label_j] = int(labes_selec_list[label_j])

        print(len(labes_selec_list))
        print(labes_selec_list[0])

    elif model_mode == 'others':
        ########## distances
        dists_list = joblib.load(dists_path)
        dists_selec_list = dists_list[selec_order['start']:selec_order['end']]
        for dists_i in range(len(dists_selec_list)):
            dists_selec_list[dists_i] = np.abs(dists_selec_list[dists_i]) / np.max(np.abs(dists_selec_list[dists_i]))

        print(len(dists_selec_list))
        print(len(dists_selec_list[0]))

        ########## relations
        relats_selec_list = []
        for relats_j in dists_selec_list:
            inter_list = []
            for relats_i in relats_j:
                if relats_i > 0.5:
                    inter_list.append(0)
                else:
                    inter_list.append(1)
            relats_selec_list.append(inter_list)

        ######### labels
        labes_selec_list = [0 for i in range(selec_order['end'] - selec_order['start'])]

    else:
        assert print('Errors!!!')
    return relats_selec_list, dists_selec_list, labes_selec_list




def rename_amu(files_read_path=None, father_name_order=[1, 0]):
    import os
    import natsort
    father_path_name = natsort.natsorted(os.listdir(files_read_path), alg=natsort.ns.PATH)
    father_name_order = [1, 0]
    father_path_new_name = [father_path_name[father_name_order[i]] for i in range(len(father_path_name))]

    son_path_name_list = []
    for father_i in father_path_new_name:
        new_path = os.path.join(files_read_path, father_i)
        new_path_list = natsort.natsorted(os.listdir(new_path), alg=natsort.ns.PATH)
        for new_j in new_path_list:
            for all_i in natsort.natsorted(os.listdir((os.path.join(new_path, new_j))), alg=natsort.ns.PATH):
                son_path_name_list.append(os.path.join(new_path, new_j, all_i))

    for folder_i in son_path_name_list:
        file_name_list = natsort.natsorted(os.listdir(folder_i), alg=natsort.ns.PATH)
        for count_i in range(len(file_name_list)):
            h = count_i // 31
            w = count_i % 31
            new_name = '0_' + str(h) + '_' + str(w) + '_' + '31x31.jpg'
            print(new_name)
            os.rename(os.path.join(folder_i, file_name_list[count_i]), os.path.join(folder_i, new_name))
            from skimage import io
            img = io.imread(os.path.join(folder_i, new_name))
            if np.mean(img) == 0:
                os.remove(os.path.join(folder_i, new_name))
                new_img = np.zeros((96, 96, 3)) + 255.0
                print(new_img.shape)
                io.imsave(os.path.join(folder_i, new_name), np.uint8(new_img))

    print(father_path_new_name)
    print(son_path_name_list)

    return son_path_name_list




def find_all_folder(files_read_path = None):
    import os
    import natsort
    father_path_name = natsort.natsorted(os.listdir(files_read_path), alg=natsort.ns.PATH)
    father_name_order = [1, 0]
    father_path_new_name = [father_path_name[father_name_order[i]] for i in range(len(father_path_name))]

    son_path_name_list = []
    for father_i in father_path_new_name:
        new_path = os.path.join(files_read_path, father_i)
        new_path_list = natsort.natsorted(os.listdir(new_path), alg=natsort.ns.PATH)
        for new_j in new_path_list:
            for all_i in natsort.natsorted(os.listdir((os.path.join(new_path, new_j))), alg=natsort.ns.PATH):
                #print(os.path.join(new_path[-4:], new_j, all_i))
                son_path_name_list.append(os.path.join(new_path[-4:], new_j, all_i))
    return son_path_name_list




def check_dataset_complete(path_1 = None, path_2 = None):
    path_1_list = find_all_folder(path_1)
    path_2_list = find_all_folder(path_2)
    print(len(path_1_list), len(path_2_list))

    if len(path_1_list) > len(path_2_list):
        for check_i in path_1_list:
            if check_i not in path_2_list:
                print(check_i, 'path_1 is big')
    else:
        for check_i in path_2_list:
            if check_i not in path_1_list:
                print(check_i, 'path_2 is big')



if __name__ == '__main__':

    ### File model
    '''
    visual_recon_wsi函数的使用方法：
    情况 1：target_map_mode = False，target_group_mode = False，add_map_color = 0，恢复WSI原图模样；
    情况 2：target_map_mode = False，target_group_mode = False，add_map_color = 非0，WSI组织区域是原图，背景区域是各种颜色图；
    情况 3：target_map_mode = True，target_group_mode = False，add_map_color = 非0，target_map_color = 非0，WSI组织和背景是具有对比的颜色图；
    情况 4：target_map_mode = False，target_group_mode = True，target_group_wegt = nx1数组(n是组织实例数量)，add_map_color = 非0，
           WSI的组织按target_group_wegt和group_color数组值去上色，背景是add_map_color的颜色；
    情况 5：target_map_mode = True，target_group_mode = True，我设置为assert，直接报错；
    map_color_str: 'HOT', 'JET', 'INFERNO', 'MAGMA', 'TURBO', 'MY_COLOR'
    '''

    '''
    read_path = r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Datasets/DHMC_Kidney/Kidney_WSI_without_PE/Train/Benign/DHMC_0003'
    save_path = r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/2.jpg'
    cam_save_path = r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/2_Heat_HOT_CAM.jpg'
    #group_label_path = r'C:\\Users\MrHuang\Pictures\Motivation_Materials_GGO_ISDC\group_label.json'
    #with open(group_label_path, 'r', encoding='utf-8') as f:
    #    target_group_wegt = json.load(f)
    target_group_wegt = [0 for i in range(100)] + [1 for i in range(255)] + [2 for i in range(100)]
    heat_wegt = np.random.rand(455)
    # 0: 纯白， 13：纯黑
    visual_recon_wsi(read_path=read_path, save_path=save_path, show_mode=True, add_map_color=0,
                     target_map_mode=False, target_map_color=1, target_class=1,  # 组织区域是颜色图的模式
                     target_group_mode=False, target_group_wegt=target_group_wegt, group_color=[1, 2, 3],  # 组织区域是不同类别颜色
                     grad_mode=False, grad_color=[255, 255, 0], grad_class=1, non_class_color = 6,  #任务相关组织是渐变颜色区域
                     heat_map_mode=False, heat_wegt=heat_wegt, heat_class=1, map_color_str='HOT', cam_save_path=cam_save_path, #热力图模式
                     grid_stat=True, grid_size=(224*8, 224*8), grid_color=(0, 0, 0), grid_thick=20, re_list_mode='2')  #画线模式
    '''

    ### Folder mode

    # get inputs, SOTAs input feat_1D, GGO_ISDC input relation
    dists_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Relations/Prostate/GGO_ISDC/distances.joblib'
    labels_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Relations/Prostate/GGO_ISDC/labels.joblib'
    relats_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Relations/Prostate/GGO_ISDC/relations.joblib'
    #selec_order = {'start': 395, 'end': 422} # DHMC-kidney
    #selec_order = {'start': 204, 'end': 240} # AMU-LSCC
    #selec_order = {'start': 156, 'end': 207} # AMU-CSCC
    #selec_order = {'start': 257, 'end': 269}  # CAMELYON
    selec_order = {'start': 159, 'end': 179}  # Prost
    


    ###########################################################
    relats_selec_list, dists_selec_list, labes_selec_list =\
            get_inputs(dists_path = dists_path, labels_path = labels_path,
                       relats_path = relats_path, selec_order = selec_order,
                       model_mode='ours')  ### IGI-PAEC: ours, SOTAs: others
    ###########################################################


    # get files name
    files_read_path = r'/root/autodl-tmp/GGO_ISDC_public/Datasets/Prostate/Prostate_WSI_without_PE'
    files_save_path = r'/root/autodl-tmp/GGO_ISDC_public/Results/Visual/Prostate/GGO_ISDC/WSIs'
    files_cam_save = r'/root/autodl-tmp/GGO_ISDC_public/Results/Visual/Prostate/GGO_ISDC/CAM_map'


    ###########################################################
    folder_read_list = get_all_path(files_read_path=files_read_path, father_name_order=[1, 0])

    ### folder_num: kidney: 8个, LSCC: 6个, Prost: 6个, Camelyon: 4个
    folder_no = 5
    folder_read_list = [folder_read_list[folder_no]]
    ###########################################################

    target_group_wegt_arr = relats_selec_list
    heat_wegt_arr = dists_selec_list
    print(len(target_group_wegt_arr), len(heat_wegt_arr))

    ####SOTAs
    '''
    target_class_arr = labes_selec_list
    grad_class_arr = labes_selec_list
    heat_class_arr = labes_selec_list
    '''

    ####GGO_ISDC

    target_class_arr = 1 - np.array(labes_selec_list)
    grad_class_arr = 1- np.array(labes_selec_list)
    heat_class_arr = 1- np.array(labes_selec_list)


    # running visualizations
    count_wegt = 0
    for folder_read_i in folder_read_list:
        visual_recon_wsi_folder(
            files_read_path=folder_read_i, files_save_path=files_save_path, show_mode=False, add_map_color=0,
            target_map_mode=False, target_map_color=1, target_class_arr=target_class_arr,  # 组织区域是颜色图的模式
            target_group_mode=False, target_group_wegt_arr=target_group_wegt_arr, group_color=[1, 2, 3],  # 组织区域是不同类别颜色
            grad_mode=False, grad_color=[255, 255, 0], grad_class_arr=grad_class_arr, non_class_color=6,  # 任务相关组织是渐变颜色区域
            heat_map_mode=False, heat_wegt_arr=heat_wegt_arr, heat_class_arr=heat_class_arr, map_color_str='HOT',
                           cam_save_path=files_cam_save,  # 热力图模式
            grid_stat=False, grid_size=(224, 224), grid_color=(255, 255, 255), grid_thick=10,  #画线模式
            re_list_mode='2', count_wegt = count_wegt, img_size=(224, 224, 3)
        )

