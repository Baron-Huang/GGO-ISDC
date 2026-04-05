############################# training_testing_functions ##############################
############################# GGO_ISDC Demo main function ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Helath, PolyU, Hong Kong
#### Attempt: training & testing functions for GGO_ISDC models

########################## API Section #########################
import torch
from torch import nn
import time
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import numpy as np
from torch import optim
import pandas as pd
from Models.ViT_models.ViT import VisionTransformer
from Models.ViT_models.ViT_model_modules import ViT_Net
import random
from Models.ViT_models.ViT_model_modules import creating_ViT
from Models.Mixer_models.models.Mixer_model_modules import creating_Mixer
import warnings

warnings.filterwarnings('ignore')


########################## seed_function #########################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


########################## learning functions #########################
def vit_public_lr_schedule(epoch):
    if epoch < 80:
        lr = 1e-4
    elif epoch < 90:
        lr = 1e-5
    else:
        lr = 1e-6
    return lr


########################## learning functions #########################
def growth_w_lr_schedule(epoch):
    if epoch < 25:
        lr = 1e-5
    elif epoch < 50:
        lr = 1e-6
    elif epoch < 75:
        lr = 1e-7
    else:
        lr = 1e-8
    return lr


########################## learning functions #########################
def graph_thre_w_lr_schedule(epoch):
    if epoch < 25:
        lr = 1e-4
    elif epoch < 50:
        lr = 1e-5
    elif epoch < 75:
        lr = 1e-6
    else:
        lr = 1e-7
    return lr


########################## learning functions #########################
def feat_sum_w_lr_schedule(epoch):
    if epoch < 25:
        lr = 1e1
    elif epoch < 50:
        lr = 1e0
    elif epoch < 75:
        lr = 1e-1
    else:
        lr = 1e-2
    return lr


########################## learning functions #########################
def vit_amu_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-5
    elif epoch < 75:
        lr = 5e-6
    else:
        lr = 1e-6
    return lr


def one_hot(org_x=None, pre_dim=3):
    one_x = np.zeros((org_x.shape[0], pre_dim))
    for i in range(org_x.shape[0]):
        one_x[i, int(org_x[i])] = 1
    return one_x


def view_results_GGO_ISDC_parallel(mil_feature=None, mil_head=None, train_loader=None, data_parallel=False,
                          loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85,
                          batch_size=4, bags_len=100, input_len = 1000, num_class = 2,
                          relation_list = []):
    mil_feature.eval()
    mil_head.eval()
    #setup_seed(1)
    train_acc = []
    train_loss = []
    train_label_sum = []
    pre_y_sum = np.zeros((1, num_class))
    label_sum = np.zeros(1)

    for train_img_list, train_label in train_loader:
        train_label = train_label.cuda()
        with torch.no_grad():
            train_pre_y = torch.zeros((1, 768)).cuda()
            train_pre_sum = torch.zeros((1, num_class)).cuda()
            for train_img in train_img_list:
                if train_img.shape[0] <= input_len:
                    train_pre_y = torch.cat((train_pre_y, mil_feature(train_img.cuda(), 0)))
                else:
                    sub_count = int(train_img.shape[0] / input_len)
                    for img_data_i in range(sub_count):
                        train_pre_y = torch.cat((train_pre_y,
                                     mil_feature(train_img[input_len*img_data_i:(input_len*(img_data_i+1)), :, :, :].cuda(), 0)))

                    if sub_count * input_len < train_img.shape[0]:
                        k = mil_feature(train_img[sub_count*input_len:, :, :, :].cuda(), 0)
                        train_pre_y = torch.cat((train_pre_y, k))
                    else:
                        pass

            train_pre_y = train_pre_y[1:]
            _, _, train_pre_proba, pre_list = mil_head(train_pre_y)
            relation_list.append(pre_list)
            #print(train_pre_proba.shape)
            #print(train_pre_sum.shape)
            train_pre_sum = torch.cat((train_pre_sum, train_pre_proba))

            train_pre_sum = train_pre_sum[1:]
            train_loss.append(loss_fn(train_pre_sum, train_label).detach().cpu().numpy())
            if proba_mode == True:
                train_pre_y = torch.softmax(train_pre_sum, dim=1)
                train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                for proba_in in range(train_pre_label_proba.shape[0]):
                    if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < torch.tensor(proba_value).cuda():
                        train_pre_label_proba[proba_in] = torch.tensor(3).cuda()
                train_pre_label = train_pre_label_proba
            elif proba_mode == False:
                train_pre_label = torch.argmax(train_pre_sum, dim=1)
                train_label_sum.append(train_pre_label.detach().cpu().numpy())
            else:
                print('error! Please select probability mode!!!')
                break
        train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                        train_pre_label.detach().cpu().numpy()))

        pre_y_sum = np.concatenate((pre_y_sum, train_pre_sum.detach().cpu().numpy()))
        label_sum = np.concatenate((label_sum, train_label.reshape(train_label.shape[0],).detach().cpu().numpy()))

        train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                        train_pre_label.detach().cpu().numpy()))
    pre_y_sum = pre_y_sum[1:, :]
    label_sum = label_sum[1:]
    return train_acc, train_loss, label_sum, pre_y_sum, relation_list




def testing_GGO_ISDC_parallel(mil_feature=None, mil_head=None, train_loader=None, data_parallel=False,
                           loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85,
                           batch_size=4, bags_len=100, val_loader=None, test_loader=None, input_len = 1000,
                            num_class = 2, roc_save_path = None, train_log_path = None, roc_name = None):
    loss_fn = nn.CrossEntropyLoss()
    relation_list = []
    train_acc, train_loss, _, _, relation_list = view_results_GGO_ISDC_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                        train_loader=train_loader, data_parallel=data_parallel,
                                                        loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=None,
                                                        proba_value=proba_value, batch_size=batch_size,
                                                        bags_len=bags_len, input_len=input_len, num_class=num_class,
                                                        relation_list = relation_list)

    test_acc, test_loss, test_true_label, test_pre_y, relation_list = view_results_GGO_ISDC_parallel(mil_feature=mil_feature,
                                                    mil_head=mil_head, train_loader=test_loader, data_parallel=data_parallel,
                                                    loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=None,
                                                    proba_value=proba_value, batch_size=batch_size,
                                                    bags_len=bags_len, input_len=input_len, num_class=num_class,
                                                    relation_list = relation_list)

    true_y_sum = one_hot(org_x = test_true_label, pre_dim = num_class)
    pre_label_sum = np.argmax(test_pre_y, axis=1)

    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))
    print('train_loss:{:.4}'.format(np.mean(train_loss)),
          ' test_loss:{:.4}'.format(np.mean(test_loss)))

    print('########################## testing set results #########################')
    k = classification_report(test_true_label, pre_label_sum, digits=4)
    print(k)


    print('########################## auc results #########################')
    t = roc_auc_score(np.reshape(true_y_sum, (true_y_sum.shape[0]*true_y_sum.shape[1])),
                    np.reshape(test_pre_y, (test_pre_y.shape[0]*test_pre_y.shape[1])))
    print(t)

    fpr, tpr, _ = roc_curve(np.reshape(true_y_sum, (true_y_sum.shape[0]*true_y_sum.shape[1])),
                    np.reshape(test_pre_y, (test_pre_y.shape[0]*test_pre_y.shape[1])))

    print(fpr.shape, tpr.shape)

    with open(train_log_path, "a", encoding="utf-8") as f:
        f.write(k)
        f.write("\n")
        f.write(str(t))

    import os
    if os.path.exists(roc_save_path):
        pass
    else:
        os.makedirs(roc_save_path)

    write_dict = {'fpr':fpr, 'tpr':tpr}
    roc_pd = pd.DataFrame(write_dict)
    roc_pd.to_csv(os.path.join(roc_save_path, roc_name))

    #relation_bags_pd = pd.DataFrame(relation_bags)
    #relation_bags_pd.to_csv(bag_relations_path)
    return relation_list



########################## single-out-parallel fitting function #########################
#### ddai_net:
#### train_loader:
#### val_loader:
#### test_loader:
#### epoch:
#### gpu_device:
#### train_mode:
def training_GGO_ISDC_parallel(mil_feature=None, mil_head=None, train_loader=None, val_loader=None, test_loader=None,
                            proba_mode=False, lr_fn=None, epoch=100, gpu_device=0, onecycle_mr=1e-2, proba_value=0.85,
                            weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1.pth',
                            batch_size=4, bags_len=100, weight_head_path=None, current_lr=None, input_len=1000,
                            num_class = 2, resu_text_path = None):
    loss_fn = nn.CrossEntropyLoss()
    mil_paras = [{'params': mil_feature.parameters()},
                 {'params': mil_head.parameters()}]


    print('########################## training results #########################')
    if lr_fn == 'onecycle':
        rmp_optim = torch.optim.AdamW(mil_paras, lr=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(rmp_optim, max_lr=onecycle_mr,
                                                  epochs=epoch, steps_per_epoch=len(train_loader))
    bench_acc = 0
    for i in range(epoch):
        print('yes!!!')
        start_time = time.time()
        if lr_fn == 'vit_public':
            rmp_optim = torch.optim.AdamW(mil_paras, lr=vit_public_lr_schedule(i))
        elif lr_fn == 'vit_amu':
            rmp_optim = torch.optim.AdamW(mil_paras, lr=vit_amu_lr_schedule(i))
        elif lr_fn == 'cnn':
            pass
        else:
            print('erorr!!!!')
            return 0

        mil_feature.train()
        mil_head.train()
        for img_data_list, img_label in train_loader:
            #print(img_data_list.shape)
            img_label = img_label.cuda()
            pre_y_sum = torch.zeros((1, 768)).cuda()

            for data_i in range(img_data_list.shape[0]):
                if img_data_list[data_i].shape[0] <= input_len:
                    pre_y = mil_feature(img_data_list[data_i].cuda(), i)
                    pre_y_sum = torch.cat((pre_y_sum, mil_feature(img_data_list[data_i].cuda(), i)))
                    #print(pre_y.shape)
                else:
                    sub_count = int(img_data_list[data_i].shape[0] / input_len)
                    for sub_data_i in range(sub_count):
                        #print(pre_y.shape)
                        sub_y = mil_feature(
                            img_data_list[data_i][input_len*sub_data_i:(input_len*(sub_data_i+1)), :, :, :].cuda(), i)
                        #print(kkk.shape)
                        pre_y_sum = torch.cat((pre_y_sum, sub_y))

                    if sub_count * input_len < img_data_list[data_i].shape[0]:
                        other_sub_y = mil_feature(img_data_list[data_i][sub_count * input_len:, :, :, :].cuda(), i)
                        pre_y_sum = torch.cat((pre_y_sum, other_sub_y))
                    else:
                        pass

            #print(count)
            pre_y_sum = pre_y_sum[1:]
            #print(pre_y_sum.shape)
            pre_y, pre_y_graph, pre_y_new, _ = mil_head(pre_y_sum)

            loss_value = loss_fn(pre_y, img_label) + 0.5 * loss_fn(pre_y_graph, img_label) + loss_fn(pre_y_new, img_label)

            loss_value.backward()

            mil_head.ggtc_block.ggt_block.graph_thre_w.data =\
                (mil_head.ggtc_block.ggt_block.graph_thre_w.data -
                 graph_thre_w_lr_schedule(i) * mil_head.ggtc_block.ggt_block.graph_thre_w.grad)
            mil_head.ggtc_block.ggt_block.growth_w.data =\
                (mil_head.ggtc_block.ggt_block.growth_w.data +
                 growth_w_lr_schedule(i) * mil_head.ggtc_block.ggt_block.growth_w.grad)

            rmp_optim.step()
            rmp_optim.zero_grad()

            #print(img_label)

        # print(ddai_net.w)
        #train_acc, train_loss, _, _ = view_results_for_GGO_ISDC_parallel(mil_feature=mil_feature, mil_head=mil_head,
        #                                                    train_loader=train_loader, loss_fn=loss_fn,
        #                                                    proba_mode=proba_mode, gpu_device=gpu_device,
        #                                                   proba_value=proba_value, batch_size=batch_size,
        #                                                   bags_len=bags_len, num_class=num_class)

        val_acc, val_loss, _, _, _ = view_results_GGO_ISDC_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                        train_loader=val_loader, loss_fn=loss_fn,
                                                        proba_mode=proba_mode, gpu_device=gpu_device,
                                                        proba_value=proba_value, batch_size=batch_size,
                                                        bags_len=bags_len, input_len=input_len, num_class=num_class)

        if np.mean(val_acc) >= bench_acc:
            g = mil_feature.state_dict()
            torch.save(g, weight_path)
            g_1 = mil_head.state_dict()
            torch.save(g_1, weight_head_path)
            bench_acc = np.mean(val_acc)
        else:
            pass


        end_time = time.time()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              #' train_loss:{:.4}'.format(np.mean(train_loss)),
             #' train_acc:{:.4}'.format(np.mean(train_acc)),
              ' val_loss:{:.4}'.format(np.mean(val_loss)),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))
        with open(resu_text_path, "a", encoding="utf-8") as f:
            f.write('epoch ' + str(i + 1) + '    Time:' + str(round(end_time - start_time, 4)) +
                    '    val_loss:' + str(round(np.mean(val_loss), 4)) + '    val_acc:' + str(round(np.mean(val_acc), 4)))
            f.write("\n")

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    test_acc, test_loss, _, _, _ = view_results_GGO_ISDC_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                      train_loader=test_loader, loss_fn=loss_fn,
                                                      proba_mode=proba_mode, gpu_device=gpu_device,
                                                      proba_value=proba_value, batch_size=batch_size,
                                                      bags_len=bags_len, num_class=num_class)

    print('########################## testing results #########################')
    print(#'train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))

    return test_acc




if __name__ == '__main__':
    max_lr = 1e-3
    min_lr = 1e-7
    max_boundary = -np.log10(max_lr)
    min_boundary = -np.log10(min_lr)

    change_log = 0
    for k in range(int(min_boundary - max_boundary) * 10):
        if k % 10 == 0:
            change_log += 1
        lr = (max_lr / (10 ** (change_log - 1))) - (k - (change_log - 1) * 10) * (max_lr / (10 ** change_log))
        print(lr)





