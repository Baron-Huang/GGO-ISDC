import subprocess
import os

# 列表中包含你想要顺序执行的脚本名
read_path = (r'/root/autodl-tmp/GGO_ISDC_public/Codes/Main_func_SOTAs')
model_names = ['AB_MIL_Gated', 'AB_MIL_Linear',
               'CLAM_SB_B', 'CLAM_SB_S', 'CLAM_MB_B', 'CLAM_MB_S',
               'DGR_MIL', 'DTFD_MIL', 'FRMIL', 'HAG_MIL',
               'ILRA_MIL', 'RRTMIL', 'S4MIL', 'TransMIL']

scripts = []
#scripts.append(r'/home/dataset-hpfs-0/Kevin_Huang/IGI_PAEC_public/Codes/IGI_PAEC_main_DHMC_Kidney.py')
for name_i in model_names:
    scripts.append(os.path.join(read_path, name_i + '.py'))


# 遍历列表，依次运行每个脚本
for script in scripts:
    subprocess.run(['python', script])

# 并行运行
'''
processes = []
for script in scripts:
    p = subprocess.Popen(['python', script])
    processes.append(p)


for p in processes:
    p.wait()
'''

