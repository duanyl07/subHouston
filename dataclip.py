import h5py
import numpy as np
import json
import os

# ============ 0. 路径 ============ #
target_h, target_w =  601,-601
save_root      = r'D:\code\data\data\HoustonU\crop_'+str(target_h)+'_'+str(target_w)
os.makedirs(save_root, exist_ok=True)
save_img_path  = os.path.join(save_root, 'HoustonU_'+str(target_h)+'_'+str(target_w)+'.mat')
save_gt_path   = os.path.join(save_root, 'HoustonU_gt_'+str(target_h)+'_'+str(target_w)+'.mat')
save_map_path  = os.path.join(save_root, 'label_mapping.json')
# ============ 1. 读数据 ============ #
file_path = 'D:\code\data\data\HoustonU\HoustonU.mat'
file_path_gt = 'D:\code\data\data\HoustonU\HoustonU_gt.mat'
with h5py.File(file_path, 'r') as mat_file:
    data = mat_file['houstonU'][:]
    data = data.T
with h5py.File(file_path_gt, 'r') as mat_file1:
    gt = mat_file1['houstonU_gt'][:]
    gt = gt.T


# ============ 2. 裁剪 ============ #

img_crop = data[:target_h, target_w:, 0:48]    # y × x × C
gt_crop  = gt [:target_h, target_w:]       # y × x
# img_crop = data[:, :, :]    # y × x × C
# gt_crop  = gt [:, :]   
# ============ 3. 重新排序标签 ============ #
# 3.1 统计裁剪后实际出现的标签
labels, counts = np.unique(gt_crop, return_counts=True)

# 3.2 建立“原标签 → 新标签”的映射
#      0 背景/未定义 保持为 0，其余按升序重新编号 1,2,...
label_map = {}
next_id = 1
for lab in labels:
    if lab == 0:        # 背景不变
        label_map[int(lab)] = 0
    else:
        label_map[int(lab)] = next_id
        next_id += 1

# 3.3 按映射表替换 GT
gt_remap = np.vectorize(label_map.get)(gt_crop)

# ============ 4. 保存结果 ============ #
# # 影像
# with h5py.File(save_img_path, 'w') as f_out:
#     f_out.create_dataset('HoustonU', data=img_crop, compression='gzip')

# # GT
# with h5py.File(save_gt_path, 'w') as f_out:
#     f_out.create_dataset('HoustonU_gt', data=gt_remap, compression='gzip')

# 映射表
with open(save_map_path, 'w', encoding='utf-8') as f_map:
    json.dump(label_map, f_map, indent=4, ensure_ascii=False)

print('裁剪 + 重映射完成！')
print('标签映射表：', label_map)
print('标签映射表：',  counts)
# 简单可视化核查
import matplotlib.pyplot as plt
plt.imshow(gt_remap, cmap='tab20')
plt.title('Remapped GT')
plt.axis('off')
plt.show()
