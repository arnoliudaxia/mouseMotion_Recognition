# ffmpeg -i 呕吐合集.avi %d.png
# ffmpeg -i "/mnt/Data-Mouse/Motion_Recognition/vomit_test/For training/正常动作/白色背景合集/正常动作白色背景视角.mp4" %d.png


from pathlib import Path
import os
import shutil
import numpy as np


def split_data_into_train_val_test(src_folder, train_folder, val_folder, test_folder, train_ratio=0.85, val_ratio=0.05,
                                   test_ratio=0.10, vomitNum=None):

    subfolders = [subfolder for subfolder in src_folder.iterdir() if subfolder.is_dir()]

    # 创建训练、验证、测试的文件夹路径
    train_folders = [train_folder / subfolder.name for subfolder in subfolders]
    val_folders = [val_folder / subfolder.name for subfolder in subfolders]
    test_folders = [test_folder / subfolder.name for subfolder in subfolders]
    # 创建相应的文件夹
    for folder in train_folders + val_folders + test_folders:
        folder.mkdir(parents=True, exist_ok=True)

    for src_folder_sub, train_folder, val_folder, test_folder in zip(subfolders,train_folders,val_folders,test_folders):
        # 获取源文件夹中所有的图片
        images = [img for img in src_folder_sub.iterdir() if img.is_file()]

        # 如果是 'Normal' 文件夹，我们只挑选一部分的图片
        if "Normal" in src_folder_sub.name and len(images)>vomitNum:
            num_selected_images = vomitNum
            images = np.random.choice(images, num_selected_images, replace=False)


        # 随机打乱图片
        np.random.shuffle(images)

        # 计算训练、验证和测试图片的个数
        num_images = len(images)
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)
        num_test = num_images - num_train - num_val

        print(f"类别: {src_folder_sub.name}")
        print(f"num_images: {num_images}")
        print(f"num_train: {num_train}")
        print(f"num_val: {num_val}")
        print(f"num_test: {num_test}")

        # 划分训练、验证和测试图片
        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]

        # 复制图片到相应的文件夹
        for img in train_images:
            shutil.copy(img, train_folder)
        for img in val_images:
            shutil.copy(img, val_folder)
        for img in test_images:
            shutil.copy(img, test_folder)


# 源文件夹路径
src_folder = Path("datasets/source/vomit_v7_stepbystep")

# 训练、验证和测试文件夹的路径
targetPathParent=src_folder.parent.parent / src_folder.name
train_folder = targetPathParent / "train"
val_folder = targetPathParent / "val" 
test_folder =  targetPathParent / "test"

print(f"源文件在 {src_folder}")
print(f"Train,Validation,Test分区在 {train_folder},{val_folder},{test_folder}")

# 划分数据集
split_data_into_train_val_test(src_folder, train_folder, val_folder, test_folder,val_ratio=0.1,vomitNum=8000)