"""
import os
import json

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def read_csv_classes(csv_dir: str, csv_name: str):
    data = pd.read_csv(os.path.join(csv_dir, csv_name))
    # print(data.head(1))  # filename, label

    label_set = set(data["label"].drop_duplicates().values)

    print("{} have {} images and {} classes.".format(csv_name,
                                                     data.shape[0],
                                                     len(label_set)))
    return data, label_set


def calculate_split_info(path: str, label_dict: dict, rate: float = 0.2):
    # read all images
    image_dir = os.path.join(path, "images")
    images_list = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
    print("find {} images in dataset.".format(len(images_list)))

    train_data, train_label = read_csv_classes(path, "train.csv")
    val_data, val_label = read_csv_classes(path, "val.csv")
    test_data, test_label = read_csv_classes(path, "test.csv")

    # Union operation
    labels = (train_label | val_label | test_label)
    labels = list(labels)
    labels.sort()
    print("all classes: {}".format(len(labels)))

    # create classes_name.json
    classes_label = dict([(label, [index, label_dict[label]]) for index, label in enumerate(labels)])
    json_str = json.dumps(classes_label, indent=4)
    with open('classes_name.json', 'w') as json_file:
        json_file.write(json_str)

    # concat csv data
    data = pd.concat([train_data, val_data, test_data], axis=0)
    print("total data shape: {}".format(data.shape))

    # split data on every classes
    num_every_classes = []
    split_train_data = []
    split_val_data = []
    for label in labels:
        class_data = data[data["label"] == label]
        num_every_classes.append(class_data.shape[0])

        # shuffle
        shuffle_data = class_data.sample(frac=1, random_state=1)
        num_train_sample = int(class_data.shape[0] * (1 - rate))
        split_train_data.append(shuffle_data[:num_train_sample])
        split_val_data.append(shuffle_data[num_train_sample:])

        # imshow
        imshow_flag = False
        if imshow_flag:
            img_name, img_label = shuffle_data.iloc[0].values
            img = Image.open(os.path.join(image_dir, img_name))
            plt.imshow(img)
            plt.title("class: " + classes_label[img_label][1])
            plt.show()

    # plot classes distribution
    plot_flag = False
    if plot_flag:
        plt.bar(range(1, 101), num_every_classes, align='center')
        plt.show()

    # concatenate data
    new_train_data = pd.concat(split_train_data, axis=0)
    new_val_data = pd.concat(split_val_data, axis=0)

    # save new csv data
    new_train_data.to_csv(os.path.join(path, "new_train.csv"))
    new_val_data.to_csv(os.path.join(path, "new_val.csv"))


def main():
    data_dir = "F:/BaiduNetdiskDownload/mini-imagenet"  # 指向数据集的根目录
    json_path = "F:/BaiduNetdiskDownload/imagenet_class_index.json"  # 指向imagenet的索引标签文件

    # load imagenet labels
    label_dict = json.load(open(json_path, "r", encoding='utf-8'))
    label_dict = dict([(v[0], v[1]) for k, v in label_dict.items()])

    calculate_split_info(data_dir, label_dict)


if __name__ == '__main__':
    main()
"""

import pandas as pd
import os

import shutil
"""
#读取CSV文件，获取所有的img文件名称
test_csv = "F:/BaiduNetdiskDownload/mini-imagenet/new_val.csv"
data=pd.read_csv(test_csv)
#print(data,type(data))

test_filename = list(data["filename"].values)
#print(test_filename)
#print(len(test_filename))

dst = "F:/BaiduNetdiskDownload/mini-imagenet/val"  #提前创建一个新的train文件夹，将CSV对应的train img 复制到文件夹中
for i,name in enumerate(test_filename):
    imgx = os.path.join("F:/BaiduNetdiskDownload/mini-imagenet/images",name)
    #print(f"第{i}张图片已经copy完成")
    print(imgx)
    shutil.copy(imgx,dst)

import pandas as pd
import os

import shutil
"""
files = os.listdir("F:/BaiduNetdiskDownload/mini-imagenet/val/") #上一步创建的文件夹
pre = "F:/BaiduNetdiskDownload/mini-imagenet/val/"

for i,img in enumerate(files):

    #1. 首先遍历每个文件，创建文件夹
    #n0153282900000138.jpg
    dir_name = img.split(".")[0][:9]  #这里就是为了截取label，根据img name 前9个为label
    dir_path = pre+dir_name
    #print(dir_path)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path) #创建该类文件夹

    #直接判断该文件，归类
    img_path=pre+img
    if not os.path.isdir(img_path):
        if img[:9]==dir_name:   #由于每个类包含很多img文件，判断该文件是否属于该类
            shutil.move(img_path,dir_path) #true的话，移动到该类目录
