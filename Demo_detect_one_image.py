# -*- coding: utf-8 -*-
# @Author  : LG
from Model import SSD
from Configs import _C as cfg
from PIL import Image
from matplotlib import pyplot as plt

# 实例化模型
net = SSD(cfg)
# 使用cpu或gpu
net.to('cuda')
# 模型从权重文件中加载权重
net.load_pretrained_weight('Weights/trained/model_4000.pkl')
# 打开图片
image = Image.open(r'..\貨櫃資料集\驗證集\image_0001.jpg')
# 进行检测, 分别返回 绘制了检测框的图片数据/回归框/标签/分数.
drawn_image, boxes, labels, scores = net.Detect_single_img(image=image,score_threshold=0.2)

plt.imsave('Images/det.jpg',drawn_image)
plt.imshow(drawn_image)
plt.show()