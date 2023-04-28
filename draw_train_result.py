from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# help(SummaryWriter.add_scalar)
# help(SummaryWriter)
writer = SummaryWriter("./logs/logs_tensorboard")# writer.add_image()
image_path =  "F:/hymenoptera_data/train/bees_image/1092977343_cb42b38d62.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
#print(img_array.shape)

writer.add_image("test", img_array, 2, dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)  # 参数分别是标题、y轴、x轴

# writer.add_scalar()  # 标量的意思
#writer.add_image() # 参数为标题，图像数据类型，global_step:global step value to record。如果我们的图片不是chw,我们需要指定dataformats=我们的格式，比如hwc
writer.close()  # 生成了事件文件