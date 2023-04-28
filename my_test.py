import torch
import torchvision
from PIL import Image
from torch import nn
from models.fasternet import *

#image_path = "data/imagenet/val/dog/dog6.jpeg"
image_path = "data/imagenet/airplane.png"
image = Image.open(image_path) #注意，PNG格式的图片是4个通道，还有一个透明度通道，所以我们要调用image=image.convert('RGB'),保留颜色通道。
print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

tudui = FasterNet()

model = torch.load("tudui_CIFAR10_0.pth", map_location=torch.device('cpu')) # 下载的模型是用gpu训练的，下载过来的时候要说明一下，
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

#写一下类别

classes = {0:'airplane', 1:'automobile ', 2:'bird ', 3:' cat', 4:' deer', 5:'dog ', 6:'frog ', 7:'horse ', 8:'ship ', 9:' truck'}

print("this is a {}".format(classes[output.argmax(1).item()]))
#print(output.argmax(1)) # 输出是6，但是6表示的是青蛙，我输入的是狗 ，在学校的服务器上，训练了6轮才识别对是狗狗。


