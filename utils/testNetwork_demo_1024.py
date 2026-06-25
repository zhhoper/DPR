import sys
sys.path.append('model')
sys.path.append('utils')

from utils_SH import *

# other modules
import os
import numpy as np
from defineHourglass_1024_gray_skip_matchFeature import *
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2

__IMG_SIZE = 256

def test_network_demo_1024(**kwargs) -> None:
    # --------------------- configs ---------------------
    modelFolder = kwargs.get("modelFolder", 'trained_model/')
    lightFolder = kwargs.get("lightFolder", 'data/example_light/')
    saveFolder = kwargs.get("saveFolder", 'result_1024')
    img = kwargs.get("img", 'data/obama.jpg')
    net = None

    # ---------------- create normal for rendering half sphere ------
    x = np.linspace(-1, 1, __IMG_SIZE)
    z = np.linspace(1, -1, __IMG_SIZE)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <=1
    y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))
    #-----------------------------------------------------------------



    # load model
    try:
        net = HourglassNet(16)
        net = HourglassNet_1024(net, 16)
        net.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_1024_03.t7')))
        net.cuda()
        net.train(False)
    except FileExistsError as e:
        print(f"File error '{e}'")

    except RuntimeError as e:
        print(f"RunTimeError as '{e}'")

    except Exception as e:
        print(f"Error '{e}'")

    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)


    img = cv2.imread(img)
    row, col, _ = img.shape
    img = cv2.resize(img, (1024, 1024))
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    inputL = Lab[:,:,0]
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    inputL = Variable(torch.from_numpy(inputL).cuda())

    for file in os.listdir(lightFolder):
        if os.path.isdir(os.path.join(lightFolder, file)):
            continue
        sh = np.loadtxt(os.path.join(lightFolder, file))
        sh = sh[0:9]
        sh = sh * 0.7

        # rendering half-sphere
        sh = np.squeeze(sh)
        shading = get_shading(normal, sh)
        value = np.percentile(shading, 95)
        ind = shading > value
        shading[ind] = value
        shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
        shading = (shading *255.0).astype(np.uint8)
        shading = np.reshape(shading, (256, 256))
        shading = shading * valid
        cv2.imwrite(os.path.join(saveFolder, file.replace(".txt", '.jpg')), shading)


        #----------------------------------------------
        #  rendering images using the network
        #----------------------------------------------
        sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
        sh = Variable(torch.from_numpy(sh).cuda())
        outputImg, _, outputSH, _  = net(inputL, sh, 0)
        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1,2,0))
        outputImg = np.squeeze(outputImg)
        outputImg = (outputImg*255.0).astype(np.uint8)
        Lab[:,:,0] = outputImg
        resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        resultLab = cv2.resize(resultLab, (col, row))
        cv2.imwrite(os.path.join(saveFolder, file.replace(".txt", '.jpg')), resultLab)
