import os

from alive_progress import alive_bar
from torch.autograd import Variable
import torch
import cv2

from configs.load_configs import configs
from utils.utils_SH import *

if configs["mode"] == "mode_512":
    from model.defineHourglass_512_gray_skip import *
else:
    from model.defineHourglass_1024_gray_skip_matchFeature import *


def video(predictor, n_lighting, fps=5, name=os.path.basename(configs['path_image']).split('.')[0]):
    print(f'Processing image {name} ------ \nPlease Uong mieng nuoc & an mieng banh de...')
    image = cv2.imread(configs["path_image"])
    frame_width = image.shape[1] * 2
    frame_height = image.shape[0]
    size = (frame_width, frame_height)
    os.makedirs('results/', exist_ok=True)
    saved_path = f'results/{name}.mp4'
    out = cv2.VideoWriter(saved_path,
                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                          fps, size)

    with alive_bar(total=n_lighting, theme='musical', length=200) as bar:
        for i in range(n_lighting):
            try:
                frame = predictor(i)
                out.write(frame)
                bar()
            except KeyboardInterrupt:
                print("Stoped!")
                out.release()
                break
    out.release()
    print(f"Video saved in: {saved_path}")


class DPR:
    def __init__(self, img=configs["path_image"], device=None):
        self.mode = configs["mode"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is not None else device
        self.img = cv2.imread(img) if type(img) is str else img
        self.img_ori = self.img.copy()
        self.row, self.col = self.img.shape[:2]
        self.img = cv2.resize(self.img, tuple((int(configs[self.mode]['size']), int(configs[self.mode]['size']))))
        self.Lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        self.inputL = self._process_inputL()

        # load model
        if self.mode == 'mode_512':
            self.my_network = HourglassNet()

        elif self.mode == 'mode_1024':
            my_network_512 = HourglassNet(16)
            self.my_network = HourglassNet_1024(my_network_512, 16)

        self.my_network.load_state_dict(
            torch.load(os.path.join(configs["modelFolder"], configs[self.mode]["checkpoint"])))
        self.my_network.to(self.device)
        self.my_network.train(False)

    def _process_inputL(self):
        inputL = self.Lab[:, :, 0]
        inputL = inputL.astype(np.float32) / 255.0
        inputL = inputL.transpose((0, 1))
        inputL = inputL[None, None, ...]
        inputL = Variable(torch.from_numpy(inputL).to(self.device))
        return inputL

    def _process_output(self, outputImg):
        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1, 2, 0))
        outputImg = np.squeeze(outputImg)
        outputImg = (outputImg * 255.0).astype(np.uint8)
        return outputImg

    def _create_sh(self, i):
        # rendering half-sphere
        sh = np.loadtxt(os.path.join(configs["lightFolder"], 'rotate_light_{:02d}.txt'.format(i)))
        sh = sh[0:9]
        sh = sh * 0.7
        sh = np.squeeze(sh)
        return sh

    def _create_normal_and_valid(self, img_size=256):
        # ---------------- create normal for rendering half sphere ------
        img_size = 256
        x = np.linspace(-1, 1, img_size)
        z = np.linspace(1, -1, img_size)
        x, z = np.meshgrid(x, z)

        mag = np.sqrt(x ** 2 + z ** 2)
        valid = mag <= 1
        y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
        x = x * valid
        y = y * valid
        z = z * valid
        normal = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
        normal = np.reshape(normal, (-1, 3))
        return normal, valid

    def _get_shading(self, normal, SH):
        """
            get shading based on normals and SH
            normal is Nx3 matrix
            SH: 9 x m vector
            return Nxm vector, where m is the number of returned images
        """
        sh_basis = SH_basis(normal)
        shading = np.matmul(sh_basis, SH)
        return shading

    def __call__(self, i):
        # for i in range(n_lighting):
        sh = self._create_sh(i)
        normal, valid = self._create_normal_and_valid()
        shading = self._get_shading(normal, sh)
        value = np.percentile(shading, 10)
        ind = shading > value
        shading[ind] = value
        shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
        shading = (shading * 255.0).astype(np.uint8)
        shading = np.reshape(shading, (256, 256))
        shading = shading * valid

        # Save visualize SH
        if configs["saveVisualize"]:
            cv2.imwrite(os.path.join(configs[self.mode]["saveFolder"], 'light_{:02d}.png'.format(i)), shading)

        #  rendering images using the network
        sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
        sh = Variable(torch.from_numpy(sh).to(self.device))

        if self.mode == "mode_1024":
            outputImg, _, outputSH, _ = self.my_network(self.inputL, sh, 0)
        else:
            outputImg, outputSH = self.my_network(self.inputL, sh, 0)

        outputImg = self._process_output(outputImg)
        self.Lab[:, :, 0] = outputImg
        resultLab = cv2.cvtColor(self.Lab, cv2.COLOR_LAB2BGR)
        resultLab = cv2.resize(resultLab, (self.col, self.row))

        final = cv2.hconcat([self.img_ori, resultLab])

        if configs["showResult"]:
            cv2.imshow("shading", shading)
            cv2.imshow(configs[self.mode]["windowName"], final)
            cv2.waitKey(0)

        if configs["saveResult"]:
            path_saveFolder = os.path.join(configs[self.mode]["saveFolder"],
                                           os.path.basename(configs["path_image"])[:-4])
            os.makedirs(path_saveFolder, exist_ok=True)

            cv2.imwrite(os.path.join(path_saveFolder, str(i) + ".jpg"), resultLab)
            cv2.imwrite(os.path.join(path_saveFolder, "hconcat_" + str(i) + ".jpg"), final)

        return final


if __name__ == "__main__":
    dpr = DPR()
    n_lighting = len(os.listdir(configs["lightFolder"]))
    video(predictor=dpr, fps=5, n_lighting=n_lighting, name='face1_light3')
