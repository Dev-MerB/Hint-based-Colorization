import torch.utils.data as data
import numpy as np
from torchvision import transforms
import random
import os
import cv2
from utils import tensor2im

class ColorHintTransform(object):
    def __init__(self, size=256, mode="training"):
        super(ColorHintTransform, self).__init__()
        self.size = size
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])

    def bgr_to_lab(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, ab = lab[:, :, 0], lab[:, :, 1:]
        return l, ab

    def hint_mask(self, bgr, threshold=[0.99,0.993,0.995]):
        h, w, c = bgr.shape
        mask_threshold = random.choice(threshold)
        mask = np.random.random([h, w, 1]) > mask_threshold

        hint = [32,64,96,128,160,192,224]
        for i in hint:
            for j in hint:
                mask[i][j] = True
        return mask

    def img_to_mask(self, mask_img):
        mask = mask_img[:, :, 0, np.newaxis] >= 255
        return mask

    def img_to_edge(self, img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray,-1,ksize=3)
        return laplacian

    def img_to_noisy(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        for i in range(80):
            x=random.randint(0,245)
            y=random.randint(0,245)
            v_a=random.choice([127,255])
            if v_a == 127:
                v_b= 255
            else:
                v_b =127
            a[x:x+40,y:y+40]=v_a
            b[x:x+40,y:y+40]=v_b
        return l,cv2.merge([a,b])


    def __call__(self, img, mask_img=None):
        # threshold = [0.95, 0.97, 0.99 ] # default = 0.95, 0.97, 0.99/ 0.99,0.993,0.995
        if (self.mode == "train") | (self.mode == "valid") | (self.mode =="coco"):
            image = cv2.resize(img, (self.size, self.size))
            if self.mode == "train":
                threshold = [0.99,0.993,0.995]
            else:
                threshold = [0.995]
            mask = self.hint_mask(image, threshold)

            hint_image = image * mask
            l, ab = self.bgr_to_lab(image)
            l_hint, ab_hint = self.bgr_to_lab(hint_image)
            edge = self.img_to_edge(image)

            return self.transform(l), self.transform(ab), self.transform(ab_hint), self.transform(mask), self.transform(edge)

        elif self.mode == "test":
            image = cv2.resize(img, (self.size, self.size))
            mask = self.img_to_mask(mask_img)
            
            hint_image = image * mask

            l, _ = self.bgr_to_lab(image)
            _, ab_hint = self.bgr_to_lab(hint_image)
            edge = self.img_to_edge(image)

            return self.transform(l), self.transform(ab_hint), self.transform(mask),self.transform(edge)
        else:
            return NotImplementedError


class ColorHintDataset(data.Dataset):
    def __init__(self, root_path, size, mode="train"):
        super(ColorHintDataset, self).__init__()

        self.root_path = root_path
        self.size = size
        self.mode = mode
        self.transforms = ColorHintTransform(self.size, self.mode)
        self.examples = None
        self.hint = None
        self.mask = None

        if (self.mode == "train"):
            train_dir = os.path.join(self.root_path, "train")
            self.examples = [os.path.join(self.root_path, "train", dirs) for dirs in os.listdir(train_dir)]
        elif (self.mode == "valid"):
            val_dir = os.path.join(self.root_path, "val")
            self.examples = [os.path.join(self.root_path, "val", dirs) for dirs in os.listdir(val_dir)]
        elif self.mode == "coco":
            test_dir = os.path.join(self.root_path)
            self.examples = [os.path.join(self.root_path, dirs) for dirs in os.listdir(test_dir)]
        elif self.mode == "test":
            hint_dir = os.path.join(self.root_path,"test_dataset", "hint")
            mask_dir = os.path.join(self.root_path,"test_dataset", "mask")
            self.hint = [os.path.join(self.root_path,"test_dataset", "hint", dirs) for dirs in os.listdir(hint_dir)]
            self.mask = [os.path.join(self.root_path,"test_dataset", "mask", dirs) for dirs in os.listdir(mask_dir)]
        else:
            raise NotImplementedError

    def __len__(self):
        if self.mode != "test":
            return len(self.examples)
        else:
            return len(self.hint)

    def __getitem__(self, idx):
        if self.mode == "test":
            hint_file_name = self.hint[idx]
            mask_file_name = self.mask[idx]
            hint_img = cv2.imread(hint_file_name)
            mask_img = cv2.imread(mask_file_name)
            
            input_l, input_hint, mask, edge = self.transforms(hint_img, mask_img)

            # sample = {"l": input_l, "hint": input_hint,"mask":mask,
            #           "file_name": "image_%06d.png" % int(os.path.basename(hint_file_name).split('.')[0])}
            sample = {"l": input_l, "hint": input_hint,"mask":mask,
                      "file_name": "image_%06d.png" % int(os.path.basename(hint_file_name).split('.')[0])}
        elif self.mode =="coco":
            file_name = self.examples[idx]
            img = cv2.imread(file_name)

            l, ab, hint, mask,edge = self.transforms(img)

            # sample = {"l": l,"hint": hint, "mask":mask,"file_name":file_name.split('/')[-1]}
            sample = {"l": l,"hint": hint, "mask":mask,"file_name":file_name.split('/')[-1]}
        else:
            file_name = self.examples[idx]
            img = cv2.imread(file_name)
            l, ab, hint, mask, edge = self.transforms(img)
 
            # sample = {"l": l, "ab": ab, "hint": hint, "mask":mask,"file_name":file_name.split('/')[-1]}
            sample = {"l": l, "ab": ab, "hint": hint, "mask":mask,"file_name":file_name.split('/')[-1]}

        return sample
