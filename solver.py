import os
import numpy as np
import torch
from tqdm import tqdm
from utils import tensor2im
from network import *
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from loss import MS_SSIM_L1_LOSS

class Solver(object):
    def __init__(self, config):
        self.mode = config.mode
        self.model = None
        self.optimizer = None
        self.criterions =MS_SSIM_L1_LOSS()
        self.lr = config.lr
        self.epochs = config.epochs
        self.epoch_decay = config.epoch_decay
        self.pretrain_model = config.pretrain_model

        self.batch_size = config.batch_size
        self.pretrain =config.pretrain
        self.result_path = config.result_path
        self.early_stop = config.early_stop

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.init_model()

    def init_model(self):
        self.model = MSU_Net()
        #MSU_Net()
        #CAUnet()
        #AttU_Net_CBAM()
        # init_weights(self.model, 'kaiming')
        self.model.to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, self.lr)
        self.criterions.to(self.device)

    def train(self, train_loader, valid_loader):
        lr = self.lr
        min_loss = 99999
        best_psnr , best_ssim = 0,0
        
        
        for epoch in range(self.epochs):
            self.model.train(True)
            train_loss=[]
            valid_loss=[]

            for i, data in enumerate(tqdm(train_loader)):
                l = data["l"].to(self.device)
                ab = data["ab"].to(self.device)
                hint = data["hint"].to(self.device)
                mask = data["mask"].to(self.device)


                cv2.imshow('l',tensor2im(l))
                # np.concatenate(mask,np.zeros(256,256,1))
                cv2.imshow('mask',tensor2im(mask))
                # print(hint.shape)
                # print(tensor2im(mask).dtype)
                print(tensor2im(hint).shape)
                print(np.zeros((256,256,1),np.uint8).shape)

                # print(cv2.merge(tensor2im(hint),np.zeros((256,256,1),np.uint8).shape))
                # print(np.concatenate(tensor2im(hint),np.zeros((256,256,1),dtype=np.uint8),axis=2))
                cv2.imshow('hint',cv2.merge((tensor2im(hint),np.zeros((256,256,1),np.uint8))))
                cv2.waitKey()

                gt_image = torch.cat((l,ab),dim=1)
                hint_image = torch.cat((l,hint,mask),dim=1)

                pred_ab = self.model(hint_image)
                pred_image = tensor2im(torch.cat((l,pred_ab),dim=1))

                loss = self.criterions(torch.cat((l,pred_ab),dim=1),torch.cat((l,ab),dim=1))
                train_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            ###################
            # Validation step #
            ###################
            self.model.train(False)
            self.model.eval()

            psnr, ssim = [],[]
            for i, data in enumerate(tqdm(valid_loader)):
                
                l = data["l"].to(self.device)
                ab = data["ab"].to(self.device)
                hint = data["hint"].to(self.device)
                mask = data["mask"].to(self.device)
                file_name = data["file_name"]

                gt_image = torch.cat((l,ab),dim=1)
                hint_image = torch.cat((l,hint,mask),dim=1)
                
                pred_ab = self.model(hint_image)

                pred_image = cv2.cvtColor(tensor2im(torch.cat((l,pred_ab),dim=1)),cv2.COLOR_LAB2BGR)
                gt_image = cv2.cvtColor(tensor2im(gt_image),cv2.COLOR_LAB2BGR)

                loss = self.criterions(torch.cat((l,pred_ab),dim=1),torch.cat((l,ab),dim=1))
                valid_loss.append(loss.item())

                psnr.append(peak_signal_noise_ratio(gt_image,pred_image))
                ssim.append(structural_similarity(gt_image,pred_image, channel_axis=2))
                cv2.imwrite(os.path.join("./outputs",file_name[0]),pred_image)
            
            t_loss = np.average(train_loss)
            v_loss = np.average(valid_loss)
            print('Epoch [%d/%d], Train Loss: %.8f, Valid Loss: %.8f, LR:%f' % 
									(epoch+1, self.epochs, t_loss, v_loss, lr))

            save_path = os.path.join(self.pretrain,"%s_%d_%f.pkl"% ("MSAUnet",epoch+1,lr))

            if min_loss > v_loss:
                min_loss = v_loss
                torch.save(self.model.state_dict(), save_path)
                if np.average(psnr) > 33 :
                    lr -= (5e-4/70)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr']=lr

            if best_psnr<np.average(psnr):
                best_psnr = np.average(psnr)
                best_ssim = np.average(ssim)
            print("PSNR : %.6f SSIM: %.6f \nB_PSNR : %.6f b_SSIM: %.6f" %(np.average(psnr),np.average(ssim),best_psnr,best_ssim))
    
    def test(self, test_loader):
        model_path = os.path.join(self.pretrain,self.pretrain_model)
        self.model.load_state_dict(torch.load(model_path))

        self.model.train(False)
        self.model.eval()

        for i, data in enumerate(tqdm(test_loader)):
            l = data["l"].to(self.device)
            ab = data["hint"].to(self.device)
            file_name = data["file_name"]#.to(self.device)
            mask = data["mask"].to(self.device)
            hint_image = torch.cat((l,ab,mask),dim=1)

            pred = self.model(hint_image)
            pred =torch.cat((l,pred),dim=1)
            
            output = tensor2im(pred)    
            output = cv2.cvtColor(output,cv2.COLOR_LAB2BGR)
            cv2.imwrite(os.path.join(self.result_path,file_name[0]),output)