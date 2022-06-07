import argparse
import os 
from dataset import ColorHintDataset
from solver import *
import torch.utils.data as data
from utils import PSNR_SSIM
def main(config):
    os.makedirs("./Pretrained", exist_ok=True)
    os.makedirs("./Result", exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"]= config.cuda_idx
    solver = Solver(config)

    if 'train' ==  config.mode:
        train_dataset = ColorHintDataset(config.data_path, config.image_size,"train")
        valid_dataset = ColorHintDataset(config.data_path, config.image_size, "valid")

        train_dataloader = data.DataLoader(train_dataset, batch_size= config.batch_size, 
                                            shuffle=True,num_workers=config.num_workers)
        valid_dataloader = data.DataLoader(valid_dataset, batch_size= 1, 
                                            shuffle=False,num_workers=config.num_workers)
        solver.train(train_dataloader, valid_dataloader)
    elif 'test' == config.mode:
        test_dataset = ColorHintDataset(config.data_path, config.image_size, "test")
        test_dataloader = data.DataLoader(test_dataset, batch_size= 1, 
                                            shuffle=False,num_workers=config.num_workers)
        solver.test(test_dataloader)
    else:
        test_dataset = ColorHintDataset(config.test_data, config.image_size, "coco")
        test_dataloader = data.DataLoader(test_dataset, batch_size= 1, 
                                            shuffle=True,num_workers=config.num_workers)
        solver.test(test_dataloader)
        PSNR_SSIM(os.path.join(config.test_data),config.result_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--test_data',type=str, default='/media/vom/HDD1/test2017')
    parser.add_argument('--result_path',type=str, default="./Result")#/media/vom/HDD1/hi_output // ./Result
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--pretrain', type=str, default="./Pretrained")
    parser.add_argument('--image_size', type=int, default=256)

    parser.add_argument('--epochs', type=int, default= 1000)
    parser.add_argument('--lr', type=float, default=5e-4) #  5e-4
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch_decay', type= int, default= 75)
    parser.add_argument('--early_stop', type=int, default=5)


    parser.add_argument('--cuda_idx', type=str, default="2")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pretrain_model', type=str, default='MSAUnet.pkl') #ATCBAM64

    config = parser.parse_args()
    print(config)
    main(config)
