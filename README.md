  # Hint-based-Image-Colorization-Using-MSAU-Net   
2022-1, Gachon University, Department of Software, Computer Vision Term Project <br><br><br><br>

## Hint-based Image Colorization
Hint-based colorization reconstructs complete colorful images by propagating the color hint given by users. For the hint-based colorization task, each solution should first convert the provided RGB image to LAB image. The inputs of the model should be an ”L” image (i.e. grayscale image or intensity image) and a color hint map. The output will be a predicted ”AB” image that will be concatenated with the input L image to produce the final colorized image. An overview is shown below.

<p align="center">
  <img alt="flow" src="https://user-images.githubusercontent.com/50789540/171644257-a9d49515-fe88-405d-803a-40ccd66f0e86.png">
  <img alt="example" src="https://user-images.githubusercontent.com/50789540/171644603-d3bde206-4c83-4d50-851a-74dd5e94ec50.png">
</p>
[example : Real-Time User-Guided Image Colorization with Learned Deep Priors](https://richzhang.github.io/InteractiveColorization)

<br><br><br>

# MSAU-Net Model Architecture
* Multi-scale Attention(Spatial & Channel) U-Net
<p align="center">
  <img width="873" alt="network_overview" src="https://user-images.githubusercontent.com/39505929/172365914-25d837cb-becb-4c76-8985-72904026e82e.png">
</p>

* Multi Scale Convolution Block
<p align="center">
<img width="607" alt="image" src="https://user-images.githubusercontent.com/39505929/172366496-efe69692-e4a9-4b76-8454-b3f5161df798.png">
</p>

* Skip-connection with Spatial & Channel Attention
<p align="center">
  <img width="1030" alt="image" src="https://user-images.githubusercontent.com/39505929/172370340-7130cb57-4c8b-41d0-9001-f9a730e09f15.png">
</p>

# Loss Function 
<p align="center">
   <img width="1030" alt="image" src="https://user-images.githubusercontent.com/39505929/172368686-9223e1d9-3e34-480f-9623-60cb401f2619.png">
</p>

<br><br>
## Training dataset
* Dataset Size : 10000 / 2000
* Using Place365 and ImageNet datasets
* Image size : 256 x 256
<p align="center">
  <img alt="skip-connection" width="800" src="https://user-images.githubusercontent.com/50789540/171644914-05a0c6ea-99a8-4452-a12a-c6d136b54fab.png">
</p>

<br><br>

## Using Hyper-parameter
* Batch size : 16 <br>
* Optimizer : Adam (betas = (0.5, 0.999))<br>
* Learning rate : 5e-4 , Decrease 5e-4/70 every time the validation loss falls from PSNR 33
* Develop Environment RTX Titan in CUDA 10.1 

<br>
<br>


## Experiment & Result
* Model Robust Study
* Validation is carried out with additional datasets to assess the possibility of overfitting specific data
* Conduct performance evaluation with 2017 coco test dataset
* Colorization was performed based on the hint ratio 0.5 %

|Model|PSNR|SSIM|
|------|---|---|
|Ours|34.0958|0.9679|
<br>

* Ablation Study
* It was experimentally confirmed how much the performance of additional components affected
|Model|PSNR|SSIM|
|------|---|---|
|Unet|30.9374|0.9647|
|Attention Unet|31.7498|0.9683|
|Ours|31.7905|0.9685|
    
## Test dataset Result 
<p align="center">
  <img alt="ex1" width="800" src="https://user-images.githubusercontent.com/39505929/172370031-9cfed122-d57c-42ba-9c87-15f47d635fd4.png">
  <img alt="ex2" width="800" src="https://user-images.githubusercontent.com/39505929/172370049-5feb6a38-5ec3-447b-9956-93ae9e32ed88.png">
</p>

<br><br>
## How to use model
You can download the optimal weight pth from the link below and run this model.
```
https://drive.google.com/file/d/1IowH8LEjdWut9chW8cxtM8d3oHGtfJ1d/view?usp=sharing
```

<br><br>

## Reference paper
1. Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
2. Zhao, Hang, et al. "Loss functions for image restoration with neural networks." IEEE Transactions on computational imaging 3.1 (2016): 47-57
3. Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Colorful image colorization." European conference on computer vision. Springer, Cham, 2016.
4. Zhang, Richard, et al. "Real-time user-guided image colorization with learned deep priors." arXiv preprint arXiv:1705.02999 (2017)
5. Woo, Sanghyun, et al. "Cbam: Convolutional block attention module." Proceedings of the European conference on computer vision (ECCV). 2018
6. Xiao, Yi, et al. "Interactive deep colorization using simultaneous global and local inputs." ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.
7. Khanh, Trinh Le Ba, et al. "Enhancing U-Net with spatial-channel attention gate for abnormal tissue segmentation in medical imaging." Applied Sciences 10.17 (2020): 5729.
8. Su, Run, et al. "MSU-Net: Multi-scale U-Net for 2D medical image segmentation." Frontiers in Genetics 12 (2021): 140.

<br><br>


## License
This project belongs to computer vision Team 4, Gachon Univ. (2022-1)
