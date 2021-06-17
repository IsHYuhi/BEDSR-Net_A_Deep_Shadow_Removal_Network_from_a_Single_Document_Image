# BEDSR-Net A Deep Shadow Removal Network from a Single Document Image

This repository is unofficial implementation of  [BEDSR-Net: A Deep Shadow Removal Network From a Single Document Image](https://openaccess.thecvf.com/content_CVPR_2020/html/Lin_BEDSR-Net_A_Deep_Shadow_Removal_Network_From_a_Single_Document_CVPR_2020_paper.html) [Lin+, **CVPR** 2020] with PyTorch.

### [🔥Online Demo!(Google CoLab)](https://github.com/IsHYuhi/BEDSR-Net_A_Deep_Shadow_Removal_Network_from_a_Single_Document_Image/blob/main/demo.ipynb)



## Results
### Results from BEDSR-Net pretrained on Jung dataset  

|Shadow image|Non-shadow image|Attention Map|Background color|
|:-:|:-:|:-:|:-:|
|![](result/shadow_image.jpg)|![](result/non-shadow_image.jpg)|![](result/attention_map.jpg)|![](result/background_color.jpg)|

## TODO
* [x] implementation of BE-Net
* [x] training code for BE-Net
* [x] implementation of SR-Net
* [x] training code for BEDSR-Net
* [ ] implementation of ST-CGAN-BE
* [ ] calculating code (PSNR/SSIM)
* [ ] inference code
* [x] Democode in Colab.
* [ ] cleaning up / formatting 
* [ ] Writing README
* [ ] providing pretrained model
* [ ] providing synthesized data

## Requirements
* Python3.x
* PyTorch 1.8.0
* matplotlib==3.4.2
* albumentations==0.4.6

## Usage

### Training

### Testing

When you would like to test your own image.

## Results

## Trained model

## References
* BEDSR-Net_A_Deep_Shadow_Removal_Network_from_a_Single_Document_Image, Yun-Hsuan Lin, Wen-Chin Chen, Yung-Yu Chuang, **National Taiwan University**, [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Lin_BEDSR-Net_A_Deep_Shadow_Removal_Network_From_a_Single_Document_CVPR_2020_paper.html)
