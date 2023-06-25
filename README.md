## STDN
This is the PyTorch implementation for "Structure-Texture Disentangled Network (STDN) for Underwater Image Enhancement"
<div align=center>
<img src="Fig/overview.png" width="80%">
</div>

## Requirements
- CUDA 10.1
- Python 3.6 (or later)
- Pytorch 1.6.0
- Torchvision 0.7.0
- OpenCV 3.4

## test
please download the pretrained VGG model [MyVGG.pt](https://drive.google.com/file/d/1v67HJre81RrNJbnLmdpspwSsiMkLBSnP/view?usp=sharing) and [vggfeature.pth](https://drive.google.com/file/d/1TUmfNIPT6PIf0sVNl88CZiqtkNOh13jq/view?usp=share_link) and put them into the folder 'pretrain'

Baidu Disk pretrained VGG model [MyVGG.pt](https://pan.baidu.com/s/1pRx5zGLfV2Co0x_BcJOtJQ?pwd=c915)  and a [vggfeature.pth](https://pan.baidu.com/s/1bfbThbMeErJJYLv693FuSg?pwd=84zk)


## Dataset
Please download the following datasets:
*   [UIEB](https://ieeexplore.ieee.org/document/8917818)
*   [EUVP](http://irvlab.cs.umn.edu/resources/euvp-dataset)
*   [RUIE](https://ieeexplore.ieee.org/document/8949763)
*   [USOD](https://github.com/xahidbuffon/SVAM-Net)
