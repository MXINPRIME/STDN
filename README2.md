# STDN  
**Structure-Texture Disentangled Network (STDN) for Underwater Image Enhancement**  
*The Visual Computer Journal* | [Permanent Resources](https://github.com/MXINPRIME/STDN) | [Cite This Work](#citation)

This repository provides the PyTorch implementation for our paper "*Structure-Texture Disentangled Network (STDN) for Underwater Image Enhancement*", published in *The Visual Computer*. Comprehensive documentation, datasets, and pretrained models are permanently hosted on [GitHub](https://github.com/MXINPRIME/STDN).

---

## Table of Contents  
1. [Dependencies and Requirements](#dependencies)  
2. [Installation](#installation)  
3. [Dataset Preparation](#datasets)  
4. [Training](#training)  
5. [Testing](#testing)  
6. [Method Overview](#method)  
7. [Pretrained Models](#models)  
8. [Reproducibility Guide](#reproducibility)  
9. [Citation](#citation)  

---

<a name="dependencies"></a>
## Dependencies and Requirements  
- **CUDA 10.1**  
- **Python 3.6+**  
- **PyTorch 1.6.0**  
- **Torchvision 0.7.0**  
- **OpenCV 3.4**  

Install dependencies:  
```bash
pip install torch==1.6.0 torchvision==0.7.0 opencv-python==3.4.0.14
```

---

<a name="installation"></a>
## Installation  
Clone the repository:  
```bash
git clone https://github.com/MXINPRIME/STDN.git
cd STDN
```

---

<a name="datasets"></a>
## Dataset Preparation  
### Supported Datasets  
| Dataset | Link | Description |  
|---------|------|-------------|  
| **UIEB** | [Download](https://ieeexplore.ieee.org/document/8917818) | 950 underwater images with reference labels |  
| **EUVP** | [Download](http://irvlab.cs.umn.edu/resources/euvp-dataset) | Paired/Unpaired underwater images |  
| **RUIE** | [Download](https://ieeexplore.ieee.org/document/8949763) | Three sub-datasets for different conditions |  
| **USOD** | [Download](https://github.com/xahidbuffon/SVAM-Net) | Underwater salient object detection |  

### Organize Data  
1. Place datasets in `./data/{dataset_name}/train` and `./data/{dataset_name}/test`.  
2. Generate image lists:  
```bash
python ./data/make_img_list.py --dataset_root ./data/UIEB --output_list ./data/list/UIEB_train.txt
```

---

<a name="training"></a>
## Training  
### Step 1: Download Pretrained VGG Models  
- **MyVGG.pt**: [Google Drive](https://drive.google.com/file/d/1v67HJre81RrNJbnLmdpspwSsiMkLBSnP/view?usp=sharing) | [Baidu Disk](https://pan.baidu.com/s/1pRx5zGLfV2Co0x_BcJOtJQ?pwd=c915)  
- **vggfeature.pth**: [Google Drive](https://drive.google.com/file/d/1TUmfNIPT6PIf0sVNl88CZiqtkNOh13jq/view) | [Baidu Disk](https://pan.baidu.com/s/1bfbThbMeErJJYLv693FuSg?pwd=84zk)  

Place them in `./pretrain/`.  

### Step 2: Run Training  
```bash
python train.py --train_root ./data/UIEB/train --train_list ./data/list/UIEB_train.txt
```
Checkpoints are saved in `./checkpoints/`.

---

<a name="testing"></a>
## Testing  
### Step 1: Download Pretrained STDN Model  
- **UIEB Model**: [Google Drive](https://drive.google.com/file/d/1VqryfoKZBenS4mlFM3SZL338pX3a6m9B/view?usp=sharing) | [Baidu Disk](https://pan.baidu.com/s/1Aq_1JA46sW6uyyU_Xibjiw?pwd=nacf)  

Place it in `./checkpoints/`.  

### Step 2: Run Inference  
```bash
python test_UIEB.py --test_root ./data/UIEB/test --test_list ./data/list/UIEB_test.txt
```
Enhanced images are saved in `./checkpoints/test/`.

---

<a name="method"></a>
## Method Overview  
### Key Algorithms  
1. **Structure-Texture Disentanglement**  
   - Dual-branch architecture separates structural and textural features.  
   - Structural branch uses dilated convolutions; texture branch employs wavelet transforms.  

2. **Hybrid Loss Function**  
   - Combines L1 loss, perceptual loss (VGG-based), and adversarial loss.  

For implementation details, see [Algorithm Documentation](https://github.com/yourusername/STDN/wiki/Algorithm-Details).

---

<a name="models"></a>
## Pretrained Models  
All models are permanently hosted on:  
- [Google Drive Folder](https://drive.google.com/drive/folders/your_folder_id)  
- [Baidu Disk](https://pan.baidu.com/s/your_link)  

---

<a name="reproducibility"></a>
## Reproducibility Guide  
To replicate our experiments:  
1. Follow the [Training Guide](https://github.com/yourusername/STDN/wiki/Training-Guide).  
2. Use the provided [Evaluation Scripts](https://github.com/yourusername/STDN/tree/main/evaluation) for PSNR/SSIM metrics.  

---

<a name="citation"></a>
## Citation  
```bibtex
@article{STDN2023,
  title={Structure-Texture Disentangled Network for Underwater Image Enhancement},
  author={Your Name and Co-authors},
  journal={The Visual Computer},
  year={2023},
  volume={39},
  pages={4025--4039},
  doi={10.1007/s00371-023-03134-1}
}
```
