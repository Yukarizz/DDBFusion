DDBFusion: An Unified Image Decomposition and Fusion Framework
This is the official implementation of “DDBFusion: An unified image decomposition and fusion framework based on dual decomposition and Bézier curves”, published in Information Fusion 2025.

📖 Framework
🧠 Network Architecture
🚀 Usage
Important Note:

Our method processes input image pairs by converting them into grayscale latent features for fusion. For color images, we fuse the luminance channel and reconstruct the color details.

1. Grayscale Fusion
To generate fusion results for grayscale images:

Bash

python fuse_gray.py
2. Color Fusion (General)
To generate results for color multi-exposure and multi-focus images:

Bash

python fuse_rgb.py
3. Medical Image Fusion
To generate results for color medical images (e.g., MRI-CT, MRI-PET):

Bash

python fuse_medical.py
📂 Results
You can download the fusion results of DDBFusion from the link below: Download Results (Google Drive)

📝 Citation
If you find this work helpful for your research, please consider citing our paper:

代码段

@article{zhang2025ddbfusion,
  title={DDBFusion: An unified image decomposition and fusion framework based on dual decomposition and B{\'e}zier curves},
  author={Zhang, Zeyang and Li, Hui and Xu, Tianyang and Wu, Xiao-Jun and Kittler, Josef},
  journal={Information Fusion},
  volume={114},
  pages={102655},
  year={2025},
  publisher={Elsevier}
}
