### Multi-modal Gated Mixture of Local-to-Global Experts for Dynamic Image Fusion

![](docs/imgs/moefusion.png)

> **[Multi-modal Gated Mixture of Local-to-Global Experts for Dynamic Image Fusion (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Cao_Multi-Modal_Gated_Mixture_of_Local-to-Global_Experts_for_Dynamic_Image_Fusion_ICCV_2023_paper.html)**,  
> Bing Cao<sup>\*</sup>, Yiming Sun<sup>\*</sup>(Equal contribution), Pengfei Zhu<sup>†</sup>, Qinghua Hu,
> arXiv preprint ([arXiv:2302.01392](https://arxiv.org/abs/2302.01392)) / CVPR [Open access](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Multi-Modal_Gated_Mixture_of_Local-to-Global_Experts_for_Dynamic_Image_Fusion_ICCV_2023_paper.pdf). 

The repo is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

### Introduction
Infrared and visible image fusion aims to integrate comprehensive information from multiple sources to achieve superior performances on various practical tasks, such as detection, over that of a single modality. However, most existing methods directly combined the texture details and object contrast of different modalities, ignoring the dynamic changes in reality, which diminishes the visible texture in good lighting conditions and the infrared contrast in low lighting conditions. To fill this gap, we propose a dynamic image fusion framework with a multi-modal gated mixture of local-to-global experts, termed MoE-Fusion, to dynamically extract effective and comprehensive information from the respective modalities. Our model consists of a Mixture of Local Experts (MoLE) and a Mixture of Global Experts (MoGE) guided by a multi-modal gate. The MoLE performs specialized learning of multi-modal local features, prompting the fused images to retain the local information in a sample-adaptive manner, while the MoGE focuses on the global information that complements the fused image with overall texture detail and contrast. Extensive experiments show that our MoE-Fusion outperforms state-of-the-art methods in preserving multi-modal image texture and contrast through the local-to-global dynamic learning paradigm, and also achieves superior performance on detection tasks.

## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Getting Started

Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage.


## Citation

```
@InProceedings{Cao_2023_ICCV,
    author    = {Cao, Bing and Sun, Yiming and Zhu, Pengfei and Hu, Qinghua},
    title     = {Multi-Modal Gated Mixture of Local-to-Global Experts for Dynamic Image Fusion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {23555-23564}
}
```