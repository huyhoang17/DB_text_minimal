[![](https://forthebadge.com/images/badges/built-by-developers.svg)](https://forthebadge.com)
[![](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/huyhoang17/DB_text_minimal)
[![](https://forthebadge.com/images/badges/makes-people-smile.svg)](https://forthebadge.com)

# A Pytorch implementation of [DB-Text paper](https://arxiv.org/abs/1911.08947)

*Make awesome things that matter.*

<p align="center">
  <img src="./assets/out.gif" />
</p>

### Command

#### Train model

- Modify parameters in config.yaml

```bash
make train
```

#### Test model

```bash
make test-all
```

#### Evaluate model

- For evaluation metric, please refer to [this](https://github.com/Megvii-CSG/MegReader/blob/master/concern/icdar2015_eval) repository

```bash
# iou-based Pascal
make ioueval  # not recommend for polygon ground-truth

# overlap-based DetEval
make deteval
```

### History (on TotalText dataset)

#### Train history

![](./assets/train_history.png)

#### Test history

![](./assets/test_history.png)

![](./assets/test_img_history_01.png)

![](./assets/test_img_history_02.png)

### Some results on TotalText's test dataset

| Heatmap | Polygon | Rotated rectangle |
|:-----:|:-----:|:-----:|
| ![](./assets/tt_heatmap_01.jpg) | ![](./assets/tt_poly_01.jpg) |  ![](./assets/tt_rect_01.jpg) |
| ![](./assets/tt_heatmap_02.jpg) | ![](./assets/tt_poly_02.jpg) |  ![](./assets/tt_rect_02.jpg) |
| ![](./assets/tt_heatmap_03.jpg) | ![](./assets/tt_poly_03.jpg) |  ![](./assets/tt_rect_03.jpg) |

### Text-line detection (the model trained on CTW1500 dataset)

| Origin image | Text-line detected |
|:-----:|:-----:|
| ![](./assets/ctw_gt_01.jpg) | ![](./assets/ctw_result_01.jpg) |
| ![](./assets/ctw_gt_02.jpg) | ![](./assets/ctw_result_02.jpg) |

### Full pipeline

- Recognition model was trained on lmdb dataset

![](./assets/ocr_01.jpg)

![](./assets/ocr_02.jpg)

![](./assets/ocr_03.jpg)

### Metric evaluation (DetEval - P/R/HMean)

```bash
# for TotalText dataset
make deteval
```

| Method                   | image size | init lr | b-thresh | p-thresh | unclip ratio | Precision | Recall | F-measure |
|:--------------------------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:------------:|:---------------:|
| TotalText-resnet18-fcn (word-level) | 640 | 0.005 | 0.25 | 0.50 | 1.50 | 0.70 | 0.64 | 0.67 |
| CTW1500-resnet18-fcn (line-level) | 640 | 0.005 | 0.25 | 0.50 | 1.50 | 0.83 | 0.66 | 0.74 |

### ToDo

- [ ] Support other datasets
	- [x] [TotalText](https://github.com/cs-chan/Total-Text-Dataset)
	- [x] [ICDAR2015](https://rrc.cvc.uab.es/?ch=4)
	- [x] [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)
	- [x] [MSRA-TD500](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))
	- [ ] [COCO-Text](https://rrc.cvc.uab.es/?ch=5)
	- [ ] [Synthtext](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
	- [ ] [ArT2019](https://rrc.cvc.uab.es/?ch=14) (included Total-Text, SCUT-CTW1500 and Baidu Curved Scene Text)
- [ ] Convert code to pytorch-lightning
- [x] Serve model with Torchserve
- [x] Add metric callbacks (P/R/F1)
- [x] Add metric & code evaluation (P/R/F1 - IoU-based Pascal eval)
- [x] Add metric & code evaluation (P/R/F1 - Overlap-based DetEval eval)
- [ ] Model quantization
- [ ] Model pruning
- [ ] Docker / docker-compose
- [ ] Integrate with ONNX, TensorRT
- [x] Baseline text recognition model

### Reference

- [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)
- [Evaluation metrics](https://github.com/Megvii-CSG/MegReader/blob/master/concern/icdar2015_eval)
- [DBNet.pytorch](https://github.com/WenmuZhou/DBNet.pytorch)
- [DBNet.keras](https://github.com/xuannianz/DifferentiableBinarization/)
- [Real-time-Text-Detection](https://github.com/SURFZJY/Real-time-Text-Detection)
- [PSENet.pytorch](https://github.com/whai362/PSENet)
- [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
- [volksdep](https://github.com/Media-Smart/volksdep)
- [TedEval](https://github.com/clovaai/TedEval)
- [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)