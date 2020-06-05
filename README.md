[![](https://forthebadge.com/images/badges/built-by-developers.svg)](https://forthebadge.com)
[![](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/huyhoang17/DB_text_minimal)
[![](https://forthebadge.com/images/badges/makes-people-smile.svg)](https://forthebadge.com)

# A Pytorch implementation of DB-Text paper

### Command

#### Train model

- Modify hyperparameters in config.yaml

```bash
python3 src/train.py
```

#### Test model

```bash
python3 src/test.py --image_path path-to-image
```

- For evaluation metric, please refer to [this](https://github.com/Megvii-CSG/MegReader/blob/master/concern/icdar2015_eval) repository

### Results

- Heatmap

![](./assets/heatmap_result_foo.jpg)

- Polygon result

![](./assets/poly_result_foo.jpg)

- Rotated rectangle result

![](./assets/rect_result_foo.jpg)

### TODO

- Support other dataset
	- [TotalText](https://github.com/cs-chan/Total-Text-Dataset) :heavy_check_mark:
	- [ICDAR2015](https://rrc.cvc.uab.es/?ch=4)
	- [COCO-Text](https://rrc.cvc.uab.es/?ch=5)
	- [Synthtext](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
	- [CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)
	- [ArT2019](https://rrc.cvc.uab.es/?ch=14)
- Convert code to pytorch-lightning
- Serve model with Torchserve
- Add callbacks
- Add metric & code evaluation (P/R/F1 - IoU-based Pascal eval) :heavy_check_mark:
- Add metric & code evaluation (P/R/F1 - Overlap-based DetEval eval)
- Model quantization
- Model pruning
- Docker / docker-compose

### Reference

- [Evaluation metrics](https://github.com/Megvii-CSG/MegReader/blob/master/concern/icdar2015_eval)