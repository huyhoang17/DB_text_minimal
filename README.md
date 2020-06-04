### A Pytorch implementation of DB-Text paper

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
- Add metric (P/R/F1 | PascalVOC/DetEval)
- Model quantization
- Model pruning
- Docker / docker-compose
