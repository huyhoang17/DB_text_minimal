lint:
	yapf -i src/*.py

ioueval:
	python3 src/iou.py \
	--iou 0.4 \
	--area 0.8

deteval:
	python3 src/deteval.py \
	--tp 0.4 \
	--tr 0.8

train: lint
	python3 src/train.py

img_path=./assets/foo.jpg
model_path=./models/db_resnet18.pth
thresh=0.25
box_thresh=0.50
unclip_ratio=1.5

test-heatmap:
	python3 src/test.py --image_path $(img_path) \
	--model_path $(model_path) \
	--heatmap True \
	--is_output_polygon True \
	--unclip_ratio $(unclip_ratio) \
	--thresh $(thresh)

test-poly:
	python3 src/test.py --image_path $(img_path) \
	--model_path $(model_path) \
	--heatmap False \
	--is_output_polygon True \
	--unclip_ratio $(unclip_ratio) \
	--thresh $(thresh) \
	--box_thresh $(box_thresh)

test-rect:
	python3 src/test.py --image_path $(img_path) \
	--model_path $(model_path) \
	--heatmap False \
	--is_output_polygon False \
	--unclip_ratio $(unclip_ratio) \
	--thresh $(thresh) \
	--box_thresh $(box_thresh)

test-all: test-heatmap test-poly test-rect