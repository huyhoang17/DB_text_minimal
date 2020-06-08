lint:
	yapf -i src/*.py

ioueval:
	python3 src/iou.py \
	--iou 0.4 \
	--area 0.8

deteval:
	python3 src/deteval_v2.py \
	--tp 0.4 \
	--tr 0.8

train:
	python3 src/train.py

test-heatmap:
	python3 src/test.py --image_path ./assets/foo.jpg \
	--model_path models/db_resnet18.pth \
	--heatmap True \
	--is_output_polygon True \
	--unclip_ratio 1.5 \
	--thresh 0.7

test-poly:
	python3 src/test.py --image_path ./assets/foo.jpg \
	--model_path models/db_resnet18.pth \
	--heatmap False \
	--is_output_polygon True \
	--unclip_ratio 3.0 \
	--thresh 0.3 \
	--box_thresh 0.7

test-rect:
	python3 src/test.py --image_path ./assets/foo.jpg \
	--model_path models/db_resnet18.pth \
	--heatmap False \
	--is_output_polygon False \
	--unclip_ratio 1.5 \
	--thresh 0.3 \
	--box_thresh 0.7