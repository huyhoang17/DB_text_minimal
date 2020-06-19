# LINTING
lint:
	yapf -i src/*.py

# EVALUATE
ioueval:
	python3 src/iou.py \
	--iou 0.4 \
	--area 0.8

deteval:
	python3 src/deteval.py \
	--tp 0.4 \
	--tr 0.8

# TRAINING
train: lint
	python3 src/train.py

img_path=./assets/foo.jpg
model_path=./models/db_resnet18.pth
thresh=0.25
box_thresh=0.50
unclip_ratio=1.5

# TESTING
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

# MODEL SERVING
model_name=dbtext
save-jit:
	python3 src/save_jit.py

ts-archive:
	torch-model-archiver \
	--model-name $(model_name) \
	--version 1.0 \
	--serialized-file /home/phan.huy.hoang/phh_workspace/DB_text_minimal/models/db_resnet18_jit.pt \
	--handler /home/phan.huy.hoang/phh_workspace/DB_text_minimal/src/db_handler.py \
	--export-path model_store -f

ts-start:
	torchserve --start \
	--model-store model_store \
	--models $(model_name)=dbtext.mar

ts-stop:
	torchserve --stop

ts-restart: ts-stop ts-archive ts-start

ts-curl:
	curl -X POST http://127.0.0.1:8080/predictions/dbtext -T ./assets/foo.jpg

ts-request:
	python3 src/ts_request.py --image_path ./assets/foo.jpg
