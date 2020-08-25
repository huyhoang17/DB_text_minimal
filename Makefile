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

cwd=$(CURDIR)
serve_dir=$(cwd)/model_store
img_path=$(cwd)/assets/foo18.jpg
model_path=$(cwd)/models/db_resnet18.pth
# model_path=./models/ctw_best_cp_1806.pth
# model_path=./models/quantized/db_resnet18_quantized.pth
thresh=0.25
box_thresh=0.50
unclip_ratio=1.5
device=cuda  # cpu / cuda

# TESTING
test-heatmap:
	python3 $(cwd)/src/test.py --image_path $(img_path) \
	--model_path $(model_path) \
	--device $(device) \
	--heatmap True \
	--is_output_polygon True \
	--unclip_ratio $(unclip_ratio) \
	--thresh $(thresh)

test-poly:
	python3 $(cwd)/src/test.py --image_path $(img_path) \
	--model_path $(model_path) \
	--device $(device) \
	--heatmap False \
	--is_output_polygon True \
	--unclip_ratio $(unclip_ratio) \
	--thresh $(thresh) \
	--box_thresh $(box_thresh)

test-rect:
	python3 $(cwd)/src/test.py --image_path $(img_path) \
	--model_path $(model_path) \
	--device $(device) \
	--heatmap False \
	--is_output_polygon False \
	--unclip_ratio $(unclip_ratio) \
	--thresh $(thresh) \
	--box_thresh $(box_thresh)

test-all: test-heatmap test-poly test-rect

# MODEL SERVING
model_name=dbtext
save-jit:
	python3 $(cwd)/src/save_jit.py

save-trt:
	python3 $(cwd)/src/save_trt.py

ts-archive:
	torch-model-archiver \
	--model-name $(model_name) \
	--version 1.0 \
	--serialized-file $(cwd)/models/db_resnet18_jit.pt \
	--handler $(cwd)/src/db_handler.py \
	--export-path $(serve_dir) -f

ts-start:
	torchserve --start \
	--model-store $(serve_dir) \
	--models $(model_name)=dbtext.mar

ts-stop:
	torchserve --stop

ts-restart: ts-stop ts-archive ts-start

ts-curl:
	curl -X POST http://127.0.0.1:8080/predictions/$(model_name) -T $(cwd)/assets/foo.jpg

ts-request:
	python3 $(cwd)/src/ts_request.py --image_path $(cwd)/assets/foo.jpg

### TEXT RECOGNITION
rect_model_path=$(cwd)/models/recognition/best_norm_ED.pth
# cropped char images
test-img:
	python3 $(cwd)/src/test_ocr.py \
	--device $(device) \
	--img_path $(cwd)/tmp/reconized/word_10.jpg \
	--workers 1 \
	--batch_size 1 \
	--saved_model $(rect_model_path) \
	--Transformation None --FeatureExtraction ResNet \
	--SequenceModeling BiLSTM --Prediction Attn

# cropped char images
test-folder:
	python3 $(cwd)/src/test_ocr.py \
	--device $(device) \
	--img_folder $(cwd)/tmp/reconized \
	--workers 1 \
	--batch_size 1 \
	--saved_model $(rect_model_path) \
	--Transformation None --FeatureExtraction ResNet \
	--SequenceModeling BiLSTM --Prediction Attn

### FULL PIPELINE
# detect --> recognize
test-pp:
	python3 $(cwd)/src/test_ocr.py \
	--device $(device) \
	--img_path $(cwd)/assets/foo.jpg \
	--out_path $(cwd)/tmp/ocr_04.jpg \
	--workers 1 \
	--batch_size 1 \
	--det_model_path $(model_path) \
	--saved_model $(rect_model_path) \
	--Transformation None --FeatureExtraction ResNet \
	--SequenceModeling BiLSTM --Prediction Attn

test-webcam:
	python3 $(cwd)/src/test_webcam.py \
	--show_video \
	--per_frame 1 \
	--device $(device) \
	--workers 1 \
	--batch_size 1 \
	--det_model_path $(model_path) \
	--saved_model $(rect_model_path) \
	--Transformation None --FeatureExtraction ResNet \
	--SequenceModeling BiLSTM --Prediction Attn

test-video:
	python3 $(cwd)/src/test_webcam.py \
	--video_path /home/phan.huy.hoang/testvideo.mp4 \
	--per_frame 1 \
	--device $(device) \
	--workers 1 \
	--batch_size 1 \
	--det_model_path $(model_path) \
	--saved_model $(rect_model_path) \
	--Transformation None --FeatureExtraction ResNet \
	--SequenceModeling BiLSTM --Prediction Attn