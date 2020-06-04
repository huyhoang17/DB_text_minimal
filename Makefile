train:
	python3 src/train.py

test:
	python3 src/test.py --image_path ./assets/foo.jpg \
	--model_path models/best_cp.pth \
	--heatmap False \
	--is_output_polygon False \
	--unclip_ratio 1.5 \
	--thresh 0.7