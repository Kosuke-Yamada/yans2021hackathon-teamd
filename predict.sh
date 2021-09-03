python ./src/predict.py \
    --input_plain_path ./dataset/yans2021hackathon_plain/ \
    --input_annotation_path ./dataset/yans2021hackathon_annotation/ \
    --output_path ./output/ \
    --category Company \
    --block char \
    --model shiba \
    --batch_size 32 \
    --cuda 3