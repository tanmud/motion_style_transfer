# Video Style and Motion Transfer

### Training Single Video

```bash
python MotionDirector_train.py --config ./configs/config_single_video.yaml
```

### Inference Single Video

```bash
python MotionDirector_inference.py \
  --model /path/to/zeroscope_v2_576w/ \
  --prompt "A tiger running" \
  --checkpoint_folder ./outputs/train/INPUT_TRAINING_FOLDER \
  --checkpoint_index 150 \
  --content-lora ./custom/content/lora \
  --style-lora ./custom/style/lora
```
