# Video Style and Motion Transfer

### Training Single Video

```bash
python MotionDirector_train.py --config ./configs/config_single_video.yaml
```

### Inference Single Video

```bash
python MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "A tiger running" \
  --checkpoint_folder ./outputs/train/INPUT_TRAINING_FOLDER \
  --checkpoint_index 150
```

### FOR ME

```bash
python MotionDirector_inference.py \
  --model /work/10572/tmudali/vista/MotionDirector/models/zeroscope_v2_576w/ \
  --prompt "A tiger running" \
  --checkpoint_folder ./outputs/train/train_2026-01-23T14-56-49 \
  --checkpoint_index 150
```
