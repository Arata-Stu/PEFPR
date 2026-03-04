# PEFPR

## setup

### setup python env
python3.11でvenv環境でテストしている

```bash
python3.11 -m venv env
source env/bin/activate
pip3 install -r requirements.txt

mkdir lib
cd lib
git clone https://github.com/facebookresearch/detectron2.git
pip3 install -e detectron2 --no-build-isolation
```

### install dataset
DSEC-Detをダウンロード 約600GB
```bash
DSEC_ROOT=/path/to/dsec/

bash scripts/download_dsec.sh $DSEC_ROOT
bash scripts/download_dsec_extra.sh $DSEC_ROOT
bash scripts/download_remapped_images.sh $DSEC_ROOT
```

## run

### train
RGBだけを利用する学習
```bash
python3 train.py \
dataset=dsec_det \
model=yolox \
dataset.dataset_root=/path/to/dataset/ \
dataset.use_events=false \
dataset.use_image=true \
training.pretrained_path=./weights/yolox_l.pth \
training.batch_size=16 \
training.num_workers=12 \
wandb.project=yolox \
wandb.name=yolox \
```

### eval
RGBだけ
```bash
python3 eval.py \
dataset=dsec_det \
model=yolox \
dataset.dataset_root=/path/to/dataset/ \
dataset.use_events=false \
dataset.use_image=true \
ckpt_path=/path/to/.pth
```