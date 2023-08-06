### Build docker iamge
```
make clean
make init
make build
```
### Run docker image
```
make run
```
### Get audio.npy from auido.wav
```
cd /root/articulatory/egs/ema/voc1/input && \
wget https://www.signalogic.com/melp/EngSamples/Orig/male.wav -O /root/articulatory/egs/ema/voc1/input/audio.wav && \
cd /root/articulatory/egs/ema/voc1 && \
python3 local/predict_ema.py hprc_no_m1f2_h2emaph_gru_joint_nogan_model /root/articulatory/egs/ema/voc1/input /root/articulatory/egs/ema/voc1/output
```

### Predict wav
```
echo "audio /root/articulatory/egs/ema/voc1/output/audio.npy" > /root/articulatory/egs/ema/voc1/output/feats.scp && \
python3 local/predict_wav.py \
--scp /root/articulatory/egs/ema/voc1/output/feats.scp \
--outdir /root/articulatory/egs/ema/voc1/output/ \
--checkpoint /root/mngu0_fema2w/best_mel_ckpt.pkl \
--config /root/mngu0_fema2w/config.yml && \
ls /root/articulatory/egs/ema/voc1/output/
```
