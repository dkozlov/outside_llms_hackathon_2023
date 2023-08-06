FROM python:3.10
COPY articulatory /root/articulatory
WORKDIR /root/articulatory
RUN pip3 install -e .
RUN pip3 install s3prl resampy --upgrade
COPY mngu0_ema2w /root/mngu0_ema2w
COPY mngu0_fema2w /root/mngu0_fema2w
COPY hprc_no_m1f2_h2emaph_gru_joint_nogan_model /root/hprc_no_m1f2_h2emaph_gru_joint_nogan_model
COPY hubert_large_ll60k.pt /root/.cache/s3prl/download/544f5b47ce48ad4c05bc671e6e1ee1a077f5804393136a526c40d78719a314ee.hubert_large_ll60k.pt
COPY patches/predict_ema.py /root/articulatory/egs/ema/voc1/local/predict_ema.py
RUN mkdir -p /root/articulatory/egs/ema/voc1/input && \
  mkdir -p /root/articulatory/egs/ema/voc1/output && \
  mkdir -p /root/articulatory/egs/ema/voc1/exp && \
  ln -s /root/hprc_no_m1f2_h2emaph_gru_joint_nogan_model /root/articulatory/egs/ema/voc1/exp/hprc_no_m1f2_h2emaph_gru_joint_nogan_model
