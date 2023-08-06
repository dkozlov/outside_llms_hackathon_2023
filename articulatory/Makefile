SHELL          := /bin/bash
REGISTRY       := dfkozlov
GIT_REPO       := $$(basename -s .git `git config --get remote.origin.url`)
GIT_BRANCH     := $$(if [ -n "$$BRANCH_NAME" ]; then echo "$$BRANCH_NAME"; else git rev-parse --abbrev-ref HEAD; fi)
GIT_BRANCH     := $$(echo "${GIT_BRANCH}" | tr '[:upper:]' '[:lower:]')
GIT_PROJECT_NAME   := articulatory
CONTAINER_NAME := "${GIT_PROJECT_NAME}"
GIT_SHA1       := $$(git rev-parse HEAD)
NAME           := ${REGISTRY}/${CONTAINER_NAME}
IMG            := "${NAME}:${GIT_REPO}-${GIT_BRANCH}-${GIT_SHA1}"
LATEST         := "${NAME}:latest"
DOCKER_CMD     := docker

run:
	(${DOCKER_CMD} rm -f ${CONTAINER_NAME} || true) && \
	${DOCKER_CMD} run --rm -it \
      -v $(PWD)/input:/root/articulatory/egs/ema/voc1/input \
      -v $(PWD)/output:/root/articulatory/egs/ema/voc1/output \
      --name ${CONTAINER_NAME} ${LATEST} /bin/bash
clean:
	@ set -ex; \
	rm -rf mngu0_fema2w mngu0_ema2w hprc_no_m1f2_h2emaph_gru_joint_nogan_model hubert_large_ll60k.pt

init:
	@ set -ex; \
	git submodule update --init --recursive && \
    wget https://huggingface.co/s3prl/converted_ckpts/resolve/main/hubert_large_ll60k.pt && \
	docker run -it --rm -v $(PWD):/root -w /root python:3 bash -c "python3 -m pip install -U gdown && \
      gdown --folder https://drive.google.com/drive/folders/1OMoF8DgNkobANBVLTId067RgUVq27_WL && \
      gdown --folder https://drive.google.com/drive/folders/13_RrfSuAEMA17V6O6FIv8pSbZl8Z8vwm && \
      gdown --folder https://drive.google.com/drive/folders/1O-1kX_ngHf1T8EN7HXWABCaEIJLNqxUI"

build:
	@ set -ex; \
	${DOCKER_CMD} build -t ${IMG} -t ${LATEST} .

push:
	${DOCKER_CMD} push ${LATEST}
	${DOCKER_CMD} push ${IMG}
