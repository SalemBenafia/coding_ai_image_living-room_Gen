# =============================================================================
# Workshop pipeline — Kaggle -> MinIO -> BLIP -> LoRA -> SDXL
#
#   make all            run the whole pipeline (ingest -> publish)
#   make data           ingest + prepare + upscale (+ MinIO upload)
#   make captions       BLIP caption + Qwen enhance
#   make dataset        build the LoRA imagefolder
#   make train          fine-tune the SDXL LoRA
#   make publish        upload LoRA to MinIO + hot-reload ai-service
#
# Override knobs via env, e.g.  make train MAX_TRAIN_STEPS=1500 LORA_RANK=32
# =============================================================================
PY ?= python
INCLUDE_HIRES ?= 1
HIRES_PER_STYLE ?= 120
BLUR_THRESHOLD ?= 30
SIZE ?= 1024

ifeq ($(INCLUDE_HIRES),1)
HIRES_FLAG := --include-hires --hires-per-style $(HIRES_PER_STYLE)
else
HIRES_FLAG :=
endif

.PHONY: help all data ingest prepare upscale captions caption enhance dataset train publish clean

help:
	@grep -E '^#|^[a-zA-Z_-]+:' Makefile | sed 's/^# //'

all: data captions dataset train publish

data: ingest prepare upscale

ingest:
	$(PY) s01_ingest.py $(HIRES_FLAG) --upload

prepare:
	$(PY) s02_prepare.py --blur-threshold $(BLUR_THRESHOLD)

upscale:
	$(PY) s03_upscale.py --size $(SIZE) --upload

captions: caption enhance

caption:
	$(PY) s04_caption.py

enhance:
	$(PY) s05_enhance.py

dataset:
	$(PY) s06_build_dataset.py --upload

train:
	bash s07_train_lora.sh

publish:
	$(PY) s08_publish_lora.py

clean:
	rm -rf $(WORKSHOP_DATA)/prepared $(WORKSHOP_DATA)/upscaled \
	       $(WORKSHOP_DATA)/lora_dataset $(WORKSHOP_DATA)/*.csv output
