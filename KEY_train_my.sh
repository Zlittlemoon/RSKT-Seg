#!/bin/sh

BASE=train_outputs/rskt_train_$(date +%Y%m%d_%H%M)
mkdir -p "$BASE"

sh run.sh configs/vitl_336_DLRSD.yaml 4 "$BASE/output_vitl_336_DLRSD/"
sh run.sh configs/vitb_384_DLRSD.yaml 4 "$BASE/output_vitb_384_DLRSD/"

sh run.sh configs/vitl_336_iSAID.yaml 4 "$BASE/output_vitl_336_iSAID/"
sh run.sh configs/vitb_384_iSAID.yaml 4 "$BASE/output_vitb_384_iSAID/"