#!/bin/bash

conda run -n medsam python medsam2_infer_CT_lesion_npz_recist.py \
    --checkpoint /mnt/D/models/MedSAM2/MedSAM2_CTLesion.pt \
    --cfg sam2/configs \
    -i data/validation_public_npz \
    -o data/test_output \
    --save_overlay
