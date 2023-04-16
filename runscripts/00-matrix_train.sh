#!/bin/sh

echo "Starting training for matrix experiments"

python ../scripts/52-matrix_train.py casE 1_corner --wandb_project "matrix_case_1corner"
python ../scripts/52-matrix_train.py casE 2_corners_recog --wandb_project "matrix_case_2corners_recog"
python ../scripts/52-matrix_train.py casE 2_corners_ern --wandb_project "matrix_case_2corners_ern"
python ../scripts/52-matrix_train.py casE 2_corners_diag --wandb_project "matrix_case_2corners_diag"
python ../scripts/52-matrix_train.py casE 3_corners --wandb_project "matrix_case_3corners"
python ../scripts/52-matrix_train.py casE 4_corners --wandb_project "matrix_case_4corners"
python ../scripts/52-matrix_train.py casE all --wandb_project "matrix_case_all"

python ../scripts/52-matrix_train.py csy4 1_corner --wandb_project "matrix_csy4_1corner"
python ../scripts/52-matrix_train.py csy4 2_corners_recog --wandb_project "matrix_csy4_2corners_recog"
python ../scripts/52-matrix_train.py csy4 2_corners_ern --wandb_project "matrix_csy4_2corners_ern"
python ../scripts/52-matrix_train.py csy4 2_corners_diag --wandb_project "matrix_csy4_2corners_diag"
python ../scripts/52-matrix_train.py csy4 3_corners --wandb_project "matrix_csy4_3corners"
python ../scripts/52-matrix_train.py csy4 4_corners --wandb_project "matrix_csy4_4corners"
python ../scripts/52-matrix_train.py csy4 all --wandb_project "matrix_csy4_all"

echo "Training finished"
