. ./env.sh

python main.py \
    --data_dir $DATA_DIR \
    --image_root $IMAGE_ROOT \
    --full_annot $FULL_ANNOT \
    --eval_only $EVAL_ONLY \
    --rendering_root $RENDERING_ROOT \
    --output_dir $OUTPUT_DIR \
    --override_output 1 \
    --max_iter 80000 \
    --checkpoint $CHECKPOINT \
    --resume $RESUME \
    --workers $NUM_WORKERS \
    --freq_scale image \
    --lr 1e-3 \
    --augment 1 \
    --noc_weights $NOC_WEIGHTS \
    --custom_noc_weights $NOC_WEIGHTS \
    --wild_retrieval 0 \
    --steps 60000 \
    --retrieval_mode $RETRIEVAL_MODE \
    --e2e $E2E \
    --seed $SEED
