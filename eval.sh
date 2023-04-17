CUDA_VISIBLE_DEVICES=0 python3 eval.py \
    --eval_corrupt \
    --ckpt outputs/cls_1024_0.3/models/model.t7 \
    --model dual_dgc \
    --if_attn