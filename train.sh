CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --exp_name=cls_1024_0.3 \
    --num_points=1024 \
    --k=20 \
    --model 'dual_dgc' \
    --if_attn \
    --batch_size 64 \
    --test_batch_size 64 \
    --workers 16 \
    --epochs 350 \
    --ratio 0.3
    
