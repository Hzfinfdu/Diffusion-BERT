CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 29500 DDP_main_conditional.py \
  --lr 5e-5 \
  --batch_size 64 \
  --task_name 'QT' \
  --eval_steps 50
