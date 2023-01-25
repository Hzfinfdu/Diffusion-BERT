CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 29501 DDP_main.py \
  --lr 5e-5 \
  --batch_size 128 \
  --timestep 'layerwise' \
  --from_scratch false