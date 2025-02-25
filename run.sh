echo "=> Generate data.."

cd data/ || exit

rm -r mnist/all_data

python generate_data.py \
  --dataset mnist \
  --n_clients 10 \
  --iid \
  --frac 0.2 \
  --save_dir mnist \
  --seed 1234

cd ../

echo "=> Train.."

python train.py \
  --experiment "mnist" \
  --n_rounds 10 \
  --local_steps 100 \
  --sampling_rate 0.2 \
  --sample_with_replacement \
  --local_optimizer sgd \
  --local_lr 0.001 \
  --server_optimizer sgd \
  --server_lr 0.1 \
  --bz 128 \
  --device "cpu" \
  --log_freq 1 \
  --verbose 1 \
  --logs_dir "logs/mnist/" \
  --seed 12
