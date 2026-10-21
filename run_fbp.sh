set -e
# 1) CycleGAN, CUT, VAE + Zeng                       ~1 h
python run_benchmark.py --data data/piglet --recon FBP --doses 5 10 25 --quick --workers 8 --tag v3 --baselines zeng
# 2) PhysHVS-LDCT (CADM): train, then sample          ~1.5-2 h
python train_cadm.py train --data data/piglet --recon FBP --doses 5 10 25 --sd_path ~/models/sd15 --iters 3000
python train_cadm.py sample --ckpt cadm_runs/cadm_FBP_s-n-c_f0/final --doses 5 10 25 --steps 25 --runs runs_v3
# 3) classical diffusion, one model per dose          ~1 h
python train_cddpm.py --data data/piglet --recon FBP --doses 5 10 25 --iters 5000 --steps 25 --runs runs_v3
# 4) final table with every method                    ~15 min
python run_benchmark.py --data data/piglet --recon FBP --doses 5 10 25 --quick --workers 8 --tag v3 \
    --models cyclegan cut vae cadm cddpm --baselines zeng
