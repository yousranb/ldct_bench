export PATH=$HOME/conda_envs/ldct/bin:$PATH
cd ~/ldct_bench
# GPU 0: finish the long CADM (resumes at it10000), then sample it as "cadm_long"
CUDA_VISIBLE_DEVICES=0 sh -c "\
python train_cadm.py train --data data/piglet --recon FBP --doses 5 10 25 --sd_path ~/models/sd15 --iters 15000 --tag long && \
python train_cadm.py sample --ckpt cadm_runs/cadm_FBP_s-n-c_f0_long/final --doses 5 10 25 --steps 25 --runs runs_v3 --name cadm_long" \
  > cadm_long.log 2>&1 &
# GPU 1: ablation only (4 variants, 3000 iterations each, like the reference CADM)
CUDA_VISIBLE_DEVICES=1 python run_ablation.py --data data/piglet --recon FBP --doses 5 10 25 --sd_path ~/models/sd15 \
   --iters 3000 --steps 25 --runs runs_v3 --configs no_pm no_parm ndct_only no_dose > ablation.log 2>&1 &
wait
