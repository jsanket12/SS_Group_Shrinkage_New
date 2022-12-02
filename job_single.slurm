#!/bin/bash
#SBATCH --job-name=Sun2021_cifar_run  # create a short name for your job
#SBATCH --account=FASTBAYES
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=BEGIN        # send email when job begins
#SBATCH --mail-type=END          # send email when job ends
#SBATCH --mail-user=sjantre@anl.gov

# module purge

# module load anaconda3/2020.11

# export PATH="/gpfs/fs1/soft/swing/manual/anaconda3/2020.11/bin:$PATH"
source activate test1

# python3 SS_Gaussian_Node_resnet_new.py --depth 20 --nepoch 300 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --num_MC_test 11 --clip 1 --sigma_0 0.2
# python Gaussian_resnet.py --depth 32 --nepoch 300 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 1024 --num_MC_test 11 --clip 1 --sigma_0 0.2 
# python3 SS_Gaussian_Edge_resnet.py --depth 20 --nepoch 300 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 1024 --num_MC_test 1 --clip 1 --sigma_0 0.1414
# python SS_GHS_Node_resnet_center_reg.py --depth 20 --nepoch 300 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --num_MC_test 10 --sigma_0 0.2 --c_reg 1 --tau_0 1 --clip 1
# python GHS_resnet_reg.py --depth 20 --nepoch 300 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 1024 --num_MC_test 10 --sigma_0 0.1 --c_reg 1 --tau_0 1 --clip 1
# python SS_Group_Lasso_Node_resnet_non_center.py --depth 32 --nepoch 300 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --num_MC_test 11 --clip 1 --sigma_0 0.2 --lamb_c 4 --lamb_d 2
# python Group_Lasso_resnet_non_center.py --depth 32 --nepoch 300 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --num_MC_test 11 --clip 1 --sigma_0 0.2 --lamb_c 4 --lamb_d 2

# python SS_Group_Lasso_Node_resnet_non_center_new.py --seed 3 --depth 32 --nepoch 300 --fine_tune_nepoch 100 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --num_MC_test 11 --clip 1 --sigma_0 0.2 --lamb_c 4 --lamb_d 2
# python SS_Gaussian_Node_resnet_new.py --seed 3 --depth 32 --nepoch 300 --fine_tune_nepoch 100 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --num_MC_test 11 --clip 1 --sigma_0 0.2
# python SS_GHS_Node_resnet_non_center_reg_new.py --seed 3 --depth 32 --nepoch 300 --fine_tune_nepoch 100 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --num_MC_test 11 --clip 1 --sigma_0 0.2 --c_reg 1 --tau_0 1 

# python Sun2021_cifar_run_vb.py --model_path 'resnet_20_10percent_vb_relu_new_3/' --seed 3 --depth 20 --nepoch 300 --nepoch_fine_tune 100 --num_MC_test 11 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --sigma0 0.00004 --lambdan 0.000001 --prune_ratio 0.1 
# python Sun2021_cifar_run_vb.py --model_path 'resnet_20_20percent_vb_relu_new_3/' --seed 3 --depth 20 --nepoch 300 --nepoch_fine_tune 100 --num_MC_test 11 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --sigma0 0.000006 --lambdan 0.000001 --prune_ratio 0.2
# python Sun2021_cifar_run_vb.py --model_path 'resnet_32_10percent_vb_relu_new_3/' --seed 3 --depth 32 --nepoch 300 --nepoch_fine_tune 100 --num_MC_test 11 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --sigma0 0.00002 --lambdan 0.00001 --prune_ratio 0.1
# python Sun2021_cifar_run_vb.py --model_path 'resnet_32_5percent_vb_relu_new_3/' --seed 3 --depth 32 --nepoch 300 --nepoch_fine_tune 100 --num_MC_test 11 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --sigma0 0.00006 --lambdan 0.0000001 --prune_ratio 0.05

# python Sun2021_cifar_run.py --model_path 'resnet_20_10percent_relu_3/' --seed 3 --depth 20 --nepoch 300 --nepoch_fine_tune 100 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --sigma0 0.00004 --lambdan 0.000001 --prune_ratio 0.1
# python Sun2021_cifar_run.py --model_path 'resnet_20_20percent_relu_3/' --seed 3 --depth 20 --nepoch 300 --nepoch_fine_tune 100 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --sigma0 0.000006 --lambdan 0.000001 --prune_ratio 0.2
# python Sun2021_cifar_run.py --model_path 'resnet_32_10percent_relu_3/' --seed 3 --depth 32 --nepoch 300 --nepoch_fine_tune 100 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --sigma0 0.00002 --lambdan 0.00001 --prune_ratio 0.1
# python Sun2021_cifar_run.py --model_path 'resnet_32_5percent_relu_3/' --seed 3 --depth 32 --nepoch 300 --nepoch_fine_tune 100 --init_lr 0.1 --lr_decay_factor 10 --lr_decay_time 150 225 --batch_train 128 --sigma0 0.00006 --lambdan 0.0000001  --prune_ratio 0.05

# python SS_Gaussian_Node_resnet_new.py --seed 1 --depth 20 --nepoch 300 --init_lr 0.024179064159738344 --momentum 0.9868973069500376 --lr_decay_factor 5 --lr_decay_time 150 225 --batch_train 128 --num_MC_test 1 --clip 1 --sigma_0  0.8473518178783007 --temp 0.7225601563244488 --gamma_prior 0.009566626337770997

conda deactivate 