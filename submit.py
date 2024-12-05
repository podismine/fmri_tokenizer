import subprocess
# 25 /6 = 4 * 4 = 16h

cc = 0
for state in [2,4,8,16,32]:
    for transit in [2,4,8,16,32]:
        # cmd = ['sbatch','--job-name',f'token_{state}_{transit}','submit_task.sh',str(cc % 2),str(state),str(transit)]
        base_cmd = f'sbatch --job-name=token_{state}_{transit} --partition=gpu --output=/home/yyang/yang/fmri_tokenizer/run_logs/mamba_token_{state}_{transit}.log --error=/home/yyang/yang/fmri_tokenizer/run_logs/mamba_token_{state}_{transit}.log'
        # subprocess.run(base_cmd)
        cmd = base_cmd.split(' ')
        cmd.append(f'--wrap=/bin/bash -c "source /home/yyang/miniconda3/bin/activate; CUDA_VISIBLE_DEVICES={int(cc % 2)} python 02-cvqvae_train.py --model mamba -l 2 --hidden 256 --epochs 20000 -st {state} -tt {transit} --log_name mamba_run_token"')
        print(cmd)
        subprocess.run(cmd)
        cc +=1