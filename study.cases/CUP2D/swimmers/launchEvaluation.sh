for E in {0..9}; do export EVAL=$E; ./sbatch-eval-vracer-swimmer.sh halfDisk_largeDomain.eval 0; done

for E in {0..9}; do export EVAL=$E; ./sbatch-eval-vracer-swimmer.sh hydrofoil_largeDomain.eval 1; done

for E in {0..9}; do export EVAL=$E; ./sbatch-eval-vracer-swimmer.sh hydrofoil_largeDomain.eval.halfdisk 0; done

for E in {0..9}; do export EVAL=$E; ./sbatch-eval-vracer-swimmer.sh halfDisk_largeDomain.eval.hydrofoil 1; done

for E in {0..9}; do export EVAL=$E; ./sbatch-eval-vracer-swimmer.sh multitask_largeDomain.eval.halfdisk -1; done

for E in {0..9}; do export EVAL=$E; ./sbatch-eval-vracer-swimmer.sh multitask_largeDomain.eval.hydrofoil -1; done