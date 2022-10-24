#!/usr/bin/env python
import os
import os
import argparse
from korali.auxiliar.printing import *

def mkdir_p(dir):
    """Make a directory if it doesn't exist and create intermediates as well."""
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conduit",
        help="Conduit to use",
        choices=["Sequential", "Distributed"],
        default="Sequential",
        required=False,
    )
    parser.add_argument(
        "-N",
        "--nodes",
        help="[SLURM] Nodes to use",
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        "-n",
        "--ntasks",
        help="[SLURM] Number of total tasks to use",
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        "--ntasks-per-node",
        help="[SLURM] Number of total tasks to use",
        type=int,
        default=None,
        required=False
    )
    parser.add_argument(
        "-c",
        "--cpus-per-task",
        help="[SLURM] Number of cpus to use per task",
        type=int,
        default=12,
        required=False
    )
    parser.add_argument(
        "--job-duration",
        help="[SLURM] time",
        default="0-4",
        required=False
    )
    parser.add_argument(
        "-p",
        "--partition",
        help="[SLURM] partition to use.",
        choices=["debug", "large", "long", "low", "normal", "prepost", "xfer"],
        default="normal",
        required=False
    )
    parser.add_argument(
        "--submit",
        help="Whether to submit the jobs.",
        action="store_true",
        required=False
    )
    # script arguments ==================================================================================
    parser.add_argument(
        '--engine',
        help='NN backend to use',
        default='OneDNN',
        required=False)
    parser.add_argument(
        '--epochs',
        help='Maximum Number of epochs to run',
        default=1,
        type=int,
        required=False)
    parser.add_argument(
        '--optimizer',
        help='Optimizer to use for NN parameter updates',
        default='Adam',
        required=False)
    parser.add_argument(
        '--initialLearningRate',
        help='Learning rate for the selected optimizer',
        default=0.001,
        type=float,
        required=False)
    parser.add_argument(
        '--trainingSetSize',
        help='Batch size to use for training data',
        default=2**15,
        required=False)
    parser.add_argument(
        '--trainingBS',
        help='Batch size to use for training data',
        type=int,
        default=32,
        required=False)
    # parser.add_argument(
    #     "-s",
    #     "--save",
    #     help="Indicates if to save models to _korali_results.",
    #     required=False,
    #     action="store_true"
    # )
    parser.add_argument(
        "-l",
        "--load-model",
        help="Load previous model",
        required=False,
        action="store_true")
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     help="Run model with one hidden or many hidden layers (of same size).",
    #     type=str,
    #     required=True,
    #     choices=["single", "multi"]
    # )
    # parser.add_argument(
    #     "--weight-dim",
    #     help="Weights Siz",
    #     required=True,
    #     choices=[15,17,19,21,23,25],
    #     type=int,
    # )
    parser.add_argument(
        '--verbosity',
        help='Verbosity to print',
        default="Silent",
        required=False)
    parser.add_argument(
        '--other',
        help='Verbosity to print',
        default="",
        required=False)
    parser.add_argument(
        "--plot",
        help="Whether to submit the jobs.",
        action="store_true",
        required=False
    )
    args = parser.parse_args()
    print_header('Script', color=bcolors.HEADER, width=140)
    print_args(vars(args), sep=' ', header_width=140)
    # ===================================================================================================
    # if args.conduit == "Distributed":
    #     assert 2*args.nodes == args.batch_concurrency
    # COPY EXECUTABLE, UTILITIES And MODELS to SCRATCH ==================================================
    # EXECUTABLE = "run-scaling-benchmarks.py"
    # EXECUTABLE = "run-scaling-benchmarks-tensorflow.py"
    EXECUTABLE = "run-scaling-benchmarks-pytorch.py"
    # ===================================================================================================
    # threads = [1, 2, 4, 8, 16, 24, 28, 36]
    threads = [36]
    weights = [23]
    # weights = [15, 17, 19, 21, 23]
    models = ["multi"]
    # batch_sizes = [8192]
    # batch_sizes = [16, 32, 256, 512, 1024, 16384, 32768]
    batch_sizes = [128]
    # batch_sizes = [8192]
    SCRIPT_DIR_ON_SCRATCH = os.path.dirname(__file__)
    for m in models:
        for w in weights:
            for b in batch_sizes:
                for t in threads:
                    args.threads = t
                    # pattern: _korali_result/model/lat10/timepoint
                    jname = f"Model_{m}_WeightExp_{w}_BS_{b}_Threads_{t}"
                    args.path = os.path.join("_korali_result", "daint", args.other, f"model_{m}", f"BS_{b}", f"WeightExp_{w}", f"Threads_{t}")
                    job_output_path = args.path
                    mkdir_p(args.path)
                    jfile = os.path.join(args.path, f"{jname}.job")
                    # CREATE SBATCH FILES ======================================================================
                    with open(jfile, "w+") as fh:
                        # Set CLUSTER CONFIGURARTIONS ==========================================================
                        fh.writelines("#!/bin/bash\n")
                        # To run EXECUTABLE from this directory no matter from where the job file is submitted
                        fh.writelines(f"#SBATCH --chdir={SCRIPT_DIR_ON_SCRATCH}\n")
                        fh.writelines(f"#SBATCH --job-name={jname}.job\n")
                        fh.writelines(f"#SBATCH --output={os.path.join(job_output_path, jname)}.out\n")
                        fh.writelines(f"#SBATCH --nodes={args.nodes}\n")
                        fh.writelines(f"#SBATCH --time={args.job_duration}\n")
                        # if args.ntasks_per_node:
                        #     fh.writelines(f"#SBATCH --ntasks-per-node={args.ntasks_per_node}\n")
                        #     del args.ntasks
                        # else:
                        #     del args.ntasks_per_node
                        #     fh.writelines(f"#SBATCH --ntasks={args.ntasks}\n")
                        # fh.writelines(f"#SBATCH --cpus-per-task={args.cpus_per_task}\n")
                        # fh.writelines(f"#SBATCH --partition={args.partition}\n")
                        # fh.writelines("#SBATCH --mem=12000\n")
                        # fh.writelines("#SBATCH --mail-type=ALL\n")
                        fh.writelines("#SBATCH --account eth2\n")
                        # fh.writelines("#SBATCH --constraint gpu\n")
                        fh.writelines("#SBATCH --constraint mc\n")
                        fh.writelines(f"export OMP_NUM_THREADS={t}\n")
                        # Set python run commands ===============================================================
                        command = (
                            f"srun python {EXECUTABLE}"
                            f" --model {m}"
                            f" --weight-dim {w}"
                            f" --trainingBS {b}"
                            f" --engine {args.engine}"
                            f" --epochs {args.epochs}"
                            f" --optimizer {args.optimizer}"
                            f" --initialLearningRate {args.initialLearningRate}"
                            f" --trainingSetSize {args.trainingSetSize}"
                            f" --verbosity {args.verbosity}"
                            f" --path {args.path}"
                            # f" --other {args.other}"
                            # f" --batch-concurrency {args.batch_concurrency}"
                            # f" --conduit {args.conduit}"
                            # f" --frequency {args.frequency}"
                            # f" --result-file {args.result_file}"
                        )
                        command += f" --save"
                        if args.load_model:
                            command += f" --load-model"
                        if args.submit:
                            command += f" --submit"
                        if args.plot:
                            command += f" --plot"
                        fh.writelines(command)
                    # SUBMIT JOBS ===============================================================
                    if args.submit:
                        print(f"Submitting job {jname}")
                        os.system(f"sbatch {jfile}")
