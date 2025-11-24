import concurrent.futures
import glob
import os
from functools import partial
from itertools import product

import omegaconf

import argparse

# Global Variables
LR = [5e-1, 1e-1]
MASKING = ["hard_discrete"]
RE = [".*"]
L = [1, 0]
BETA = [3, 4, 5, 6, 7]
EPS = [1e-8, 1e-3, 1e-1, 1]
NUM_INTERVENTION = [50]
DEVICE = "cuda:0"
DEBUG = False
PROJECT_NAME = "sdxledit_vram_runtime"
CHECKPOINTING = True


def generate_configs(args):
    base_cfg = omegaconf.OmegaConf.load(args.config)
    for (
        alr,
        flr,
        nlr,
        masking,
        re,
        loss_reg_norm,
        loss_recons_norm,
        beta,
        num_intervention,
        eps,
        ds,
        c,
    ) in product(
        args.attn_learning_rate,
        args.ffn_learning_rate,
        args.n_learning_rate,
        args.masking,
        args.regex,
        args.loss_reg,
        args.loss_recons,
        args.beta,
        args.num_intervention,
        args.eps,
        args.data_size,
        args.concept,
    ):
        cfg = base_cfg
        cfg.data.size = ds
        cfg.trainer.attn_lr = alr
        cfg.trainer.n_lr = nlr
        cfg.trainer.ff_lr = flr
        cfg.trainer.masking = masking
        cfg.trainer.masking_eps = eps
        cfg.trainer.regex = re
        cfg.trainer.beta = beta
        cfg.trainer.device = f"cuda:{args.device}"
        cfg.trainer.attn_name = args.attn_name
        cfg.trainer.num_intervention_steps = num_intervention
        cfg.loss.reg = loss_reg_norm
        cfg.loss.reconstruct = loss_recons_norm
        cfg.logger.project = args.project_name
        cfg.trainer.grad_checkpointing = args.ncheckpointing
        cfg.debug = False
        cfg.trainer.epochs = args.max_epochs

        if args.concept is not None:
            cfg.data.concept = c
        cfg.data.filter_ratio = args.filter_ratio
        cfg.data.style = args.style
        cfg.data.with_fg_filter = args.with_fg_filter
        cfg.data.with_synonyms = args.with_syno

        # naming convention for ablation, epoch, ds, lr, eps, beta, latent range
        output_file = os.path.join(
            args.output_dir,
            f"concept_{c}_epoch{cfg.trainer.epochs}_ds{ds}_filter_ratio{args.filter_ratio}_"
            f"with_syno_{args.with_syno}_eps{eps}_beta{beta}_lr_{alr}{flr}{nlr}.yaml",
        )
        omegaconf.OmegaConf.save(cfg, output_file)


def run_experiment(config_file, task="general"):
    base_cmd = f'python ./scripts/train.py --cfg "{config_file}" --task {task}'
    os.system(base_cmd)


def parse_args():
    parser = argparse.ArgumentParser("Hyperparameter tuning, generate hyperparameters config files")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="gen for generating config files, run for running experiments",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/flux.yaml",
        help="Path to the basic config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="configs/param_tuning_vram_runtime_no_checkpoint",
        help="Path to the output directory",
    )
    parser.add_argument("--max_job", type=int, default=1, help="Max running job at a time")
    parser.add_argument("--data_size", "-ds", type=int, nargs="+", default=[40])
    parser.add_argument("--attn_learning_rate", "-alr", type=float, nargs="+", default=LR)
    parser.add_argument("--ffn_learning_rate", "-flr", type=float, nargs="+", default=LR)
    parser.add_argument("--n_learning_rate", "-nlr", type=float, nargs="+", default=[0])
    parser.add_argument("--masking", "-mask", type=str, nargs="+", default=MASKING)
    parser.add_argument("--regex", "-re", type=str, nargs="+", default=RE)
    parser.add_argument("--loss_reg", "-lreg", type=int, nargs="+", default=[1], help="Loss regularization")
    parser.add_argument("--loss_recons", "-lrec", type=int, nargs="+", default=[2], help="Loss reconstruction")
    parser.add_argument("--beta", "-b", type=float, nargs="+", default=BETA)
    parser.add_argument("--num_intervention", "-ni", type=int, nargs="+", default=NUM_INTERVENTION)
    parser.add_argument("--device", "-d", type=int, default=1)
    parser.add_argument("--attn_name", "-an", type=str, default="attn")
    parser.add_argument("--max_epochs", "-me", type=int, default=10)
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=EPS,
        help="Epsilon for hard-discrete masking",
    )
    parser.add_argument(
        "--train_task",
        "-ts",
        type=str,
        default="general",
        help="Training task type",
    )
    parser.add_argument("--project_name", "-pn", type=str, default=PROJECT_NAME)
    parser.add_argument("--ncheckpointing", "-ncp", action="store_false")
    parser.add_argument("--concept", type=str, default=None, nargs="+", help="concept for the prompt")
    parser.add_argument("--neutral_concept", type=str, default=None, help="neutral concept for the prompt")
    parser.add_argument("--filter_ratio", "-fr", type=float, default=1, help="filter ratio for the dataset")
    parser.add_argument("--style", type=str, default="concept", help="style (art style) or concept")
    parser.add_argument("--with_fg_filter", action="store_true", help="filter foreground")
    parser.add_argument("--with_syno", action="store_true", help="use synonym concept")
    return parser.parse_args()


def main(args):
    if args.task == "gen":
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        generate_configs(args)
    elif args.task == "run":
        cfg_path_list = glob.glob(os.path.join(args.output_dir, "*.yaml"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_job) as executor:
            func_run_list = [
                partial(run_experiment, config_file=cfg_path, task=args.train_task) for cfg_path in cfg_path_list
            ]
            futures = [executor.submit(func) for func in func_run_list]
            for future in concurrent.futures.as_completed(futures):
                print("Finished task", future)
        print("All tasks are done")
    else:
        raise ValueError("Invalid task, should be gen or run")


if __name__ == "__main__":
    args = parse_args()
    main(args)
