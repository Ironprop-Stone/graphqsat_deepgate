import gym, minisat

import argparse
import time
import torch
import os
import numpy as np

def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_to_best_eval_path", type=str, default="./model/best_eval.pkl")

    parser.add_argument("--save_freq", type=int, default=500)

    parser.add_argument("--eps_init", type=float, default=1.0)
    parser.add_argument("--eps_final", type=float, default=0.01)
    parser.add_argument("--init_exploration_steps", type=int, default=5000)
    parser.add_argument("--eps_decay_steps", type=int, default=30000)

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--eval_separately_on_each", dest="eval_separately_on_each", action="store_true")
    parser.add_argument("--no_eval_separately_on_each", dest="eval_separately_on_each", action="store_false")
    parser.set_defaults(eval_separately_on_each=True)

    parser.add_argument("--eval_problems_paths", default='./aigdata/eval-problems-paths', type=str)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--test_time_max_decisions_allowed", default=500, type=int)

    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discounting")
    parser.add_argument("--lr_scheduler_frequency", default=3000, type=int)
    parser.add_argument("--lr_scheduler_gamma", default=1.0, type=float)

    parser.add_argument("--step_freq", default=4, type=int)
    parser.add_argument("--batch_size", default=64, type=int)

    parser.add_argument("--penalty_size", default=0.1, type=float)

    parser.add_argument("--aig_dir", default='./aigdata/train', type=str)
    parser.add_argument("--cnf_dir", default='./cnf', type=str)
    parser.add_argument("--tmp_dir", default='./tmp', type=str)
    
    parser.add_argument("--target-update-freq", type=int, default=10, help="How often to copy the parameters to traget.")

    parser.add_argument("--with_restarts", action="store_true", dest="with_restarts")
    parser.add_argument("--no_restarts", action="store_false", dest="with_restarts")
    parser.set_defaults(with_restarts=False)

    parser.add_argument("--compare_with_restarts", action="store_true", dest="compare_with_restarts")
    parser.add_argument( "--compare_no_restarts", action="store_false", dest="compare_with_restarts")
    parser.set_defaults(compare_with_restarts=False)

    parser.add_argument("--train_max_time_decisions_allowed", type=int, default=500)

    parser.add_argument("--buffer-size", type=int, default=10000)

    parser.add_argument("--env_name", type=str, default="sat-v0")

    parser.add_argument("--max_cap_fill_buffer", dest="max_cap_fill_buffer", action="store_true")

    parser.add_argument("--no_max_cap_fill_buffer", dest="max_cap_fill_buffer",action="store_false")

    parser.set_defaults(max_cap_fill_buffer=False)

    parser.add_argument("--train-problems-paths", type=str, default="./aigdata/train")

    parser.add_argument("--eval-problems-paths", type=str, default="./aigdata/eval-problems-paths")

    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--grad_clip_norm_type", type=float, default=2)

    parser.add_argument("--batch_updates", type=int, default=50000)

    parser.add_argument("--history_len", type=int, default=1)

    parser.add_argument("--no_cuda", action="store_true", help="Use the cpu/gpu")

    parser.add_argument("--input_type", type=str, default="ckt")

    return parser

def build_eval_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--core-steps",
        type=int,
        help="Number of message passing iterations. "
        "\-1 for the same number as used for training",
        default=-1,
    )
    
    parser.add_argument(
        "--eval-time-limit",
        type=int,
        help="Time limit for evaluation. If it takes more, return what it has and quit eval. In seconds.",
    )
    
    parser.add_argument(
        "--with_restarts",
        action="store_true",
        help="Do restarts in Minisat if set",
        dest="with_restarts",
    )
    parser.add_argument(
        "--no_restarts",
        action="store_false",
        help="Do not do restarts in Minisat if set",
        dest="with_restarts",
    )
    parser.set_defaults(with_restarts=False)

    parser.add_argument(
        "--compare_with_restarts",
        action="store_true",
        help="Compare to MiniSAT with restarts",
        dest="compare_with_restarts",
    )
    parser.add_argument(
        "--compare_no_restarts",
        action="store_false",
        help="Compare to MiniSAT without restarts",
        dest="compare_with_restarts",
    )
    parser.set_defaults(compare_with_restarts=False)
    parser.add_argument(
        "--test_max_data_limit_per_set",
        type=int,
        help="Max number of problems to load from the dataset for the env. EVAL/TEST mode.",
        default=None,
    )

    parser.add_argument(
        "--test_time_max_decisions_allowed",
        type=int,
        help="Number of steps the agent will act from the beginning of the episode when evaluating. "
        "Otherwise it will return -1 asking minisat to make a decision. "
        "Float because I want infinity by default (no minisat at all)",
    )
    parser.add_argument("--env-name", type=str, default="sat-v0", help="Environment.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Modify the flow of the script, i.e. run for less iterations",
    )

    parser.add_argument(
        "--model-dir",
        help="Path to the folder with checkpoints and model.yaml file",
        type=str,
    )
    parser.add_argument(
        "--model-checkpoint",
        help="Filename of the checkpoint, relative to the --model-dir param.",
        type=str,
    )
    parser.add_argument("--logdir", type=str, help="Dir for writing the logs")
    parser.add_argument(
        "--eps-final", type=float, default=0.1, help="Final epsilon value."
    )
    parser.add_argument(
        "--eval-problems-paths",
        help="Path to the problem dataset for evaluation",
        type=str,
    )
    parser.add_argument(
        "--train_max_data_limit_per_set",
        type=int,
        help="Max number of problems to load from the dataset for the env. TRAIN mode.",
        default=None,
    )
    parser.add_argument("--no-cuda", action="store_false", help="Use the cpu")

    parser.add_argument(
        "--dump_timings_path",
        type=str,
        help="If not empty, defines the directory to save the wallclock time performance",
    )
    
    return parser
    

def make_env(problems_paths, args, test_mode = False):
    max_data_limit_per_set = None

    if test_mode and hasattr(args, "test_max_data_limit_per_set"):
        max_data_limit_per_set = args.test_max_data_limit_per_set
    if not test_mode and hasattr(args, "train_max_data_limit_per_set"):
        max_data_limit_per_set = args.train_max_data_limit_per_set

    return gym.make(
        args.env_name,
        problems_paths=problems_paths,
        args=args,
        test_mode=test_mode,
        max_cap_fill_buffer=False if test_mode else args.max_cap_fill_buffer,
        penalty_size=args.penalty_size if hasattr(args, "penalty_size") else None,
        with_restarts=args.with_restarts if hasattr(args, "with_restarts") else None,
        compare_with_restarts=args.compare_with_restarts
        if hasattr(args, "compare_with_restarts")
        else None,
        max_data_limit_per_set=max_data_limit_per_set,
    )

def evaluate(agent, args, include_train_set=False):
    agent.net.eval()
    problem_sets = (
        [args.eval_problems_paths]
        if not args.eval_separately_on_each
        else [k for k in args.eval_problems_paths.split(":")]
    )
    if include_train_set:
        problem_sets.extend(
            [args.train_problems_paths]
            if not args.eval_separately_on_each
            else [k for k in args.train_problems_paths.split(":")]
        )
    
    res = {}

    st_time = time.time()
    print("Starting evaluation.")

    total_iters_ours = 0
    total_iters_minisat = 0

    for pset in problem_sets:
        eval_env = make_env(pset, args, test_mode=True)
        DEBUG_ROLLOUTS = None
        pr = 0
        walltime = {}
        propagations = {}
        scores = {}
        with torch.no_grad():
            while eval_env.test_to != 0 or pr == 0:
                p_st_time = time.time()
                obs = eval_env.reset(max_decisions_cap=args.test_time_max_decisions_allowed)
                done = eval_env.isSolved

                while not done:
                    action = agent.act([obs], eps=0)
                    obs, _, done, _ = eval_env.new_step(action)
                
                if eval_env.aig_problem != None:
                    walltime[eval_env.aig_problem] = time.time() - p_st_time
                    propagations[eval_env.aig_problem] = int(eval_env.S.getPropagations() / eval_env.step_ctr)
                    
                    print(
                        f"It took {walltime[eval_env.aig_problem]} seconds to solve problem {eval_env.aig_problem}."
                    )

                    sctr = 1 if eval_env.step_ctr == 0 else eval_env.step_ctr
                    ns = eval_env.normalized_score(sctr, eval_env.aig_problem)
                    print(f"Evaluation episode {pr+1} is over. Your score is {ns}.")
                    total_iters_ours += sctr
                    pdir, pname = os.path.split(eval_env.aig_problem)
                    total_iters_minisat += eval_env.metadata[pdir][pname][1]
                    scores[eval_env.aig_problem] = ns
                else:
                    walltime[eval_env.curr_problem] = time.time() - p_st_time
                    propagations[eval_env.curr_problem] = int(eval_env.S.getPropagations() / eval_env.step_ctr)

                    sctr = 1 if eval_env.step_ctr == 0 else eval_env.step_ctr
                    ns = eval_env.normalized_score(sctr, eval_env.curr_problem)
                    print(f"Evaluation episode {pr+1} is over. Your score is {ns}.")
                    total_iters_ours += sctr
                    pdir, pname = os.path.split(eval_env.curr_problem)
                    total_iters_minisat += eval_env.metadata[pdir][pname][1]
                    scores[eval_env.curr_problem] = ns
                    
                pr += 1
                if DEBUG_ROLLOUTS is not None and pr >= DEBUG_ROLLOUTS:
                    break
        
        print(
            f"Evaluation is done. Median relative score: {np.nanmedian([el for el in scores.values()]):.2f}, "
            f"mean relative score: {np.mean([el for el in scores.values()]):.2f}, "
            f"iters frac: {total_iters_minisat/total_iters_ours:.2f}"
        )
        res[pset] = scores

        agent.net.train()

        return (
            res,
            {
                "metadata": eval_env.metadata,
                "iters_frac": total_iters_minisat / total_iters_ours,
                "mean_score": np.mean([el for el in scores.values()]),
                "median_score": np.median([el for el in scores.values()])
            },
            False,
        )

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])