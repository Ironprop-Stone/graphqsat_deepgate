from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import deque


import deepgate
import torch
import os
import copy

from deepgatesat.utils import build_argparser, make_env, evaluate
from deepgatesat.agents import CircuitAgent
from deepgatesat.learners import CircuitLearner
from deepgatesat.buffer import CircuitBuffer

from deepgatesat.ckt_model import ckt_net
import pickle
import yaml
import numpy as np
import json

from tensorboardX import SummaryWriter


def save_training_state(
    model,
    learner,
    episodes_done,
    transitions_seen,
    best_eval_so_far,
    args,
    model_to_best_eval,
    in_eval_mode=False,
):
    # save the model
    model_path = os.path.join(args.logdir, f"model_{learner.step_ctr}.chkp")
    torch.save(model.state_dict(), model_path)

    # save the experience replay
    buffer_path = os.path.join(args.logdir, "buffer.pkl")

    with open(buffer_path, "wb") as f:
        pickle.dump(learner.buffer, f)

    # save important parameters
    train_status = {
        "step_ctr": learner.step_ctr,
        "latest_model_name": model_path,
        "buffer_path": buffer_path,
        "args": args,
        "episodes_done": episodes_done,
        "logdir": args.logdir,
        "transitions_seen": transitions_seen,
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "optimizer_class": type(learner.optimizer),
        "best_eval_so_far": best_eval_so_far,
        "scheduler_class": type(learner.lr_scheduler),
        "scheduler_state_dict": learner.lr_scheduler.state_dict(),
        "in_eval_mode": in_eval_mode,
    }
    
    
    if not learner.step_ctr % args.eval_freq: 
        model_to_best_eval[learner.step_ctr] = best_eval_so_far
        with open(args.model_to_best_eval_path, "wb") as f:
            pickle.dump(model_to_best_eval, f)
    
    status_path = os.path.join(args.logdir, "status.yaml")

    with open(status_path, "w") as f:
        yaml.dump(train_status, f, default_flow_style=False)

    return status_path

def get_annealed_eps(n_trans, args):
    if n_trans < args.init_exploration_steps:
        return args.eps_init
    if n_trans > args.eps_decay_steps:
        return args.eps_final
    else:
        assert n_trans - args.init_exploration_steps >= 0
        return (args.eps_init - args.eps_final) * (
            1 - (n_trans - args.init_exploration_steps) / args.eps_decay_steps
        ) + args.eps_final

if __name__ == '__main__':
    # 初始化命令行参数
    parser = build_argparser()
    args = parser.parse_args()
    args.device = (
        torch.device("cpu")
        if args.no_cuda or not torch.cuda.is_available()
        else torch.device("cuda")
    )
    print('Device: ', args.device)

    writer = SummaryWriter()
    args.logdir = writer.logdir

    model_to_best_eval = {}

    model_save_path = os.path.join(args.logdir, "model.yaml")
    best_eval_so_far = (
        {args.eval_problems_paths: -1}
        if not args.eval_separately_on_each
        else {k: -1 for k in args.eval_problems_paths.split(":")}
    )

    env = make_env(args.train_problems_paths, args, test_mode=False)

    net = ckt_net(args)
    
    # 输出网络模型结构
    print(str(net))

    target_net = copy.deepcopy(net)

    buffer = CircuitBuffer(args, args.buffer_size)

    agent = CircuitAgent(net, args)
    learner = CircuitLearner(net, target_net, buffer, args)

    n_trans = 0
    ep = 0
    
    # 输出命令行参数
    print(args.__str__())
    
    jsonlist = []
    
    batch_updates = 100000
    
    while learner.step_ctr < batch_updates:
        # 学习器的步数
        print('Step of learner: ', learner.step_ctr)
        
        # 每个episode的总奖励值
        ret = 0
        
        # 环境初始化训练数据
        obs = env.reset(args.train_max_time_decisions_allowed)
        
        # 每个episode的问题求解是否完成
        done = env.isSolved

        # 存储
        hist_buffer = deque(maxlen=args.history_len)
        for _ in range(args.history_len):
            hist_buffer.append(obs)

        # 每个episode的步数，即需要选择动作的次数
        ep_step = 0

        eval_resume_signal = False

        save_flag = False

        while not done:
            # 获取模拟退火系数annealed_eps，该系数逐渐递减
            annealed_eps = get_annealed_eps(n_trans, args)
            
            # 选取动作，获得后续状态
            action = agent.act(hist_buffer, annealed_eps)
            next_obs, reward, done, _ = env.new_step(action)
            buffer.add_transition(obs, action, reward, done)
            obs = next_obs
            hist_buffer.append(obs)
            
            # 返回值ret增加环境的奖励值reward，reward都为-0.1，若求解一个问题结束，reward为0
            # 返回值ret越小，说明环境求解SAT问题越快，效果越好
            ret += reward

            # 首先要满足buffer元素数量大于init_exploration_steps(5000)，每隔一定步数，即step_freq之后，对学习器的参数进行更新
            if (not n_trans % args.step_freq) and (buffer.ctr > max(args.init_exploration_steps, args.batch_size + 1) or buffer.full):
                # 进入学习器训练网络
                step_info = learner.step()
                
                # 保存相关参数到步骤信息step_info，step_info存入jsonlist，便于绘制参数变化
                if annealed_eps is not None: 
                    step_info["annealed_eps"] = annealed_eps
                    
                step_info['grad_norm'] = step_info['grad_norm'].cpu().numpy().item()
                step_info['average_q'] = step_info['average_q'].cpu().detach().numpy().item()
                
                jsonlist.append(step_info)
                
                # 学习器的步数每隔eval_freq之后进行案例测试
                if (not learner.step_ctr % args.eval_freq) or eval_resume_signal:
                    scores, _, eval_resume_signal = evaluate(agent, args, include_train_set=False)
                    for sc_key, sc_val in scores.items():
                        if len(sc_val) > 0:
                            res_vals = [el for el in sc_val.values()]
                            median_score = np.nanmedian(res_vals)
                            if best_eval_so_far[sc_key] < median_score or best_eval_so_far[sc_key] == -1:
                                best_eval_so_far[sc_key] = median_score
                                
                            # 在writer中记录得分
                            writer.add_scalar(f"data/median relative score: {sc_key}", np.nanmedian(res_vals), learner.step_ctr - 1)
                            writer.add_scalar(f"data/mean relative score: {sc_key}", np.nanmean(res_vals), learner.step_ctr - 1)
                            writer.add_scalar(f"data/max relative score: {sc_key}", np.nanmax(res_vals), learner.step_ctr - 1)
                    
                    # 在writer中记录测试案例的最优得分
                    for k, v in best_eval_so_far.items():
                        writer.add_scalar(k, v, learner.step_ctr - 1)
                        
                # 在writer中记录训练步骤的信息，包括loss值、grad_norm、学习率lr、平均Q值average_q
                for k, v in step_info.items():
                    writer.add_scalar(k, v, learner.step_ctr - 1)
                
                # 在writer中写入学习器的步数
                writer.add_scalar("data/num_episodes", ep, learner.step_ctr - 1)
                
                # 按照一定的训练步骤保存训练模型，save_freq设置为500
                if not learner.step_ctr % args.save_freq:
                    status_path = save_training_state(net, learner, ep - 1, n_trans, best_eval_so_far, args, model_to_best_eval, in_eval_mode=eval_resume_signal)
                    save_flag = True

            # 转移数加1，episode步数加1
            n_trans += 1
            ep_step += 1

        # 输出每个episode的返回值，该返回值越小，说明训练效果越好
        print(f"Episode {ep + 1}: Return {ret}.")

        # episode编号加1
        ep += 1

        # 在save_freq保存模型之后再次保存模型
        if save_flag:
            status_path = save_training_state(net, learner, ep - 1, n_trans, best_eval_so_far, args, model_to_best_eval, in_eval_mode=eval_resume_signal)
            save_flag = False
    
    # 训练结束后，绘制reward值与loss值的变化
    with open(os.path.join(os.getcwd(), 'info.json'), "w") as f:
        json.dump(jsonlist, f)
    del jsonlist