import numpy as np
import gym
import random
from os import listdir
from os.path import join, realpath, split
from .GymSolver import GymSolver
import sys

import glob
import os
import torch

from utils.aiger_utils import xdata_to_cnf
from utils.circuit_utils import get_fanin_fanout
from utils.cnf_utils import save_cnf
import deepgate as dg
import copy

MINISAT_DECISION_CONSTANT = 32767
VAR_ID_IDX = (
    0  # put 1 at the position of this index to indicate that the node is a variable
)

TEST_CASE_LIST = ['uf50-0278']

class gym_sat_Env(gym.Env):
    def __init__(
        self,
        problems_paths,
        args,
        test_mode=False,
        max_cap_fill_buffer=True,
        penalty_size=None,
        with_restarts=None,
        compare_with_restarts=None,
        max_data_limit_per_set=None,
    ):
        # Debug
        self.run_cnt = 0
        
        
        self.problems_paths = [realpath(el) for el in problems_paths.split(":")]
        self.args = args
        self.test_mode = test_mode

        self.max_data_limit_per_set = max_data_limit_per_set
        # pre_test_files = [
        #     [join(dir, f) for f in listdir(dir) if f.endswith(".cnf")]
        #     for dir in self.problems_paths
        # ]
        # if self.max_data_limit_per_set is not None:
        #     pre_test_files = [
        #         np.random.choice(el, size=max_data_limit_per_set, replace=False)
        #         for el in pre_test_files
        #     ]
        # self.test_files = [sl for el in pre_test_files for sl in el]

        self.test_files = []
        if self.args.input_type == 'ckt':
            all_files = glob.glob(os.path.join(self.args.aig_dir, '*.aiger'))
        else:
            all_files = glob.glob(os.path.join(self.args.cnf_dir, '*.cnf'))

        if test_mode == True:
            all_files = glob.glob(os.path.join(self.problems_paths[0], '*.aiger'))

        for problem_path in all_files:
            problem_name = os.path.basename(problem_path).split('.')[0]
            self.test_files.append(problem_path)
            # if problem_name in TEST_CASE_LIST:
            #     self.test_files.append(problem_path)

        self.metadata = {}
        self.max_decisions_cap = float("inf")
        self.max_cap_fill_buffer = max_cap_fill_buffer
        self.penalty_size = penalty_size if penalty_size is not None else 0.0001
        self.with_restarts = True if with_restarts is None else with_restarts
        self.compare_with_restarts = (
            False if compare_with_restarts is None else compare_with_restarts
        )

        try:
            for dir in self.problems_paths:
                self.metadata[dir] = {}
                with open(join(dir, "METADATA")) as f:
                    for l in f:
                        k, rscore, msscore = l.split(",")
                        self.metadata[dir][k] = [int(rscore), int(msscore)]
        except Exception as e:
            print(e)
            print("No metadata available, that is fine for metadata generator.")
            self.metadata = None
        self.test_file_num = len(self.test_files)
        self.test_to = 0

        self.step_ctr = 0
        self.curr_problem = None
        
        self.aig_problem = None

        self.global_in_size = 1
        self.vertex_in_size = 2
        self.edge_in_size = 2
        self.max_clause_len = 0

        if self.args.input_type == 'ckt':
            self.aig_parser = dg.AigParser()

        self.aig = None
        self.first_round = True

    def new_parse_state_as_graph(self):
        (
            total_var,
            _,
            current_depth,
            n_init_clauses,
            num_restarts,
            _,
        ) = self.S.getMetadata()

        var_assignments = self.S.getAssignments()
        clauses = self.S.getClauses()
        clause_vars = sum(clauses, [])
        
        # 保存clauses中存在的所有变量
        clause_valid_decisions = []
        for el in clause_vars:
            if(el > 0):
                clause_valid_decisions.append(int((el - 1) * 2))
            elif(el < 0):
                clause_valid_decisions.append(int((abs(el) - 1) * 2 + 1))
        
        clause_valid_decisions = sorted(list(set(clause_valid_decisions)))

        clause_valid_vars = sorted(list(set([int(el / 2) for el in clause_valid_decisions])))

        # 问题：var_assignments中有些变量值为2，但是不在clauses的列表元素中，说明可能已经被消去了，我们要去掉这些元素
        # 根据clauses里真正存在的变量确定变量个数num_var、valid_decisions、valid_vars、valid_remapping等变量

        valid_decisions = [
            el
            for i in range(len(var_assignments))
            for el in (2 * i, 2 * i + 1)
            if var_assignments[i] == 2 and i in clause_valid_vars
        ]
        valid_vars = [
            idx for idx in range(len(var_assignments)) if var_assignments[idx] == 2 and idx in clause_valid_vars
        ]
        
        # 告诉我们原始的variable到现在solver的variable的映射关系
        # 如果remapping里面有的variable就是决策空间，如果没有的直接不管
        vars_remapping = {el: i for i, el in enumerate(valid_vars)}
        self.decision_to_var_mapping = {
            i: val_decision for i, val_decision in enumerate(valid_decisions)
        }

        # 利用已有变量构造observation—用掩码的形式存储哪些PI或者gate需要进行赋值
        self.aig.valid_mask = valid_vars
        self.aig.valid_decisions = valid_decisions
        # Variable state: -1 - not vaild, 0 - assign to negative, 1 - assign to positive, 2 - not assigned
        self.aig.var_state = []
        for idx in range(len(var_assignments)):
            if var_assignments[idx] == 0:
                self.aig.var_state.append(0)
            elif var_assignments[idx] == 1:
                self.aig.var_state.append(1)
            else:
                if idx in valid_vars:
                    self.aig.var_state.append(2)
                else:
                    self.aig.var_state.append(-1)
        self.aig.var_state = torch.tensor(self.aig.var_state, dtype=torch.int8)

        if self.S.getDone():
            return copy.deepcopy(self.aig), True

        return copy.deepcopy(self.aig), False

    def random_pick_satProb(self):
        if self.test_mode:  # in the test mode, just iterate all test files in order
            filename = self.test_files[self.test_to]
            self.test_to += 1
            if self.test_to >= self.test_file_num:
                self.test_to = 0
            return filename
        else:  # not in test mode, return a random file in "uf20-91" folder.
            return self.test_files[random.randint(0, self.test_file_num - 1)]

    def reset(self, max_decisions_cap=None):
        self.step_ctr = 0

        if max_decisions_cap is None:
            max_decisions_cap = sys.maxsize
        self.max_decisions_cap = max_decisions_cap
        self.curr_problem = self.random_pick_satProb()

        # 增加代码，将circuit转回cnf，即MiniSAT Solver能够求解的形式
        if self.args.input_type == 'ckt':
            problem_name = os.path.basename(self.curr_problem).split('.')[0]
            self.aig = self.aig_parser.read_aiger(self.curr_problem)
            self.aig.problem_path = self.curr_problem
            self.aig_problem = self.curr_problem

            edge_index = np.transpose(self.aig.edge_index)
            fanin_list, fanout_list = get_fanin_fanout(self.aig.x, edge_index)
            assert len(self.aig.POs) == 1
            x_data = []
            for idx in range(len(self.aig.x)):
                if self.aig.x[idx][0] == 1:
                    gate_type = 0
                elif self.aig.x[idx][1] == 1:
                    gate_type = 1
                else:
                    gate_type = 2
                x_data.append([idx, gate_type])
            cnf = xdata_to_cnf(x_data, fanin_list, const_1=self.aig.POs)
            cnf_path = os.path.join(self.args.tmp_dir, '{}.cnf'.format(problem_name))
            save_cnf(cnf, len(x_data), cnf_path)
            self.curr_problem = cnf_path

        self.S = GymSolver(self.curr_problem, self.with_restarts, max_decisions_cap)
        self.max_clause_len = 0

        self.curr_state, self.isSolved = self.new_parse_state_as_graph()

        return self.curr_state
    
    def new_step(self, decision, dummy=False):
        if decision >= 0:
            decision = self.decision_to_var_mapping[decision]
        self.step_ctr += 1

        if dummy:
            self.S.step(MINISAT_DECISION_CONSTANT)
            (
                num_var,
                _,
                current_depth,
                n_init_clauses,
                num_restarts,
                _,
            ) = self.S.getMetadata()
            return (
                None,
                None,
                self.S.getDone(),
                {
                    "curr_problem": self.curr_problem,
                    "num_restarts": num_restarts,
                    "max_clause_len": self.max_clause_len,
                },
            )
        
        old_clauses = self.S.getClauses()
        
        if self.step_ctr > self.max_decisions_cap:
            while not self.S.getDone():
                self.S.step(MINISAT_DECISION_CONSTANT)
                if self.max_cap_fill_buffer:
                    break
                self.step_ctr += 1
            else:
                self.step_ctr -= 1
        else:
            if decision < 0:
                decision = MINISAT_DECISION_CONSTANT
            elif (decision % 2 == 0):  
                decision = int(decision / 2 + 1)
            else:
                decision = 0 - int(decision / 2 + 1)
            self.S.step(decision)
        
        self.curr_state, self.isSolved = self.new_parse_state_as_graph()

        new_clauses = self.S.getClauses()

        (
            num_var,
            _,
            current_depth,
            n_init_clauses,
            num_restarts,
            _,
        ) = self.S.getMetadata()

        if self.step_ctr > self.max_decisions_cap and not self.max_cap_fill_buffer:
            step_reward = -self.penalty_size
        else:
            step_reward = 0 if self.isSolved else -self.penalty_size

        return (
            self.curr_state,
            step_reward,
            self.isSolved,
            {
                "curr_problem": self.curr_problem,
                "num_restarts": num_restarts,
                "max_clause_len": self.max_clause_len,
            },
        )

    def normalized_score(self, steps, problem):
        pdir, pname = split(problem)
        no_restart_steps, restart_steps = self.metadata[pdir][pname]
        if self.compare_with_restarts:
            return restart_steps / steps
        else:
            return no_restart_steps / steps