#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

from functional import seq
import attacksim

if __name__ == '__main__':
    for retfile in sys.argv[1:]:
        ret = attacksim.load_attack_sim_result(retfile)

        ntotal_exps = len(ret.retcoll_map)

        consq_cnts_df = (
            seq(ret.retcoll_map.values())
            .map(lambda x: x[-1].consequence.name)
            .count_by_value()
            .to_pandas(['consequence', 'count'])
            .sort_values('consequence')
            .assign(pct=lambda x: x['count'] / ntotal_exps * 100)
        )

        succ_rets = (
            seq(ret.retcoll_map.values())
            .filter(lambda x: x[-1].consequence == attacksim.AttackConsequence.SUCCESS)
            .map(lambda x: x[-1])
            .to_list()
        )

        orig_acc = next(iter(ret.retcoll_map.values()))[0].correct_pct
        succ_accs = [x.correct_pct for x in succ_rets]
        succ_acc_avg_str = f'{sum(succ_accs) / len(succ_accs):.2f}%' if succ_accs else '-'

        succ_nflips = [x.n_flips for x in succ_rets]
        succ_nflips_avg_str = f'{sum(succ_nflips) / len(succ_nflips):.2f}' if succ_nflips else '-'

        print(f'File: {retfile}')
        print(f'Total experiments: {ntotal_exps}')
        print(f'Acc: {orig_acc:.2f}% -> {succ_acc_avg_str}')
        print(f'Avg. flips: {succ_nflips_avg_str}')
        print(consq_cnts_df)
        print('-----')
