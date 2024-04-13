import re
import subprocess
from pathlib import Path


files = sorted(Path('./output/').glob('*.results'))
for file in files:
    run_name = file.stem
    qrels = '../data/qrels-and-related-files/test-ratings.qrels'
    binary_qrels = '../1/output/test-ratings.binary.qrels'
    output = subprocess.run(['../trec_eval-9.0.7/trec_eval', '-m', 'all_trec', '-c', qrels, file],
                            capture_output=True, text=True)
    binary_output = subprocess.run(['../trec_eval-9.0.7/trec_eval', '-m', 'all_trec', binary_qrels, file],
                                   capture_output=True, text=True)

    if output.returncode == 0 and binary_output.returncode == 0:
        ndcg_10 = re.findall(r'\nndcg_cut_10[ \t]+all[ \t]+([0-9.]+)', output.stdout)
        ndcg_100 = re.findall(r'\nndcg_cut_100[ \t]+all[ \t]+([0-9.]+)', output.stdout)
        p_10 = re.findall(r'\nP_10[ \t]+all[ \t]+([0-9.]+)', binary_output.stdout)
        map = re.findall(r'\nmap[ \t]+all[ \t]+([0-9.]+)', binary_output.stdout)

        if len(ndcg_10) == len(ndcg_100) == len(p_10) == len(map) == 1:
            print([run_name, float(ndcg_10[0]), float(ndcg_100[0]), float(p_10[0]), float(map[0])])
        else:
            print(f'[ERROR] Multiple matches, when processing {run_name}.')
    else:
        print(f'[ERROR] {output.stderr} and {binary_output.stderr}, when processing {run_name}.')
