"""
SQL based Reasoning
"""

import time
import json
import argparse
import copy
import os
import regex as re

from typing import List
import platform
import multiprocessing

import sys

# 현재 파일 위치 기준으로 H-STAR 디렉토리를 sys.path에 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)  # generation을 찾기 위해 H-STAR 경로 추가

from generation.generator_gpt import Generator
from utils.utils import load_data_split
from nsql.database import NeuralDB

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../..")

def worker_annotate(
        pid: int,
        args,
        generator: Generator,
        g_eids: List,
        data,
        tokenizer
):
    """
    A worker process for annotating.
    """
    g_dict = dict()
    built_few_shot_prompts = []

    pattern_row = '(f_row\(\[(.*?)\]\))'
    pattern_row_num = '\d+'
    pattern_row = re.compile(pattern_row, re.S)
    pattern_row_num = re.compile(pattern_row_num, re.S)

    # Extract Rows
    for g_eid in g_eids:
        try:
            preds=[]
            for n in range(2):
                try:
                    pred = re.findall(pattern_row,data[str(g_eid)]['output'][n])[0][1]
                    if pred == '*':
                        pred = ''
                        for i in range(len(data[str(g_eid)]['data_item']['table']['rows'])):
                            pred += f'row {i}'
                            if i != len(data[str(g_eid)]['data_item']['table']['rows'])-1:
                                pred += ', '

                        pred = pred.split(', ')
                        preds.append(pred)
                    else:
                        pred = pred.split(', ')
                        preds.append(pred)
                except:
                    pred = data[str(g_eid)]['rows'] 
                    if pred == [] or '':
                        pred = ''
                        for i in range(len(data[str(g_eid)]['data_item']['table']['rows'])):
                            pred += f'row {i}'
                            if i != len(data[str(g_eid)]['data_item']['table']['rows'])-1:
                                pred += ', '
                        pred = pred.split(', ')
                    preds.append(pred)

            pred = list(set().union(*preds, data[str(g_eid)]['rows']))

            g_data_item = data[str(g_eid)]['data_item']
            g_dict[g_eid] = {
                'generations': [],
                'cols' : [],
                'rows' : [],
                'ori_data_item': copy.deepcopy(g_data_item)
            }
            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}]
            )
            g_data_item['table'] = db.get_table_df()
            g_data_item['title'] = db.get_table_title()

            df = g_data_item['table']

            # Filter Columns
            filtered_cols = [value for value in g_data_item['table'].columns if value in data[str(g_eid)]['cols']]
            if filtered_cols == []:
                filtered_cols = [value for value in g_data_item['table'].columns]
            g_dict[g_eid]['cols'] = filtered_cols

            if [data[str(g_eid)]['cols']] != []:
                if 'row_id' not in filtered_cols:
                    filtered_cols.insert(0, 'row_id')
                df = g_data_item['table'][filtered_cols]
            
            if pred != [""]:
                row_list = [str(pattern_row_num.search(x).group()) for x in pred if pattern_row_num.search(x)]
                if row_list == []:
                    row_list = [str(j) for j in range(len(g_data_item['table']['rows']))]
                    row_unique = set(row_list)
                    row_list = list(row_unique)
                else:
                    row_unique = set(row_list)
                    row_list = list(row_unique)
                    try:
                        indices = [eval(i) for i in row_list]
                        if any(index >= len(df) for index in indices):
                            raise IndexError("Index out of bounds")
                        df = df[df.index.isin(indices)]
                    except IndexError:
                        df = df

            if pred == [] or pred == [""]:
                row_list = [str(j) for j in range(len(g_data_item['table']['rows']))]
                row_unique = set(row_list)
                row_list = list(row_unique)
                
                df = df[df.index.isin([eval(i) for i in row_list])]
            
            g_dict[g_eid]['rows'] = row_list
            g_data_item['table'] = df

            n_shots = args.n_shots
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=args.prompt_file,
                n_shots=n_shots
            )
            generate_prompt = generator.build_generate_prompt(
                data_item=g_data_item,
                generate_type=(args.generate_type,)
            )
            prompt = few_shot_prompt + "\n\n" + generate_prompt
            
            print(generate_prompt)
            # Ensure the input length fit max input tokens by shrinking the number of rows
            max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
            num_rows = (g_data_item['table'].shape[0])

            while len(tokenizer.tokenize(prompt)) >= max_prompt_tokens:
                num_rows = 5
                generate_prompt = generator.build_generate_prompt(
                data_item=g_data_item,
                generate_type=(args.generate_type,),
                num_rows = num_rows
                )

                prompt = few_shot_prompt + "\n\n" + generate_prompt
            print(f"Process#{pid}: Building prompt for eid#{g_eid}, original_id#{g_data_item['id']}")
            built_few_shot_prompts.append((g_eid, prompt))
            if len(built_few_shot_prompts) < args.n_parallel_prompts:
                continue

            print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run openai API.")
            response_dict = generator.generate_one_pass(
                prompts=built_few_shot_prompts,
                verbose=args.verbose
            )
            for eid, g_pairs in response_dict.items():
                g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
                g_dict[eid]['generations'] = g_pairs
            
            built_few_shot_prompts = []
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Process#{pid}: eid#{g_eid}, wtqid#{g_data_item['id']} generation error: {e}")

    # Final generation inference
    if len(built_few_shot_prompts) > 0:
        response_dict = generator.generate_one_pass(
            prompts=built_few_shot_prompts,
            verbose=args.verbose
        )
        for eid, g_pairs in response_dict.items():
            g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
            g_dict[eid]['generations'] = g_pairs
    
    return g_dict


def main():
    # Build paths
    args.api_keys_file = os.path.join(ROOT_DIR, args.api_keys_file)
    args.prompt_file = os.path.join(ROOT_DIR, args.prompt_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)

    # For TabFact test split, we load the small test set (about 2k examples) to test,
    # since it is expensive to test on full set
    if args.dataset == "tab_fact" and args.dataset_split == "test":
        dataset = []
        with open(os.path.join(ROOT_DIR, "utils", "tab_fact", "small_test.jsonl"), "r") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                dic = json.loads(line)
                id = dic['table_id']
                caption = dic['table_caption']
                question = dic['statement']
                answer_text = dic['label']
                header = dic['table_text'][0]
                rows = dic['table_text'][1:]
                
                data = {
                    "id": i,
                    "table": {
                        "id": id,
                        "header": header,
                        "rows": rows,
                        "page_title": caption
                    },
                    "question": question,
                    "answer_text": answer_text
                }
                dataset.append(data)
                
    if args.dataset == "fetaqa" and args.dataset_split == "test":
        dataset = []
        with open(os.path.join(ROOT_DIR, "utils", "fetaqa", "fetaQA-v1_test.jsonl"), "r") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                dic = json.loads(line)
                # print(dic)
                feta_id = dic['feta_id']
                caption = dic['table_page_title']
                question = dic['question']
                sub_title = dic['table_section_title']
                answer = dic["answer"]
                header = dic['table_array'][0]
                rows = dic['table_array'][1:]
                data = {
                "id": feta_id,
                "table": {
                    "id": feta_id,
                    "header": header,
                    "rows": rows,
                    "page_title": caption,
                    "sub_title": sub_title
                },
                "question": question,
                "answer": answer
                }
                dataset.append(data)

    # Load openai keys
    with open(args.api_keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]
    
    with open(os.path.join(args.save_dir, args.input_program_file), 'r') as f:
        data = json.load(f)
    data_dict = dict()
    for eid, _ in data.items():
        data_item = data[eid]['ori_data_item']
        if data[eid]['generations'] or data[eid]['cols']:
            output = data[eid]['generations']
            rows = data[eid]['rows']
            cols = data[eid]['cols']
        else:
            rows = []
            cols = []
        data_dict[eid] = {'rows': rows, 'cols': cols, 'output' : output, 'data_item' : data_item}
    
    # Split by processes
    data_dict_group = [dict() for _ in range(args.n_processes)]
    for idx, eid in enumerate(data_dict.keys()):
        data_dict_group[idx % args.n_processes][eid] = data_dict[eid]

    # Annotate
    generator = Generator(args, keys=keys)
    # Enter dataset size for inference: range(0, len(dataset))
    generate_eids = list(range(0,10))
    
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[int(g_eid) % args.n_processes].append(g_eid)
    print('\n******* Annotating *******')
    g_dict = dict()
    worker_results = []
    pool = multiprocessing.Pool(processes=args.n_processes)
    for pid in range(args.n_processes):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))
        worker_results.append(pool.apply_async(worker_annotate, args=(
            pid,
            args,
            generator,
            generate_eids_group[pid],
            data_dict_group[pid],
            tokenizer
        )))

    # Merge annotation results
    for r in worker_results:
        worker_g_dict = r.get()
        g_dict.update(worker_g_dict)
    pool.close()
    pool.join()

    # Save annotation results
    # "_".join(["{}={}".format(k, str(args.__dict__[k])) for k in args.__dict__ if k not in ['api_keys_file', 'prompt_file', 'save_dir', 'stop_tokens']])
    save_file_name = f'{args.dataset}_{args.dataset_split}_sql_reason.json'
    with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
        dict_keys = list(g_dict.keys())
        dict_keys.sort()
        sorted_g_dict = {i: g_dict[i] for i in dict_keys}
        json.dump(sorted_g_dict, f, indent=4)

    print(f"Elapsed time: {time.time() - start_time}")


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument('--dataset', type=str, default='wikitq',
                        choices=['wikitq', 'tab_fact', 'fetaqa'])
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--prompt_file', type=str, default='templates/prompts/wikitq_binder.txt')
    parser.add_argument('--input_program_file', type=str,
                        default='wikitq_test_col_select.json')
    parser.add_argument('--save_dir', type=str, default='results/model_gpt/')

    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=1)

    # Program generation options
    parser.add_argument('--prompt_style', type=str, default='create_table_select_full_table',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'text_full_table',
                                 'transpose',
                                 'create_table_select_3',
                                 'create_table',
                                 'no_table'])
    parser.add_argument('--generate_type', type=str, default='col',
                        choices=['col', 'answer', 'row', 'verification'])
    parser.add_argument('--n_shots', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    # LLM options
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--n_parallel_prompts', type=int, default=1)
    parser.add_argument('--max_generation_tokens', type=int, default=256)
    parser.add_argument('--max_api_total_tokens', type=int, default=14001)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--sampling_n', type=int, default=3)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--stop_tokens', type=str, default='\n\n',
                        help='Split stop tokens by ||')

    # debug options
    parser.add_argument('-v', '--verbose', action='store_false')

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main()