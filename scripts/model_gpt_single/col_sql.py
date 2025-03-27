"""
SQL based Column Extraction
"""

import time
import json
import argparse
import copy
import os

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

from transformers import AutoTokenizer

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../..")


def worker_annotate(
        pid: int,
        args,
        generator: Generator,
        g_eids: List,
        dataset,
        tokenizer
):
    """
    A worker process for annotating.
    """
    g_dict = dict()
    built_few_shot_prompts = []
    for g_eid in g_eids:
        try:
            g_data_item = dataset[g_eid]
            g_dict[g_eid] = {
                'generations': [],
                'ori_data_item': copy.deepcopy(g_data_item)
            }
            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}]
            )
            g_data_item['table'] = db.get_table_df()
            g_data_item['title'] = db.get_table_title()

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

    # # Final generation inference
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

    # ✅ 커스텀 입력 파일이 있으면 그걸 사용
    if hasattr(args, 'input_file') and args.input_file:
        input_path = os.path.join(ROOT_DIR, args.input_file)
        print(f"[DEBUG] 입력 파일에서 단일 샘플 로딩: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    else:
        # Load dataset (원래 흐름 유지)
        start_time = time.time()
        dataset = load_data_split(args.dataset, args.dataset_split)
        if args.dataset == "fetaqa" and args.dataset_split == "test":
            dataset = []
            with open(os.path.join(ROOT_DIR, "utils", "fetaqa", "fetaQA-v1_test.jsonl"), "r") as f:
                lines = f.readlines()
                for i,line in enumerate(lines):
                    dic = json.loads(line)
                    feta_id = dic['feta_id']
                    caption = dic['table_page_title']
                    question = dic['question']
                    answer = dic["answer"]
                    sub_title = dic['table_section_title']
                    header = dic['table_array'][0]
                    rows = dic['table_array'][1:]
                    data = {
                    "id": feta_id,
                    "table": {
                        "id": feta_id,
                        "header": header,
                        "rows": rows,
                        "page_title": caption,
                    },
                    "question": question,
                    "answer": answer
                    }
                    dataset.append(data)
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

    # Load openai keys
    with open(args.api_keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]

    # Annotate
    generator = Generator(args, keys=keys)
    generate_eids = list(range(0, len(dataset)))
    print(len(generate_eids))
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[int(g_eid) % args.n_processes].append(g_eid)
    print('\n******* Annotating *******')
    if 'gpt' in args.engine or 'bison' in args.engine:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.engine)
    g_dict = dict()
    worker_results = []
    pool = multiprocessing.Pool(processes=args.n_processes)
    for pid in range(args.n_processes):
        worker_results.append(pool.apply_async(worker_annotate, args=(
            pid,
            args,
            generator,
            generate_eids_group[pid],
            dataset,
            tokenizer
        )))

    for r in worker_results:
        worker_g_dict = r.get()
        g_dict.update(worker_g_dict)
    pool.close()
    pool.join()

    # Save annotation results
    if hasattr(args, 'input_file') and args.input_file:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        save_file_name = f'{base_name}_{args.generate_type}_sql.json'
    else:
        save_file_name = f'{args.dataset}_{args.dataset_split}_col_sql.json'

    with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
        json.dump(g_dict, f, indent=4)

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
    parser.add_argument('--prompt_file', type=str, default='templates/prompts/column.txt')
    parser.add_argument('--save_dir', type=str, default='results/model_gpt')
    parser.add_argument('--input_file', type=str, default=None, help='(optional) JSON file with single sample to bypass full dataset loading')

    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=1)

    # Program generation options
    parser.add_argument('--prompt_style', type=str, default='create_table_select_full_table',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',
                                 'create_table',
                                 'text_full_table',
                                 'transpose',
                                 'no_table'])
    parser.add_argument('--generate_type', type=str, default='col',
                        choices=['col', 'answer', 'row', 'verification'])
    parser.add_argument('--n_shots', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    # LLM options
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--n_parallel_prompts', type=int, default=1)
    parser.add_argument('--max_generation_tokens', type=int, default=256)
    parser.add_argument('--max_api_total_tokens', type=int, default=14000)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--sampling_n', type=int, default=1)
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
