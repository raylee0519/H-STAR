import os

ROOT_DIR = os.path.join(os.path.dirname(__file__))
print(ROOT_DIR)

# Disable the TOKENIZERS_PARALLELISM
# TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false" # Linux ver
TOKENIZER_FALSE = "set TOKENIZERS_PARALLELISM=false &&" # Window ver

# Set input file path for one-sample WIKITQ test
INPUT_FILE = "test_data/wikitq_single_sample.json"  # ← 여기에 아래 JSON 내용을 저장한 경로 사용

## Step 1: col_sql
os.system(fr"""{TOKENIZER_FALSE} python ./scripts/model_gpt/col_sql.py ^
--input_file {INPUT_FILE} ^
--prompt_file prompts/col_select_sql.txt ^
--n_parallel_prompts 3 ^
--max_generation_tokens 512 ^
--temperature 0.3 ^
--sampling_n 2 ^
-v""")

## Step 2: col_text
os.system(fr"""{TOKENIZER_FALSE} python ./scripts/model_gpt/col_text.py ^
--input_file wikitq_test_col_sql.json ^
--prompt_file prompts/col_select_text.txt ^
--n_parallel_prompts 3 ^
--max_generation_tokens 512 ^
--temperature 0.4 ^
--sampling_n 2 ^
-v""")

## Step 3: row_sql
os.system(fr"""{TOKENIZER_FALSE} python ./scripts/model_gpt/row_sql.py ^
--input_file wikitq_test_col_text.json ^
--prompt_file prompts/row_select_sql.txt ^
--n_parallel_prompts 3 ^
--max_generation_tokens 512 ^
--temperature 0.3 ^
--sampling_n 2 ^
-v""")

## Step 4: row_text
os.system(fr"""{TOKENIZER_FALSE} python ./scripts/model_gpt/row_text.py ^
--input_file wikitq_test_row_sql.json ^
--prompt_file prompts/row_select_text.txt ^
--n_parallel_prompts 3 ^
--max_generation_tokens 512 ^
--temperature 0.4 ^
--sampling_n 2 ^
-v""")

## Step 5: reason_sql
os.system(fr"""{TOKENIZER_FALSE} python ./scripts/model_gpt/reason_sql.py ^
--input_file wikitq_test_row_text.json ^
--prompt_file prompts/sql_reason_wtq.txt ^
--n_parallel_prompts 3 ^
--max_generation_tokens 512 ^
--temperature 0.1 ^
--sampling_n 1 ^
-v""")

## Step 6: final text reason
os.system(fr"""{TOKENIZER_FALSE} python ./scripts/model_gpt/reason_text.py ^
--input_file wikitq_test_sql_reason.json ^
--prompt_file prompts/text_reason_wtq.txt ^
--n_parallel_prompts 1 ^
--max_generation_tokens 512 ^
--temperature 0.0 ^
--sampling_n 1 ^
-v""")
