python_path=/home/xcheng/miniconda3/envs/qf/bin/python
proj_path="./"


# 2024131-1131
$python_path $proj_path/main.py \
--plot \
--allfield_embed \
--singlefield_embed \
--mixed_hard \
--mixed_soft \
--model "distiluse-base-multilingual-cased-v1" \
--tag "result_20250104-2223" \


# 20250103-1510
$python_path $proj_path/main.py \
--plot \
--allfield_embed \
--singlefield_embed \
--mixed_hard \
--mixed_soft \
--tag "result_20250105-0044" \
--model "GanymedeNil/text2vec-large-chinese" \