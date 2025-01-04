python_path=/home/xcheng/miniconda3/envs/qf/bin/python
proj_path="./"


# 2024131-1131
$python_path $proj_path/allemb.py \
--plot \
--allfield_embed \
--singlefield_embed \
--mixed_hard \
--mixed_soft \
# --model "distiluse-base-multilingual-cased-v1" \


# 20250103-1510
$python_path $proj_path/allemb.py \
--plot \
--allfield_embed \
--singlefield_embed \
--mixed_hard \
--mixed_soft \
--tag "result_20250103-1510" \
--model "GanymedeNil/text2vec-large-chinese" \