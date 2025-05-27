python_path=/home/xcheng/miniconda3/envs/qf/bin/python
proj_path="./"


# # 2024131-1131
# $python_path $proj_path/main.py \
# --plot \
# --allfield_embed \
# --singlefield_embed \
# --mixed_hard \
# --mixed_soft \
# --model "distiluse-base-multilingual-cased-v1" \
# --tag "result_20250104-2223" \

# zip -r result_20250104-2223.zip result_20250104-2223


# # 20250103-1510
# $python_path $proj_path/main.py \
# --plot \
# --allfield_embed \
# --singlefield_embed \
# --mixed_hard \
# --mixed_soft \
# --tag "result_20250105-0044" \
# --model "GanymedeNil/text2vec-large-chinese" \


# zip -r result_20250105-0044.zip result_20250105-0044


# # 20250107-1440
# $python_path $proj_path/main.py \
# --plot \
# --allfield_embed \
# --singlefield_embed \
# --mixed_hard \
# --mixed_soft \
# --tag "result_20250107-1440" \
# --model "distiluse-base-multilingual-cased-v1" \


# zip -r result_20250107-1440.zip result_20250107-1440


# # 20250107-1455
# $python_path $proj_path/main.py \
# --plot \
# --allfield_embed \
# --singlefield_embed \
# --mixed_hard \
# --mixed_soft \
# --tag "result_20250107-1455" \
# --model "GanymedeNil/text2vec-large-chinese" \


# zip -r result_20250107-1455.zip result_20250107-1455


# 20250527-1448
$python_path $proj_path/main.py \
--plot \
--allfield_embed \
--singlefield_embed \
--mixed_hard \
--mixed_soft \
--tag "result_20250527-1448" \
--model "distiluse-base-multilingual-cased-v1" \
--input_dir "./entity_matching2"


zip -r result_20250527-1448.zip result_20250527-1448


# 20250527-1300
$python_path $proj_path/main.py \
--plot \
--allfield_embed \
--singlefield_embed \
--mixed_hard \
--mixed_soft \
--tag "result_20250527-1300" \
--model "GanymedeNil/text2vec-large-chinese" \
--input_dir "./entity_matching2"

zip -r result_20250527-1300.zip result_20250527-1300