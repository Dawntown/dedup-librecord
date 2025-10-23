# %%
import dedupe.variables
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="SimSun", style="ticks")
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

# %%
candidate_models = [
    'shibing624/text2vec-base-chinese',
    'shibing624/text2vec-base-chinese-sentence',
]
model = SentenceTransformer("GanymedeNil/text2vec-large-chinese")
# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


# 同义词聚类
# %%

example_synonyms = [
    ('文学', '文学、美术', '文化'),
    ('医学', '肝胆外科', '妇产科'),
    ('体育教育', '体育教学'),
    ('法律', '法学'),
    ('机械', '机械工程'),
    ('副教授', '导师', '讲师'),
    ('记者', '编辑', '新闻工作者'),
    ('主治医师', '主任医师'),
    ('检查官', '检查员'),
    ('图书馆学', '图书与情报'),
]

example_synonyms_words = list(set([w for ws in example_synonyms for w in ws]))
example_synonyms_embs = model.encode(example_synonyms_words)    
example_synonyms_embs_df = pd.DataFrame(example_synonyms_embs, index=example_synonyms_words)
        
# cossim = cosine_similarity(example_synonyms_embs, example_synonyms_embs)
# cossim_df = pd.DataFrame(cossim, index=example_synonyms_words, columns=example_synonyms_words)  
# %%
sns.clustermap(
    example_synonyms_embs_df, cmap='gray_r', 
    metric='cosine', figsize=(12, 7), 
    col_cluster=False,
    dendrogram_ratio=0.2,
)
plt.tight_layout()
plt.savefig('examples/synonyms_embs_df.pdf', dpi=300)
plt.show()
# %%

# 例句聚类

example_sentences = [
    '动物生物化学；生物统计附试验设计；动物生物化学；基础化学；<br> xxx',
    '动物生物化学；动物生物化学；基础化学; 生物统计附试验设计',
    '动物生物化学',
    '中西哲学入门；心身关系问题探析',
    '著有长篇小说《悠猎乡情》、《为不贞妻子辩护的丈夫》，小说散文集《猎村纪事》、《带刺的玫瑰》，电视剧本《伊娜索的爱情》。',
    '《心身关系问题探析》'
    '国际货币经济学 = International monetary economics / 贝内特·T. 麦克勒姆著; 陈未, 张杰译',
    '国际货币经济学 陈未, 张杰译'
]
example_ss_embs = model.encode(example_sentences)
example_ss_embs_df = pd.DataFrame(example_ss_embs, index=range(1, len(example_sentences)+1))

sns.clustermap(
    example_ss_embs_df, cmap='gray_r',
    metric='cosine', figsize=(10, 5),
    col_cluster=False,
    dendrogram_ratio=0.2,
)
plt.tight_layout()
plt.savefig('examples/sentences_embs_df.pdf', dpi=300)
plt.show()
# %%

# 地点聚类

example_locations = [
    '北京市',
    '丰台区',
    '海淀区学院路',
    '济南市',
    '山东省',
    '江苏省盐城市',
    '江苏省南京市',
    '吉安市吉州区',
    '江西省青原区',
]

example_locations_embs = model.encode(example_locations)
example_locations_embs_df = pd.DataFrame(example_locations_embs, index=example_locations)

sns.clustermap(
    example_locations_embs_df, cmap='gray_r',
    metric='cosine', figsize=(10, 5),
    col_cluster=False,
    dendrogram_ratio=0.2,
)
plt.tight_layout()
plt.savefig('examples/locations_embs_df.pdf', dpi=300)
plt.show()
# %%


from GeocodingCHN import Geocoding

geocoding = Geocoding()

def check_NA_values(s):
    if "未知" in str(s) or "未提及" in str(s) or str(s) == "无":
        return True
    if str(s).isspace():
        return True
    return False

def standardize_address(x):
    if check_NA_values(x):
        return x
    try:
        loc = geocoding.normalizing(x)
        x_normal = loc.province + \
            (loc.city if not pd.isna(loc.city) else '') + \
            (loc.district if not pd.isna(loc.district) else '')
            
        return x_normal
    
    except:
        print(x, 'Not Found')
        return x

example_loc_normal = [standardize_address(loc) for loc in example_locations]
example_loc_normal_embs = model.encode(example_loc_normal)
example_loc_normal_embs_df = pd.DataFrame(example_loc_normal_embs, index=example_loc_normal)

sns.clustermap(
    example_loc_normal_embs_df, cmap='gray_r',
    metric='cosine', figsize=(10, 5),
    col_cluster=False,
    dendrogram_ratio=0.2,
);

plt.tight_layout()
plt.savefig('examples/loc_normal_embs_df.pdf', dpi=300)
plt.show()
# %%


similarity_matrix = np.zeros((len(example_locations), len(example_locations)))
for i, loc1 in enumerate(example_loc_normal):
    for j, loc2 in enumerate(example_loc_normal):
        similarity_matrix[i, j] = geocoding.similarity(loc1, loc2)
        
sns.heatmap(similarity_matrix, xticklabels=example_locations, yticklabels=example_locations, cmap='gray_r')
plt.tight_layout()
plt.savefig('examples/loc_normal_similarity_matrix.pdf', dpi=300)
plt.show()
# %%


# 日期聚类

import arrow, re
from dateutil import parser

example_dates = [
    '2021年1月',
    '2021.1',
    '2021.7',
    '2021年11月',
    '2020.11',
    '2020',
    '2012.1',
]


def standardize_date(d):
    try:
        yyyy = re.search(r'\d{4}', d).group()
        try:
            mm = re.findall(r'.*\D+(\d{1,2})\D*', d)[0]
            return f'{yyyy}.{mm.zfill(2)}'
        except:
            return yyyy
    except:
        return d
    
example_dates_normal = [standardize_date(d) for d in example_dates]

example_dates_embs = model.encode(example_dates_normal)
example_dates_embs_df = pd.DataFrame(example_dates_embs, index=example_dates)

# %%
sns.clustermap(
    example_dates_embs_df, cmap='gray_r',
    metric='cosine', figsize=(5, 3),
    col_cluster=False,
    dendrogram_ratio=0.2,
)

plt.tight_layout()
plt.savefig('examples/dates_embs_df.pdf', dpi=300)
plt.show()
# %%

# from scipy.spatial.distance import pdist, squareform
# from 


# def 

# pdist(example_dates_embs, metric='cosine')


# # Dedupe example

# import dedupe

# def check_NA_values(s):
#     if "未知" in str(s) or "未提及" in str(s) or str(s) == "无":
#         return True
#     if str(s).isspace():
#         return True
#     return False


# field_def = [
#     dedupe.variables.String('姓名'),
#     dedupe.variables.String('别名', has_missing=True),
#     dedupe.variables.String('生卒年或个人活动日期', has_missing=True),
#     dedupe.variables.Exact('性别', has_missing=True),
#     dedupe.variables.String('学历', has_missing=True),
#     dedupe.variables.String('受教育机构', has_missing=True),
#     dedupe.variables.String('在职单位', has_missing=True),
#     dedupe.variables.String('籍贯', has_missing=True),
#     dedupe.variables.String('职业', has_missing=True),
#     dedupe.variables.String('发表的著作实体', has_missing=True),
#     dedupe.variables.String('活动领域', has_missing=True),
#     dedupe.variables.Exact('民族', has_missing=True),
#     dedupe.variables.Exact('国籍', has_missing=True),
#     dedupe.variables.String('其他信息', has_missing=True),
# ]

# field_types = [
#     {'field': '姓名', 'type': 'String'},
#     {'field': '别名', 'type': 'String'},
#     {'field': '生卒年或个人活动日期', 'type': 'String'},
#     {'field': '性别', 'type': 'Exact'},
#     {'field': '学历', 'type': 'String'},
#     {'field': '受教育机构', 'type': 'String'},
#     {'field': '在职单位', 'type': 'String'},
#     {'field': '籍贯', 'type': 'String'},
#     {'field': '职业', 'type': 'String'},
#     {'field': '发表的著作实体', 'type': 'String'},
#     {'field': '活动领域', 'type': 'String'},
#     {'field': '民族', 'type': 'Exact'},
#     {'field': '国籍', 'type': 'Exact'},
#     {'field': '其他信息', 'type': 'String'},
# ]
# field_types_df = pd.DataFrame(field_types)

# demo_df = pd.read_excel('demo_刘伟.xlsx', index_col='UID')
# demo_dict = demo_df[field_types_df['field'].to_list()].T.to_dict()

# for _, d in demo_dict.items():
#     for k, v in d.items():
#         if check_NA_values(v):
#             d[k] = None

# deduper = dedupe.Dedupe(field_def)

# deduper.prepare_training(demo_dict, sample_size=10)
# deduper.train()




# import difflib

# example_sentences = [
#     '动物生物化学；生物统计附试验设计；动物生物化学；基础化学；<br> xxx',
#     '动物生物化学；动物生物化学；基础化学; 生物统计附试验设计',
#     '动物生物化学',
#     '中西哲学入门；心身关系问题探析',
#     '著有长篇小说《悠猎乡情》、《为不贞妻子辩护的丈夫》，小说散文集《猎村纪事》、《带刺的玫瑰》，电视剧本《伊娜索的爱情》。',
#     '《心身关系问题探析》'
#     '国际货币经济学 = International monetary economics / 贝内特·T. 麦克勒姆著; 陈未, 张杰译',
#     '国际货币经济学 陈未, 张杰译'
# ]


# s1 = example_sentences[0]
# s2 = example_sentences[1]

# matcher = difflib.SequenceMatcher(lambda s: s.isspace(), s1, s2)
# matchs = matcher.get_matching_blocks()
# for m in matchs:
#     if m.size > 0:
#         print(s1[m.a: m.a + m.size])
        
        
# from rapidfuzz import fuzz, process
# import jieba, re
# from Bio import Align
# import edlib
# import pandas as pd
# import numpy as np
# import parasail

# similarity = fuzz.partial_ratio(s1, s2, score_cutoff=60); similarity

# process.extract(s1, example_sentences, scorer=fuzz.partial_token_sort_ratio, score_cutoff=60)

# fuzz.partial_ratio(s1, s2)
# fuzz.token_sort_ratio(s1, s2)

# stop_pattern = re.compile(r'([;；,，、\s]+)',)
# stop_pattern.split(s1)

# def has_subseq(s1, s2):
#     # pattern = re.compile(r'.*'.join(jieba.lcut(s1)))
#     pattern = '.*'.join(s1)
#     return re.search(pattern, s2) is not None

# has_subseq(s1, s2)
# has_subseq(s2, s1)

# aligner = Align.PairwiseAligner(mode='local', match_score=2, mismatch_score=-10, gap_score=-1)
# result = aligner.align(s1, s2)
# for r in result:
#     print(r)
    
# result.score


# result = edlib.align(s1, s2, task='path', mode='HW')
# result = parasail.nw_trace(s1, s2, 10, 1, parasail.blosum62)

# def parse_cigar(cigar):
#     pattern = re.compile(r'(\d+)([=XID])')
#     count_df = pd.DataFrame(np.array(pattern.findall(cigar)), columns=['count', 'op'])
#     return count_df.groupby('op').apply(lambda x: x['count'].astype(int).to_list(), include_groups=False).to_dict()

# parse_cigar(result['cigar'])

# import itertools


# from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
# from sparkai.core.messages import ChatMessage
# import pandas as pd

# #星火认知大模型Spark Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
# SPARKAI_URL = 'wss://spark-api.xf-yun.com/v4.0/chat'
# #星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
# SPARKAI_APP_ID = 'b1ca2320'
# SPARKAI_API_SECRET = 'YjZmMGM1MDU2MTkzY2FlZGMxY2FkZTJh'
# SPARKAI_API_KEY = '5051d5cc32a7886c6eb3b08b625e991d'
# #星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
# SPARKAI_DOMAIN = '4.0Ultra'

# spark = ChatSparkLLM(
#     spark_api_url=SPARKAI_URL,
#     spark_app_id=SPARKAI_APP_ID,
#     spark_api_key=SPARKAI_API_KEY,
#     spark_api_secret=SPARKAI_API_SECRET,
#     spark_llm_domain=SPARKAI_DOMAIN,
#     streaming=False,
# )
# demo_df = pd.read_excel('demo_刘伟.xlsx', index_col='UID')
# prompt = "这是一段多个已发表实体的记录，但是分隔符不明确，需要将其分隔开来，可以帮我分割实体并直接输出成json格式吗？\n"

# # 发表实体
# batch_prompt = "以下每行文本都是一段多个已发表实体的记录，但是分隔符不明确，需要将其分隔开来，可以帮我把这些实体分割开并输出成列表吗？要求统一格式，仅保留标题而去除其他信息，例如作者、出版社、出版年份等信息，每条记录一个实体列表，把拼音转换成中文，并直接输出成json\n"
# batch_inputs = "\n".join(demo_df['发表的著作实体'].to_list())

# # 别名
# batch_prompt = "以下每行文本都是一个人的别名，但是分隔符不明确，需要将其分隔开来，可以帮我把这些名字用统一的格式分隔开吗？要求每条都使用统一的分隔符“+”，如果某条记录是“未提及”，则该条记录对应的输出为空字符，最后输出一个json列表。\n"
# batch_inputs = "\n".join(demo_df['别名'].to_list())

# messages = [ChatMessage(
#     role="user",
#     content=batch_prompt + batch_inputs,
# )]
# handler = ChunkPrintHandler()
# a = spark.generate([messages], callbacks=[handler])
# # extract the output
# str_output = a.flatten()[0].generations[0][0].text.replace('```json\n', "").replace('```', "")
# print(str_output)
# import json
# json_output = json.loads(str_output)

# print(str_output)














# from pypinyin import pinyin, lazy_pinyin, Style
# import pypinyin
# from Pinyin2Hanzi import DefaultDagParams, dag, DefaultHmmParams, viterbi, simplify_pinyin, is_pinyin

# input_pinyin = "E'erduosi Pendi huang tu di qu gong cheng jian she chang jian di zhi zai hai yan jiu"
# # change pinyin to hanzi

# def parse_pinyin(input_pinyin):
#     pinyin_list = [simplify_pinyin(p) for p in input_pinyin.lower().split()]
#     pinyin_list = [p for p in pinyin_list if is_pinyin(p)]
#     return pinyin_list

# dagparams = DefaultDagParams()
# result = dag(dagparams, parse_pinyin(input_pinyin), path_num=5, log=True)
# for item in result:
#     print(item.path, item.score)


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.spatial.distance import pdist, squareform
# from scipy.cluster.hierarchy import linkage, dendrogram


# data_df = pd.DataFrame(np.array([(2, 8), (3, 8), (6, 7), (6, 6), (1, 2), (2, 3), (7, 1), (8, 2)]), columns=['x', 'y'], index=[f'p{i}' for i in range(1, 9)])
# dist_mtx = pdist(data_df, metric='euclidean')
# linkages = linkage(dist_mtx, method='complete')
# fig, axs = plt.subplots(2, 2, figsize=(10,10))
# # [0,0]
# sns.scatterplot(x='x', y='y', data=data_df, ax=axs[0,0])
# for x, y, label in zip(data_df['x'], data_df['y'], data_df.index):
#     axs[0,0].text(x, y, label)
# axs[0,0].set_title('Scatter plot of data points')
# # [0,1]
# dendrogram(
#     linkages, 
#     labels=data_df.index, 
#     orientation='left',
#     color_threshold=2,
#     ax=axs[0,1]
# )
# axs[0,1].vlines(2, 0, 100, color='r', linestyle='--')
# axs[0,1].set_title('Dendrogram of data points (Threshold=2)')
# axs[0,1].set_xlabel('Distance')
# axs[0,1].set_ylabel('Data points')
# # [1,0]
# dendrogram(
#     linkages, 
#     labels=data_df.index, 
#     orientation='left',
#     color_threshold=6,
#     ax=axs[1,0]
# )
# axs[1,0].vlines(6, 0, 100, color='r', linestyle='--')
# axs[1,0].set_title('Dendrogram of data points (Threshold=6)')
# axs[1,0].set_xlabel('Distance')
# axs[1,0].set_ylabel('Data points')
# # [1,1]
# dendrogram(
#     linkages, 
#     labels=data_df.index, 
#     orientation='left',
#     color_threshold=8,
#     ax=axs[1,1]
# )
# axs[1,1].vlines(8, 0, 100, color='r', linestyle='--')
# axs[1,1].set_title('Dendrogram of data points (Threshold=8)')
# axs[1,1].set_xlabel('Distance')
# axs[1,1].set_ylabel('Data points')
# plt.tight_layout()

# plt.show()


        

# # %%
