#!/home/xcheng/miniconda3/envs/qf/bin/python
import collections, warnings, os, sys, time
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import re
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="SimSun")

from GeocodingCHN import Geocoding
geocoding = Geocoding()
from Pinyin2Hanzi import DefaultDagParams, dag, simplify_pinyin, is_pinyin
dagparams = DefaultDagParams()
from pypinyin import lazy_pinyin

import difflib

try:
    import torch_directml
    DEVICE = torch_directml.device()
except:
    import torch
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'


def check_NA_values(s):
    if "未知" in str(s) or "未提及" in str(s) or "未提供" in str(s) or "未填写" in str(s) or "Not Provided" in str(s) or str(s) == "无":
        return True
    if str(s).isspace() or str(s) == '':
        return True
    return False

def strip_cell_values(df):
    """
    去除DataFrame中所有字符串类型单元格的两端空白字符，包括标题空白字符
    """
    # 先处理列名
    df.columns = df.columns.str.strip()
    
    # 处理单元格内容
    for col in df.columns:
        if df[col].dtype == 'object':  # 只处理字符串类型的列
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def vers_mean(x, axis=None, weights=None, method='arithmetic'):
    #put the values in weights to correct dimension for broadcasting
    if method == 'arithmetic':
        return np.average(x, axis=axis, weights=weights)
    elif method == 'geometric':
        dims = [i for i in range(x.ndim) if i != axis]
        weights = np.expand_dims(weights, dims)
        return np.prod(x ** weights, axis=axis) ** (1 / np.sum(weights, axis=axis))
    elif method == 'harmonic':
        dims = [i for i in range(x.ndim) if i != axis]
        weights = np.expand_dims(weights, dims)
        return np.sum(weights, axis=axis) / np.sum(weights / x, axis=axis)
    elif method == 'quadratic':
        return np.sqrt(np.nansum(x**2, axis=axis) / np.sum(~np.isnan(x), axis=axis))
    elif method == 'median':
        return np.nanmedian(x, axis=axis)
    else:
        raise ValueError("Method not recognized")


def impute_dist(d_mtx, na_list, method='mean', fixed_value=None):
    na_list = np.array(na_list)
    if na_list.mean() > 0.7:
        Warning("Too many missing values, imputation may not be reliable")
        method = 'fixed'
        fixed_value = 0.5
    if method == 'fixed':
        row_impute = fixed_value
        col_impute = fixed_value
        block_impute = fixed_value
    elif method == 'mean':
        row_impute = np.nanmean(d_mtx, axis=0)[~na_list]
        col_impute = np.nanmean(d_mtx, axis=1)[~na_list]
        block_impute = np.nanmean(d_mtx)
    elif method == 'median':
        row_impute = np.nanmedian(d_mtx, axis=0)[~na_list]
        col_impute = np.nanmedian(d_mtx, axis=1)[~na_list]
        block_impute = np.nanmedian(d_mtx)
        
    for i in np.where(na_list)[0]:
        d_mtx[i, ~na_list] = row_impute
        d_mtx[~na_list, i] = col_impute
        d_mtx[i, na_list] = block_impute
        
    for i in np.where(na_list)[0]:
        d_mtx[i, i] = 0
    
    assert np.abs(d_mtx - d_mtx.T).max() < 1e-8
    
    return (d_mtx + d_mtx.T) / 2


def longest_common_subsequence(s1, s2):
    if check_NA_values(s1) or check_NA_values(s2):
        return ''
    matched_list = []
    matcher = difflib.SequenceMatcher(lambda s: s.isspace(), s1, s2)
    matchs = matcher.get_matching_blocks()
    for m in matchs:
        if m.size > 0:
            matched_list.append(s1[m.a: m.a + m.size])

    matcher = difflib.SequenceMatcher(lambda s: s.isspace(), s2, s1)
    matchs = matcher.get_matching_blocks()
    for m in matchs:
        if m.size > 0:
            matched_list.append(s2[m.a: m.a + m.size])
    length_list = [len(m) for m in matched_list]
    if len(length_list) == 0:
        return ''
    return matched_list[np.argmax(length_list)]


def find_connected_components(edges):
    # 构造邻接表
    graph = collections.defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()  # 记录访问过的节点
    components = []  # 存储每个连通分量

    def dfs(node, component):
        """深度优先搜索 (DFS)"""
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    # 遍历所有节点，找出每个连通分量
    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)

    return components



class Deduplication(object):
    def __init__(self, model_name='distiluse-base-multilingual-cased-v1', field_weights=None, device='cpu'):
        
        # required_fields = [
        #     "UID", # 用于标识记录: [文件名]_[行号]
        #     "姓名", "别名", 
        #     "生卒年或个人活动日期", 
        #     "发表的著作实体", 
        #     "籍贯", 
        #     "活动领域",
        #     "受教育机构", "在职单位", 
        #     "职业"
        # ]
        
        # assert "" in field_weights.keys()

        self.model = SentenceTransformer(model_name, device=device)
        self.field_weights = field_weights
        
    def build_prompt(self):
        prompt = "为了图书馆作者条目去重，我们列出以下信息，各字段的匹配规则和权重如下："
        for field, weight in self.field_weights.items():
            prompt += f"{field}：{weight}，\n"
        # prompt += "其他不参与匹配。\n"
        
        prompt += "匹配规则：\n"
        prompt += "如果”姓名“和”别名“包括多个的，只要其中一个值匹配即可\n"
        prompt += "如果两条记录的“生卒年或个人活动日期”完全一致或者有部分重叠，则认为在这一属性上匹配\n"
        prompt += "”发表的著作实体“中《中国文学家大辞典》《中国专家大辞典 6》《中国人才辞典》等不是作者的著作实体，这些不参与匹配\n"
        prompt += "涉及多个”受教育机构“的，只要有一个机构匹配可认为具有高匹配度；涉及二级机构的，每个值前向匹配\n"
        prompt += "”在职单位“涉及二级单位的（如**学院，系），也只匹配到一级单位\n"
        prompt += "”职业“涉及多个职称的，只要有一个匹配可认为具有高匹配度\n"
        prompt += "”活动领域“涉及多个职称的，只要有一个匹配可认为具有高匹配度\n"
        
        return prompt
    
    def build_infotext(self, rec_df):
        # infotext_head = "以下是待去重的条目：\n"
        infotext_head = ''
        rec_id_list = []
        rec_text_list = []
        for i, rec in rec_df.iterrows():
            if check_NA_values(rec['别名']):
                infotext_body = f"姓名和别名包括：{rec['姓名']}。"
            else:
                infotext_body = f"姓名和别名包括：{rec['姓名'] + ',' + rec['别名']}。"
            if not check_NA_values(rec['性别']):
                infotext_body += f"性别：{rec['性别']}，" # 未被考虑的信息
            if not check_NA_values(rec['民族']):
                infotext_body += f"民族：{rec['民族']}，"
            if not check_NA_values(rec['学历']):
                infotext_body += f"学历：{rec['学历']}，"
            if not check_NA_values(rec['国籍']):
                infotext_body += f"国籍：{rec['国籍']}，"
            if not check_NA_values(rec['生卒年或个人活动日期']):
                infotext_body += f"生卒年或个人活动日期：{rec['生卒年或个人活动日期']}，"
            if not check_NA_values(rec['籍贯']):
                infotext_body += f"籍贯：{rec['籍贯']}，"
            if not check_NA_values(rec['活动领域']):
                infotext_body += f"活动领域：{rec['活动领域']}，"
            if not check_NA_values(rec['受教育机构']):
                infotext_body += f"受教育机构：{rec['受教育机构']}，"
            if not check_NA_values(rec['在职单位']):
                infotext_body += f"在职单位：{rec['在职单位']}，"
            if not check_NA_values(rec['职业']):
                infotext_body += f"职业：{rec['职业']}，"
            if not check_NA_values(rec['发表的著作实体']):
                infotext_body += f"发表的著作实体：{rec['发表的著作实体']}。"
            
            rec_id_list.append(rec['UID'])
            rec_text_list.append(infotext_head + infotext_body)
        
        return rec_id_list, rec_text_list
    
    def build_embs(self, rec_df):
        # prompt = self.build_prompt()
        prompt = ''
        rec_id_list, rec_text_list = self.build_infotext(rec_df)
        rec_emb_list = self.model.encode(list(map(lambda x: prompt + x, rec_text_list)), batch_size=16)
        return pd.DataFrame(rec_emb_list, index=rec_id_list)
    
    def build_embs_each_field(self, rec_df, field_names):
        field_emb_dict = {}
        for field in field_names:
            na_list = rec_df[field].apply(check_NA_values).tolist()
            text_list = rec_df[field].apply(lambda x: x if not check_NA_values(x) else '').tolist()
            emb_list = self.model.encode(text_list, batch_size=16)
            field_emb_dict.update({field: {'na': na_list, 'emb': emb_list}})
        return field_emb_dict
    
    def compute_emb_dist(self, field_embs, metric='cosine', impute='mean', field_cutoffs=None):
        if isinstance(field_embs, dict):
            d_mtx_dict = {}
            for field, emb_dict in field_embs.items():
                na_list = emb_dict['na']
                emb_list = emb_dict['emb']
                d_mtx = squareform(pdist(np.array(emb_list), metric=metric))
                if impute is None:
                    d_mtx[na_list, :] = np.nan
                    d_mtx[:, na_list] = np.nan
                else:
                    d_mtx = impute_dist(d_mtx, na_list, method=impute)
                if field_cutoffs is not None:
                    c = field_cutoffs.get(field, None)
                    if c is not None:
                        cc = c if c < 0 else np.quantile(squareform(d_mtx), c)
                        d_mtx = (d_mtx > cc).astype(int)
                d_mtx_dict.update({field: d_mtx})
            return d_mtx_dict
        else:
            return squareform(pdist(np.array(field_embs), metric=metric))
        
    # def compute_str_dist(self, field_texts, metric='levenshtein'):
    #     assert metric in ['levenshtein', 'jaccard']
    #     if isinstance(field_texts, dict):
    #         d_mtx_dict = {}
    #         for field, text_list in field_texts.items():
    #             d_mtx = squareform(pdist(
    #                 np.array(text_list).reshape(-1,1), 
    #                 lambda s1, s2: eval(metric)(s1[0], s2[0])
    #             ))
    #             d_mtx_dict.update({field: d_mtx})
    #         return d_mtx_dict
    #     else:
    #         return squareform(pdist(
    #             np.array(field_texts).reshape(-1,1),
    #             lambda s1, s2: eval(metric)(s1[0], s2[0])
    #         ))
        
    def pairwise_common_subseq(self, field_texts, field_cutoffs=None):
        
        if isinstance(field_texts, dict):
            matched_dict = {}
            for field, text_list in field_texts.items():
                seq_matrix = squareform(pdist(
                    np.array(text_list).reshape(-1,1), 
                    lambda s1, s2: - len(longest_common_subsequence(s1[0], s2[0]))
                ))
                seq_matrix[np.arange(seq_matrix.shape[0]), np.arange(seq_matrix.shape[0])] = list(map(len, text_list))
                if field_cutoffs is not None:
                    c = field_cutoffs.get(field, None)
                    if c is not None:
                        seq_matrix = (seq_matrix > c).astype(int)
                        seq_matrix[np.arange(seq_matrix.shape[0]), np.arange(seq_matrix.shape[0])] = 0
                    
                matched_dict.update({field: seq_matrix})
            return matched_dict
        else:
            seq_matrix = squareform(pdist(
                np.array(field_texts).reshape(-1,1), 
                lambda s1, s2: - len(longest_common_subsequence(s1[0], s2[0]))
            ))
            seq_matrix[np.arange(seq_matrix.shape[0]), np.arange(seq_matrix.shape[0])] = list(map(len, field_texts))
            if field_cutoffs is not None:
                seq_matrix = (seq_matrix > field_cutoffs).astype(int)
                seq_matrix[np.arange(seq_matrix.shape[0]), np.arange(seq_matrix.shape[0])] = 0
            return seq_matrix
        
        
    def merge_dist(self, d_mtx_dict, weight_dict=None, method='arithmetic'):
        if method in ['geometric', 'harmonic', 'quadratic', 'median', 'arithmetic']:
            n = list(d_mtx_dict.values())[0].shape[0]
            merged_dist = np.zeros((len(d_mtx_dict), n, n))
            weights = np.zeros(len(d_mtx_dict))
            if weight_dict is None:
                weight_dict = {field: 1/len(d_mtx_dict) for field in d_mtx_dict.keys()}
            for i, (field, w) in enumerate(weight_dict.items()):
                merged_dist[i] = (d_mtx_dict[field] - d_mtx_dict[field].min()) / (d_mtx_dict[field].max() - d_mtx_dict[field].min())
                weights[i] = w
            return vers_mean(merged_dist, axis=0, weights=weights, method=method)
        elif method == 'top':
            n = list(d_mtx_dict.values())[0].shape[0]
            merged_dist = np.zeros((n, n))
            for field, d_mtx in d_mtx_dict.items():
                merged_dist += d_mtx - 1
            return merged_dist
        else:
            raise ValueError("Method not recognized")
    
    def plot_distance_hist(self, distances, dataset_name='dataset', ax=None):
        # 绘制距离分布直方图
        if len(distances.shape) == 2:
            distances = squareform(distances)
        sns.histplot(distances, bins=100, ax=ax)
        ax.set_title(f"HC Distance Distribution for {dataset_name}", fontsize=15)
        ax.set_xlabel("Distance", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
    
    def plot_dendrogram(self, linkages, labels, threshold, dataset_name='dataset', ax=None):
        # 绘制树状图（dendrogram）
        dendrogram(
            linkages, 
            labels=labels, 
            orientation='left',
            color_threshold=threshold,
            ax=ax
        )
        ax.set_title(f"HC Dendrogram for {dataset_name}", fontsize=15)
        ax.set_ylabel("Unique ID", fontsize=12)
        ax.set_xlabel("Distance", fontsize=12)
        
    def plot_overview(self, distances, labels, figsize=(10, 10), path=None):
        linkages = linkage(squareform(distances) - distances.min(), method='single')
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        self.plot_distance_hist(distances, dataset_name='dataset', ax=axs[0])
        self.plot_dendrogram(linkages, labels, 0.05, dataset_name='dataset', ax=axs[1])
        plt.tight_layout()
        if path is not None:
            print(f"Save the overview plot to {path}")
            plt.savefig(path, dpi=300)
        
        # find the optimal threshold for clustering (p < 0.05)
        
    def deduplicate(self, distances, method='single', ps=0.1, ks=3):
        if method in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
            if isinstance(ps, float):
                ps = [ps]
            linkages = linkage(squareform(distances), method=method)
            group_ids = {}
            for p in ps:
                group_id = fcluster(linkages, int(distances.shape[0] * (1 - p)), criterion='maxclust')
                cluster_size = pd.Series(collections.Counter(group_id)).sort_values(ascending=False)
                cluster_size_df = pd.DataFrame({'size': cluster_size[cluster_size > 1]})
                cluster_size_df = cluster_size_df.assign(new_id=range(1, len(cluster_size_df)+1))
                map_dict = cluster_size_df['new_id'].to_dict()
                group_ids.update({f'{p} dups': np.array(list(map(map_dict.get, group_id)))})
            return group_ids
        elif method == 'top':
            if isinstance(ks, int):
                ks = [ks]
            group_ids = {}
            for k in ks:
                edge_matrix = distances <= -k
                edges = np.array(np.where(edge_matrix)).T
                components = find_connected_components(edges)
                group_id = np.zeros(distances.shape[0])
                for i, c in enumerate(components):
                    group_id[c] = i + 1
                cluster_size = pd.Series(collections.Counter(group_id)).sort_values(ascending=False)
                cluster_size_df = pd.DataFrame({'size': cluster_size[cluster_size > 1]})
                cluster_size_df = cluster_size_df.assign(new_id=range(1, len(cluster_size_df)+1))
                map_dict = cluster_size_df['new_id'].to_dict()
                group_ids.update({f'{k} top': np.array(list(map(map_dict.get, group_id)))})
            return group_ids
                

def standardize_address(x, uid=None):
    if check_NA_values(x):
        return x
    try:
        loc = geocoding.normalizing(x)
        x_normal = str(loc.province) + \
            (str(loc.city) if not pd.isna(loc.city) else '') + \
            (str(loc.district) if not pd.isna(loc.district) else '')
            
        return x_normal
    
    except:
        print(x, 'Not Found', f'for {uid}' if uid is not None else '')
        return x
    
    
def standardize_date(d, uid=None):
    if check_NA_values(d):
        return d
    try:
        yyyy = re.search(r'\d{4}', d).group()
        try:
            mm = re.findall(r'.*\D+(\d{1,2})\D*', d)[0]
            return f'{yyyy}.{mm.zfill(2)}'
        except:
            return yyyy
    except:
        print(d, 'Not Date', f'for {uid}' if uid is not None else '')
        return d
    
def parse_pinyin(s):
    s_list = [simplify_pinyin(p) for p in s.lower().split()]
    ispinyin_list = [is_pinyin(p) for p in s_list]
    pinyin_list = [p for p, flag in zip(s_list, ispinyin_list) if flag]
    return sum(ispinyin_list) > 0.5 * len(s_list), pinyin_list

def standardize_literature(s):
    s_lst = re.split(r'[;；，,/]', s)
    output_s = ''
    for i, s in enumerate(s_lst):
        flag, pinyin_list = parse_pinyin(s)
        if flag:
            output_s += ''.join(dag(dagparams, pinyin_list, path_num=5)[0].path)
        else:
            output_s += s
        if i < len(s_lst) - 1:
            output_s += '；'
    return output_s


def remove_pinyin_name(s, name):
    s_lst = [s for s in re.split(r'[;；，,/\s]', s.lower()) if not check_NA_values(s)]
    name_elements = lazy_pinyin(name)
    name_pinyin = ' '.join(name_elements)
    if len(name_elements) == 3: # 三个字的名字，后两个字经常被合并
        name_pinyin += ' ' + name_elements[1] + name_elements[2]
    if len(name_elements) == 4: # 四个字的名字，后两个字经常被合并，姓也可能被合并
        name_pinyin += ' ' + name_elements[0] + name_elements[1] # 两字姓
        name_pinyin += ' ' + name_elements[2] + name_elements[3] # 两字名
    isname_list = [(subs in name_pinyin) or (subs == name) for subs in s_lst]
    return '；'.join([str(s) for s, flag in zip(s_lst, isname_list) if not flag])
    

def check_address(x):
    if check_NA_values(x):
        return False
    try:
        loc = geocoding.normalizing(x)
        return False
    except:
        return True

def remove_cidian(x):
    if check_NA_values(x):
        return x
    x = x.replace(' <br>', '；')
    removed_books = re.findall('《*中国.{2,8}辞典\s*\d{0,2}》*[,;，；]?', x)
    for book in removed_books:
        print(f"Remove {book} from {x}")
        x = x.replace(book, '')
    return x.strip()

def concat_fields(x):
    return '；'.join(list(set([str(f) for f in x if not check_NA_values(f)])))

def merge_records(rec_df, id_field='CID'):
    if isinstance(id_field, str):
        id_field = [id_field]
    rec_df_merged = rec_df.groupby(id_field).agg({field: concat_fields for field in rec_df.columns if field not in id_field}).reset_index()    
    print(f"Merge {rec_df.shape[0] - rec_df_merged.shape[0]} records with the same {id_field}")
    return rec_df_merged




if __name__ == '__main__':
    import glob, tqdm, argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='Whether to plot the results')
    parser.add_argument('--demo', action='store_true', help='Whether to run the demo')
    parser.add_argument('--model', type=str, default='distiluse-base-multilingual-cased-v1', help='The model name for embedding')
    parser.add_argument('--allfield_embed', action='store_true', help='Whether to use all fields for embedding')
    parser.add_argument('--singlefield_embed', action='store_true', help='Whether to use single field for embedding')
    parser.add_argument('--mixed_hard', action='store_true', help='Whether to use mixed distances for hard clustering')
    parser.add_argument('--mixed_soft', action='store_true', help='Whether to use mixed distances for soft clustering')
    parser.add_argument('--tag', type=str, default='output', help='The tag for the output files (default: output)')
    parser.add_argument('--merge_cid', action='store_true', help='Whether to merge the records with the same control ID')
    parser.add_argument('--input_dir', type=str, default='./entity_matching', help='The input directory')
    # args = parser.parse_args([])
    args = parser.parse_args()
    
    rename_dict = {
        '生卒年或个人活动日期': '生卒年或个人活动日期',
        '生卒年（活动日期）': '生卒年或个人活动日期',
        '生卒年': '生卒年或个人活动日期',
        '生卒年/个人活动日期': '生卒年或个人活动日期',
        '生卒年/活动日期': '生卒年或个人活动日期',
        '发表的著作': '发表的著作实体',
    }
    # candidate_models = [
    #     'shibing624/text2vec-base-chinese',
    #     'shibing624/text2vec-base-chinese-sentence',
    # ]
    # model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    fname_list = glob.glob(f'{args.input_dir}/*.xlsx')
    
    # clean and process the data
    rec_df_all = pd.concat([
        (strip_cell_values(pd.read_excel(fname))
         .rename(columns=rename_dict)
         .assign(文件名=fname.split('/')[-1].split('.')[0].strip()))
        for fname in tqdm.tqdm(fname_list)
    ], axis=0)
    rec_df_all.fillna('', inplace=True)
    rec_df_all['UID'] = rec_df_all['文件名'] + '_' + rec_df_all.index.astype(str)
    rec_df_all['别名'] = rec_df_all[['姓名','别名']].apply(lambda x: '，'.join(x.to_list()) if not check_NA_values(x[1]) else x[0], axis=1)
    rec_df_all['姓名'] = rec_df_all['文件名'].to_list()
    rec_df_all['别名'] = rec_df_all[['姓名','别名']].apply(lambda x: remove_pinyin_name(x['别名'], x['姓名']), axis=1)
    rec_df_all['籍贯'] = rec_df_all.apply(lambda x: standardize_address(x['籍贯'], x['UID']), axis=1)
    rec_df_all['生卒年或个人活动日期'] = rec_df_all.apply(lambda x: standardize_date(x['生卒年或个人活动日期'], x['UID']), axis=1)
    rec_df_all['发表的著作实体'] = rec_df_all['发表的著作实体'].apply(remove_cidian)
    rec_df_all['发表的著作实体'] = rec_df_all['发表的著作实体'].apply(standardize_literature)
    rec_df_all['活动领域'] = rec_df_all['活动领域'].apply(standardize_literature)
    rec_df_all['受教育机构'] = rec_df_all['受教育机构'].apply(standardize_literature)
    rec_df_all['在职单位'] = rec_df_all['在职单位'].apply(standardize_literature)
    rec_df_all['职业'] = rec_df_all['职业'].apply(standardize_literature)
    rec_df_all = merge_records(rec_df_all, id_field=['文件名', '控制号'])
    
    # rec_df_comb = rec_df_all[['生卒年或个人活动日期', '发表的著作实体', '籍贯', '活动领域', '职业', '受教育机构', '在职单位', 'UID', '文件名']]
    # rec_df_comb['姓名和别名'] = rec_df_all[['姓名', '别名']].apply(lambda x: x[0] + '；' + (x[1] if not check_NA_values(x[1]) else x[0]), axis=1)


    print('数据预处理完成')

    print(f'使用{args.model}进行中文关键词/句子嵌入')
    field_cutoffs = {
        '生卒年或个人活动日期': -4, '发表的著作实体': -8, '籍贯': -3, '活动领域': 0.05,
        '受教育机构': -4, '在职单位': -4, '别名': -2, '职业': 0.05 # 最多只有5%的记录被认为匹配
    }
    field_weights = {
        '生卒年或个人活动日期': 0.2, '发表的著作实体': 0.2, '籍贯': 0.2, '活动领域': 0.15,
        '受教育机构': 0.15, '在职单位': 0.15, '别名': 0.05, '职业': 0.05
    }

    # deduper = Deduplication(model_name='distiluse-base-multilingual-cased-v1', field_weights=field_weights)
    deduper = Deduplication(model_name=args.model, field_weights=field_weights, device=DEVICE)

    # print(rec_df_all['文件名'].unique(), rec_df_all['文件名'].unique().shape)
    # dataset_name = '刘伟'
    # plot_flag = False
    for dataset_name in tqdm.tqdm(rec_df_all['文件名'].unique()):
        # break
        print(f"Processing {dataset_name}")        
    
        # 提示词+全字段嵌入去重
        if args.allfield_embed:
            if not os.path.exists(f'./{args.tag}/提示词+全字段嵌入/{dataset_name}_提示词+全字段嵌入.xlsx'):
                t0 = time.time()
                os.makedirs(f'./{args.tag}/提示词+全字段嵌入', exist_ok=True)
                rec_df = rec_df_all.loc[rec_df_all['文件名'] == dataset_name]
                # prompt = deduper.build_prompt()
                # uid, text = deduper.build_infotext(rec_df)
                rec_embs = deduper.build_embs(rec_df)
                rec_dist = deduper.compute_emb_dist(rec_embs)
                if args.plot:
                    deduper.plot_overview(rec_dist, labels=rec_df['UID'].values, figsize=(20, 10), path=f'./{"demo" if args.demo else (args.tag + "/提示词+全字段嵌入")}/{dataset_name}_提示词+全字段嵌入_overview.png')
                group_ids = deduper.deduplicate(rec_dist, method='single', ps=[0.01, 0.05, 0.10, 0.20])
                for p, group_id in group_ids.items():
                    rec_df[p] = group_id
                
                df = pd.DataFrame(group_ids).apply(lambda x: (~pd.isna(x)).sum()).reset_index()
                df.columns = ['列','#重复条目数']
                print(f'提示词+全字段嵌入：{time.time() - t0:.2f}秒')
                print(df)
                
                rec_df.to_excel(f'./{"demo" if args.demo else (args.tag + "/提示词+全字段嵌入")}/{dataset_name}_提示词+全字段嵌入.xlsx', index=False)
            else:
                print(f"Skip 提示词+全字段嵌入/{dataset_name}: Already processed")
        
        
        # 单字段嵌入加权去重
        if args.singlefield_embed:
            if not os.path.exists(f'./{args.tag}/单字段嵌入加权/{dataset_name}_单字段嵌入加权.xlsx'):
                t0 = time.time()
                os.makedirs(f'./{args.tag}/单字段嵌入加权', exist_ok=True)
                rec_df = rec_df_all.loc[rec_df_all['文件名'] == dataset_name]
                rec_emb_dict = deduper.build_embs_each_field(
                    rec_df, 
                    ['生卒年或个人活动日期', '发表的著作实体', '籍贯', '活动领域', '职业', '受教育机构', '在职单位', '别名']
                )
                rec_dist_dict = deduper.compute_emb_dist(rec_emb_dict)
                rec_dist = deduper.merge_dist(rec_dist_dict, weight_dict=field_weights, method='arithmetic')
                if args.plot:
                    deduper.plot_overview(rec_dist, labels=rec_df['UID'].values, figsize=(20, 10), path=f'{"demo" if args.demo else (args.tag + "/单字段嵌入加权")}/{dataset_name}_单字段嵌入加权_overview.png')
                group_ids = deduper.deduplicate(rec_dist, method='single', ps=[0.01, 0.05, 0.10, 0.20])
                for p, group_id in group_ids.items():
                    rec_df[p] = group_id
                    
                df = pd.DataFrame(group_ids).apply(lambda x: (~pd.isna(x)).sum()).reset_index()
                df.columns = ['列','#重复条目数']
                print(f'单字段嵌入加权：{time.time() - t0:.2f}秒')
                print(df)
                    
                rec_df.to_excel(f'./{"demo" if args.demo else (args.tag + "/单字段嵌入加权")}/{dataset_name}_单字段嵌入加权.xlsx', index=False)
            else:
                print(f"Skip 单字段嵌入加权/{dataset_name}: Already processed")
        
        
        # 混合去重（硬汇总）
        if args.mixed_hard:
            if not os.path.exists(f'./{args.tag}/子字符串+语义匹配计数/{dataset_name}_子字符串+语义匹配计数.xlsx'):
                t0 = time.time()
                os.makedirs(f'./{args.tag}/子字符串+语义匹配计数', exist_ok=True)
                rec_df = rec_df_all.loc[rec_df_all['文件名'] == dataset_name]
                
                field_cutoffs = {
                    '生卒年或个人活动日期': -4, '发表的著作实体': -8, '籍贯': -3, '活动领域': 0.05,
                    '受教育机构': -4, '在职单位': -4, '别名': -2, '职业': 0.05 # 最多只有5%的记录被认为匹配
                }
                rec_emb_dict = deduper.build_embs_each_field(
                    rec_df, 
                    ['活动领域', '职业']
                )
                rec_emb_dist_dict = deduper.compute_emb_dist(rec_emb_dict, field_cutoffs=field_cutoffs)
                rec_subseq_dist_dict = deduper.pairwise_common_subseq(
                    rec_df[[k for k,v in field_cutoffs.items() if v and (v < 0)]].to_dict(orient='list'),
                    field_cutoffs
                )
                # 合并字典rec_emb_dist_dict和rec_subseq_len_dict
                rec_dist_dict = {**rec_emb_dist_dict, **rec_subseq_dist_dict}
                rec_dist = deduper.merge_dist(rec_dist_dict, method='top')
                if args.plot:
                    deduper.plot_overview(rec_dist - rec_dist.min(), labels=rec_df['UID'].values, figsize=(20, 10), path=f'{"demo" if args.demo else (args.tag + "/子字符串+语义匹配计数")}/{dataset_name}_子字符串+语义匹配计数_overview.png')
                group_ids = deduper.deduplicate(rec_dist, method='top', ks=[2, 3, 4, 5])
                for p, group_id in group_ids.items():
                    rec_df[p] = group_id
                    
                df = pd.DataFrame(group_ids).apply(lambda x: (~pd.isna(x)).sum()).reset_index()
                df.columns = ['列','#重复条目数']
                print(f'子字符串+语义匹配计数：{time.time() - t0:.2f}秒')
                print(df)
                
                rec_df.to_excel(f'./{"demo" if args.demo else (args.tag + "/子字符串+语义匹配计数")}/{dataset_name}_子字符串+语义匹配计数.xlsx', index=False)
            else:
                print(f"Skip 子字符串+语义匹配计数/{dataset_name}: Already processed")

        
        # 混合去重（软汇总）
        if args.mixed_soft:
            if not os.path.exists(f'./{args.tag}/子字符串+语义距离加权/{dataset_name}_子字符串+语义距离加权.xlsx'):
                t0 = time.time()
                os.makedirs(f'./{args.tag}/子字符串+语义距离加权', exist_ok=True)
                rec_df = rec_df_all.loc[rec_df_all['文件名'] == dataset_name]
                rec_emb_dict = deduper.build_embs_each_field(
                    rec_df, 
                    ['活动领域', '职业']
                )
                rec_emb_dist_dict = deduper.compute_emb_dist(rec_emb_dict) # 软汇总没有阈值
                rec_subseq_dist_dict = deduper.pairwise_common_subseq(
                    rec_df[[k for k,v in field_cutoffs.items() if v and (v < 0)]].to_dict(orient='list'),
                    field_cutoffs
                )
                rec_dist = deduper.merge_dist(rec_dist_dict, weight_dict=field_weights, method='arithmetic')
                if args.plot:
                    deduper.plot_overview(rec_dist, labels=rec_df['UID'].values, figsize=(20, 10), path=f'{"demo" if args.demo else (args.tag + "/子字符串+语义距离加权")}/{dataset_name}_子字符串+语义距离加权_overview.png')
                group_ids = deduper.deduplicate(rec_dist, method='single', ps=[0.01, 0.05, 0.10, 0.20])
                for p, group_id in group_ids.items():
                    rec_df[p] = group_id
                    
                df = pd.DataFrame(group_ids).apply(lambda x: (~pd.isna(x)).sum()).reset_index()
                df.columns = ['列','#重复条目数']
                print(f'子字符串+语义距离加权：{time.time() - t0:.2f}秒')
                print(df)
                    
                rec_df.to_excel(f'./{"demo" if args.demo else (args.tag + "/子字符串+语义距离加权")}/{dataset_name}_子字符串+语义距离加权.xlsx', index=False)
            else:
                print(f"Skip 子字符串+语义距离加权/{dataset_name}: Already processed")
                
            
    print('Done')
