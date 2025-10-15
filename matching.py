import csv
import logging
import optparse
import os
import re
import pandas as pd
import numpy as np
import glob
import tqdm
import re

import dedupe
from unidecode import unidecode
import collections

def check_NA_values(s):
    if "未知" in str(s) or "未提及" in str(s) or str(s) == "无":
        return True
    if ('中国' in str(s)) and ('辞典' in str(s)):
        return True
    if str(s).isspace():
        return True
    return False
    
def contains_chinese_regex(string):
    """
    使用正则表达式判断字符串中是否包含中文字符（基本汉字范围）
    """
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(string))

def split_keywords(s):
    return re.split(r'[;；,，、\s]', str(s))
    
rename_dict = {
    '生卒年或个人活动日期': '生卒年或个人活动日期',
    '生卒年（活动日期）': '生卒年或个人活动日期',
    '生卒年': '生卒年或个人活动日期',
    '生卒年/个人活动日期': '生卒年或个人活动日期',
}

    
fname_list = glob.glob('./entity_matching/*.xlsx')
rec_df_all = pd.concat([pd.read_excel(fname).rename(columns=rename_dict).assign(文件名=fname.split('/')[-1].split('.')[0]) for fname in tqdm.tqdm(fname_list)], axis=0)















names = pd.Series(collections.Counter([s for s in rec_df_all['姓名'].values.flatten()])).sort_values(ascending=False)
aliases = pd.Series(collections.Counter([s for s in rec_df_all['别名'].values.flatten()])).sort_values(ascending=False)
people = pd.Series(collections.Counter([s for s in rec_df_all['民族'].values.flatten()])).sort_values(ascending=False)
nationalities = pd.Series(collections.Counter([s for s in rec_df_all['国籍'].values.flatten()])).sort_values(ascending=False)
gender = pd.Series(collections.Counter([s for s in rec_df_all['性别'].values.flatten()])).sort_values(ascending=False)
education = pd.Series(collections.Counter([s for s in rec_df_all['受教育机构'].values.flatten()])).sort_values(ascending=False)
affiliation = pd.Series(collections.Counter([s for s in rec_df_all['在职单位'].values.flatten()])).sort_values(ascending=False)
birth_death = pd.Series(collections.Counter([s for s in rec_df_all['生卒年或个人活动日期'].values.flatten()])).sort_values(ascending=False)
occupation = pd.Series(collections.Counter([s for s in rec_df_all['职业'].values.flatten()])).sort_values(ascending=False)
fields = pd.Series(collections.Counter([s for s in rec_df_all['活动领域'].values.flatten()])).sort_values(ascending=False)
regions = pd.Series(collections.Counter([s for s in rec_df_all['籍贯'].values.flatten()])).sort_values(ascending=False)


names.to_csv('./des/names.csv', sep='\t')
aliases.to_csv('./des/aliases.csv', sep='\t')
people.to_csv('./des/people.csv', sep='\t')
nationalities.to_csv('./des/nationalities.csv', sep='\t')
gender.to_csv('./des/gender.csv', sep='\t')
education.to_csv('./des/education.csv', sep='\t')
affiliation.to_csv('./des/affiliation.csv', sep='\t')
birth_death.to_csv('./des/birth_death.csv', sep='\t')
occupation.to_csv('./des/occupation.csv', sep='\t')
fields.to_csv('./des/fields.csv', sep='\t')
regions.to_csv('./des/regions.csv', sep='\t')

sel_df = rec_df_all.loc[rec_df_all['生卒年或个人活动日期'].apply(lambda x: False if check_NA_values(x) else contains_chinese_regex(x))]
sel_df
sel_df['文件名'].value_counts()


from GeocodingCHN import Geocoding
geocoding = Geocoding()


def check_address(x):
    if check_NA_values(x):
        return False
    try:
        loc = geocoding.normalizing(x)
        return False
    except:
        return True


sel_df = rec_df_all.loc[rec_df_all['籍贯'].apply(check_address)]
sel_df['文件名'].value_counts()
sel_df.sort_values('文件名')