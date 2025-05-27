# dedup-librecord

这是一个用于图书馆作者记录去重的工具，主要使用语言模型进行语义匹配。该工具支持多种去重策略，包括全字段嵌入、单字段嵌入加权、子字符串匹配等多种方法。

## 功能特点

- 支持多种去重策略：
  - 提示词+全字段嵌入
  - 单字段嵌入加权
  - 子字符串+语义匹配计数
  - 子字符串+语义距离加权
- 支持中文文本处理：
  - 拼音转换
  - 地理编码
  - 文本相似度计算
- 可视化分析：
  - 距离分布直方图
  - 层次聚类树状图
- 灵活的参数配置：
  - 可自定义字段权重
  - 可调整匹配阈值
  - 支持多种聚类方法

## 环境要求

```bash
# 基础环境
python=3.11.9
numpy=2.0.1
pandas=2.2.2
scipy=1.14.0
matplotlib=3.9.1
seaborn=0.13.2

# 通过pip安装的包
sentence-transformers==3.3.1
torch==2.3.1
torch-directml==0.2.4.dev240815
GeocodingCHN==1.4.5
Pinyin2Hanzi==0.1.1
pypinyin==0.53.0
```

## 使用方法

1. 安装环境：
```bash
conda env create -f env_dedup-librecord.yaml
```

2. 运行去重：
```bash
python main.py --model distiluse-base-multilingual-cased-v1 [其他参数] # 参考run.sh
```

## 主要参数说明

- `--model`: 使用的语言模型名称
- `--allfield_embed`: 是否使用全字段嵌入
- `--singlefield_embed`: 是否使用单字段嵌入加权
- `--mixed_hard`: 是否使用子字符串+语义匹配计数
- `--mixed_soft`: 是否使用子字符串+语义距离加权
- `--plot`: 是否生成可视化结果
- `--demo`: 是否为演示模式
- `--merge_cid`: 是否合并控制号相同的记录
- `--input_dir`: 输入文件夹路径
- `--tag`: 输出文件夹路径

## 输出结果

程序会在指定目录下生成以下文件：
- 去重后的Excel文件
- 距离分布直方图
- 层次聚类树状图

## 注意事项

1. 确保输入数据格式正确，包含必要的字段
2. 对于大规模数据，建议使用GPU加速
3. 可以根据实际需求调整字段权重和匹配阈值