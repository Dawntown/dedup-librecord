import os
import json
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import re

def clean_text(text):
    """清理文本，移除多余的空白字符"""
    if isinstance(text, str):
        return re.sub(r'\s+', ' ', text.strip())
    return text

def extract_table_data(html_file):
    """从HTML文件中提取表格数据"""
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    tables = soup.find_all('table')
    
    all_data = []
    for table in tables:
        rows = table.find_all('tr')
        headers = [clean_text(th.get_text()) for th in rows[0].find_all('th')]
        
        for row in rows[1:]:
            cells = row.find_all('td')
            if len(cells) == len(headers):
                row_data = {headers[i]: clean_text(cell.get_text()) for i, cell in enumerate(cells)}
                all_data.append(row_data)
    
    return all_data

def process_author_data(data, output_dir):
    """处理作者数据并保存到对应的文件夹"""
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 按作者名字分组
    author_groups = {}
    for record in data:
        author_name = record.get('作者', '').strip()
        if author_name:
            if author_name not in author_groups:
                author_groups[author_name] = []
            author_groups[author_name].append(record)
    
    # 为每个作者创建文件夹并保存数据
    for author_name, records in author_groups.items():
        # 创建作者文件夹（使用安全的文件夹名）
        safe_name = re.sub(r'[\\/*?:"<>|]', '_', author_name)
        author_dir = output_path / safe_name
        author_dir.mkdir(exist_ok=True)
        
        # 保存作者信息为JSON文件
        output_file = author_dir / 'author_info.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        # 同时保存为CSV文件以便于查看
        df = pd.DataFrame(records)
        csv_file = author_dir / 'author_info.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

def main():
    # 设置输入和输出目录
    input_dir = 'raw/rawhtml-50'
    output_dir = 'ext/previous-50'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有HTML文件
    for html_file in Path(input_dir).glob('*.html'):
        print(f"处理文件: {html_file}")
        data = extract_table_data(html_file)
        process_author_data(data, output_dir)
    
    print("处理完成！")

if __name__ == '__main__':
    main() 