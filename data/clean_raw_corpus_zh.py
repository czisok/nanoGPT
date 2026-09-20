import json
import re
import opencc
import os
from tqdm import tqdm

cc = opencc.OpenCC('t2s')  # 繁转简

def clean_text(text):
    # 1. 繁转简
    text = cc.convert(text)
    # 2. 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 3. 去除多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 4. 去除特殊控制字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 5. 统一全角半角标点（可选）
    text = text.replace('，', '，').replace('。', '。').replace('？', '？').replace('！', '！')
    # 6. 过滤过短文本
    if len(text.strip()) < 20:
        return None
    return text.strip()

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc=f"处理 {input_file}"):
            item = json.loads(line)
            # 适配不同数据集的文本字段
            text = item.get('text', item.get('content', ''))
            cleaned = clean_text(text)
            if cleaned:
                f_out.write(cleaned + '\n\n')  # 文档间用双换行分隔

if __name__ == '__main__':
    data_dir = "/home/zhangbo.999/jupyter_workspace/dataset/llm_dataset/corpus_raw"
    minidata_input = os.path.join(data_dir, "minimind_data", "pretrain_t2t.jsonl")
    minidata_output = os.path.join(data_dir, "minimind_data", "minimind_cleaned.txt")
    final_output = os.path.join(data_dir, "all_corpus.txt")
    process_jsonl(minidata_input, minidata_output)
    # process_jsonl('pretrain_t2t_mini.jsonl', 'minimind_cleaned_mini.txt')    
    # 合并所有语料
    with open(final_output, 'w', encoding='utf-8') as out:
        # for fname in ['wiki_cleaned.txt', 'minimind_cleaned.txt']:
        for fname in [minidata_output]:
            with open(fname, 'r', encoding='utf-8') as f:
                out.write(f.read())
    print(f"语料清洗完成，总文件：{final_output}")  
