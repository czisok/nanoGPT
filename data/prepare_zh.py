"""
中文语料预处理（cl100k_base 分词，适配大语料流式处理）
------------------------------------------------------------
修复点：
1. enc.encode() 没有 num_threads 参数；多线程分词需用批量接口
   enc.encode_ordinary_batch(文本列表, num_threads=8)
2. cl100k 词表大小 100,277（中文 id 可达 8 万+），超过 uint16 上限 65,535，
   必须用 uint32 存储，否则 id 会被静默截断成错误数据
3. 输出目录不存在时 tofile 会报错，先 makedirs
4. 语料 8GB+，一次性读入并构建 Python token 列表会爆内存，
   改为：分批读行 → 批量分词 → 流式追加写入磁盘
"""
import os
import numpy as np
import tiktoken
from tqdm import tqdm

# 加载分词器（cl100k_base：GPT-4 同款 BPE，中文支持远好于 gpt2）
enc = tiktoken.get_encoding("cl100k_base")
eot = enc.eot_token  # 100257，文档分隔符（如需要可在每篇文档间插入）

# 配置
# script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = "/home/zhangbo.999/jupyter_workspace/dataset/llm_dataset"
input_file = os.path.join(script_dir, "corpus_raw", "minimind_data", "minimind_cleaned.txt")
out_dir = os.path.join(script_dir, "minimind_data")
train_ratio = 0.95
block_size = 1024
LINES_PER_BATCH = 2000  # 每批送入多线程分词器的行数，可按内存调整

DTYPE = np.uint32  # cl100k 词表 100277 > 65535，必须 uint32（uint32 上限约 42 亿）


def iter_line_batches(path, batch_size):
    """按行分批读取大文件，避免一次性读入内存"""
    batch = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def main():
    os.makedirs(out_dir, exist_ok=True)
    all_bin_path = os.path.join(out_dir, "all.bin")

    # ---- 1. 分批多线程分词，流式写入 all.bin（uint32）----
    print("开始分词（cl100k_base，8 线程批量编码）...")
    total_tokens = 0
    with open(all_bin_path, 'wb') as fout:
        bar = tqdm(iter_line_batches(input_file, LINES_PER_BATCH), desc="encoding")
        for lines in bar:
            # 批量接口才支持 num_threads；encode_ordinary 不做特殊 token 处理，速度更快
            batch_ids = enc.encode_ordinary_batch(lines, num_threads=8)
            # 展平为一个 numpy 数组后直接写盘（不在内存中累积）
            arr = np.empty(sum(len(x) for x in batch_ids), dtype=DTYPE)
            offset = 0
            for ids in batch_ids:
                arr[offset:offset + len(ids)] = ids
                offset += len(ids)
            arr.tofile(fout)
            total_tokens += len(arr)
            bar.set_postfix(tokens=f"{total_tokens:,}")
    print(f"总 Token 数：{total_tokens:,}")

    # ---- 2. 按比例切分 train / val（用 memmap 读取，不占内存）----
    data = np.memmap(all_bin_path, dtype=DTYPE, mode='r')
    n = len(data)
    split_idx = int(n * train_ratio)
    train_path = os.path.join(out_dir, 'train.bin')
    val_path = os.path.join(out_dir, 'val.bin')
    print(f"切分中... train={split_idx:,}  val={n - split_idx:,}")
    data[:split_idx].tofile(train_path)  # memmap 切片经 page cache 流式写出
    data[split_idx:].tofile(val_path)
    del data

    # 中间文件可删（保留也方便重新切分）
    os.remove(all_bin_path)

    print(f"训练集：{split_idx:,} tokens -> {train_path}")
    print(f"验证集：{n - split_idx:,} tokens -> {val_path}")
    print("预处理完成，生成 train.bin / val.bin（dtype=uint32）")
    print()
    print("⚠️ 下游配套提醒：")
    print("  1) train.py 的 get_batch 中 np.memmap(..., dtype=np.uint16) 需改为 np.uint32")
    print("  2) 模型 vocab_size 需 ≥100277，建议配置 vocab_size=100352（128 对齐）")
    print("  3) sample.py 的解码需改用 cl100k_base（默认 fallback 是 gpt2 编码，词表不匹配）")


if __name__ == '__main__':
    main()
