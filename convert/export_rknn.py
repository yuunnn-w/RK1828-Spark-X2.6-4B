import os
from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG

# 1.2.0 版 Spark-X2.5/X2.6 编译脚本（基于官方 examples/Spark_X2_5/python/export_rknn.py，
# 差别：GRQ 产物不再传 dataset；所有 LLM 配置项可用环境变量覆盖；可在独立 CWD 下运行）
env = os.environ
MODEL_SIZE   = env.get("SPARK_MODEL_SIZE", "4B")
ONNX_PATH    = env["SPARK_ONNX"]
CONFIG_PKL   = env["SPARK_CONFIG"]
RKNN_PATH    = env["SPARK_RKNN"]
PLATFORM     = env.get("SPARK_PLATFORM", "rk1828")
CTX          = int(env.get("SPARK_CTX", "4096"))
SEQ          = int(env.get("SPARK_SEQ", "128"))
CORES        = int(env.get("SPARK_CORES", "8"))
SLIDING      = int(env.get("SPARK_SLIDING", "512"))
KV_DTYPE     = env.get("SPARK_KV_DTYPE", "Float16")          # Float16 / Int8_to_F16 / Int4_to_F16
KV_METHOD    = env.get("SPARK_KV_METHOD", "Normal")          # Normal / GroupQuant
KV_GROUP     = int(env.get("SPARK_KV_GROUP", "32"))
KV_RESID     = int(env.get("SPARK_KV_RESID", "32"))
HOST_ROPE    = env.get("SPARK_HOST_ROPE", "1") == "1"
KV_RT        = env.get("SPARK_KV_RT", "0") == "1"        # True -> 全注意力层 kvcache_buffer_len='real_time'（按需分配，不预占满 ctx）
KV_BUF       = env.get("SPARK_KV_BUF", "")               # 非空 -> 全注意力层 kvcache_buffer_len 用该整数（< ctx），配合 max_position_embeddings 探分页 KV
DO_QUANT     = env.get("SPARK_DO_QUANT", "1") == "1"     # GRQ 已烘进 ONNX，但权重打包成 int4 需要 build(do_quantization=True, dataset=None)
DATASET      = env.get("SPARK_DATASET", "")

print(f"[cfg] size={MODEL_SIZE} ctx={CTX} seq={SEQ} cores={CORES} kv={KV_DTYPE}/{KV_METHOD}/g{KV_GROUP}/r{KV_RESID} "
      f"host_rope={HOST_ROPE} rt={KV_RT} do_quant={DO_QUANT} sliding={SLIDING} platform={PLATFORM}")

rknn = RKNN(verbose=True)
my_config = DEFAULT_RKNN_LLM_CONFIG.copy()

attn_config = {
    'mask_name'             : 'attention_mask',
    'position_name'         : 'position_ids',
    'kvcache_buffer_len'    : (int(KV_BUF) if KV_BUF else CTX),
    'max_position_embeddings': CTX,
    'kvcache_dtype'         : KV_DTYPE,
    'kvcache_store_method'  : KV_METHOD,
    'kvcache_group_size'    : KV_GROUP,
    'kvcache_residual_depth': KV_RESID,
    'sliding_window_size'   : -1,
    'attention_type'        : 'FullAttention',
    'position_embeddings_host_storage': HOST_ROPE,
}
attn_sld_config = {
    'mask_name'             : 'attention_mask_1',
    'position_name'         : 'position_ids_1',
    'kvcache_buffer_len'    : SLIDING,
    'max_position_embeddings': CTX,
    'kvcache_dtype'         : KV_DTYPE,
    'kvcache_store_method'  : KV_METHOD,
    'kvcache_group_size'    : KV_GROUP,
    'kvcache_residual_depth': KV_RESID,
    'sliding_window_size'   : SLIDING,
    'attention_type'        : 'SlidingAttention',
    'position_embeddings_host_storage': HOST_ROPE,
}
my_config['attention_config'] = [attn_sld_config, attn_config]

# --- llm_head / vocab 修正（默认 llm_head 的 device='RK1820' 对 rk1828 匹配失败，
#     日志会打 "Match llm_head with 'auto' failed"，输出头退化成 w4a16）---
HEAD_NAME  = env.get("SPARK_HEAD_NAME", "")      # 一般留空（用 auto）；官方示例里 name 不是节点名而是 auto
HEAD_DEV   = env.get("SPARK_HEAD_DEVICE", "")    # 例：RK1828 —— 官方默认写死 RK1820，对 rk1828 目标会 "Match llm_head with 'auto' failed"
HEAD_DTYPE = env.get("SPARK_HEAD_DTYPE", "")     # 例：w6a16
HEAD_METH  = env.get("SPARK_HEAD_METHOD", "group32")
TIE_EMB    = env.get("SPARK_TIE_EMB", "")        # 1 -> vocab[0]['tie_embeddings']=True（Spark tie_word_embeddings=True）
if HEAD_NAME or HEAD_DTYPE or HEAD_DEV:
    hd = dict(my_config['llm_head'][0])
    if HEAD_DEV:  hd['device'] = HEAD_DEV
    else:         hd['device'] = PLATFORM.upper()
    if HEAD_NAME: hd['name'] = HEAD_NAME
    if HEAD_DTYPE: hd['quantized_dtype'] = HEAD_DTYPE; hd['quantized_method'] = HEAD_METH
    my_config['llm_head'] = [hd]
if TIE_EMB:
    vv = dict(my_config['vocab'][0]); vv['tie_embeddings'] = (TIE_EMB == "1")
    my_config['vocab'] = [vv]
print('[llm_head]', my_config['llm_head'], ' [vocab]', my_config['vocab'])

print('LLM config is:', my_config)

dynamic_input = [[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1]],
                 [[1, SEQ], [1, SEQ], [1, SEQ], [1, SEQ], [1, SEQ], [1]]]

rknn.config(target_platform=PLATFORM, dynamic_input=dynamic_input,
            quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
            llm_config=my_config, core_num=CORES)

print('--> Loading model')
ret = rknn.load_llm(model=ONNX_PATH, config=CONFIG_PKL, seq=[1, SEQ])
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

print('--> Building model')
if DO_QUANT:
    ret = rknn.build(do_quantization=True, dataset=(DATASET or None))
else:
    ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

print('--> Export RKNN model')
ret = rknn.export_rknn(RKNN_PATH, save_ctx=True)
if ret != 0:
    print('Export rknn failed!')
    exit(ret)
print('done')
rknn.release()
print("RKNN_EXPORTED", RKNN_PATH)
