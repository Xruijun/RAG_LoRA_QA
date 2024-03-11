from modelscope import snapshot_download
# chatglm3-6b模型下载
model_dir = snapshot_download(
    'ZhipuAI/chatglm3-6b', cache_dir='chatglm3-6b'
)