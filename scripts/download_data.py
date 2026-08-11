from huggingface_hub import snapshot_download
import time

t0 = time.time()
path = snapshot_download(
    repo_id='rezzzq/RSCD-1million',
    repo_type='dataset',
    allow_patterns=['test_50k/*'],
    local_dir='c:/Users/Rivan/Projects/AI_Grand_Prix/data/_hf_snapshot',
    max_workers=8,
)
print("downloaded to", path, "in", time.time()-t0, "sec")
