import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download
import time

t0 = time.time()
for attempt in range(8):
    try:
        path = snapshot_download(
            repo_id='rezzzq/RSCD-1million',
            repo_type='dataset',
            allow_patterns=['test_50k/*'],
            local_dir='c:/Users/Rivan/Projects/AI_Grand_Prix/data/_hf_snapshot',
            max_workers=3,
        )
        print("downloaded to", path, "in", time.time()-t0, "sec")
        break
    except Exception as e:
        print(f"attempt {attempt} failed: {e}")
        time.sleep(20 * (attempt + 1))
else:
    print("all attempts failed")
