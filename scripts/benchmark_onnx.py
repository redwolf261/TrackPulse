"""
Step 4: benchmark the actual ONNX artifact - CPU (and GPU if available) latency,
FPS, file size, and peak memory (best-effort).
"""
import os, time, json
import numpy as np
import onnxruntime as ort
import psutil

ONNX_PATH = 'c:/Users/Rivan/Projects/AI_Grand_Prix/models/trackpulse_classifier.onnx'
EXP_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp00_rscd_baseline'

file_size_bytes = os.path.getsize(ONNX_PATH)
print(f"ONNX file size: {file_size_bytes} bytes ({file_size_bytes/1024/1024:.2f} MB)")

dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

def benchmark_provider(provider_list, label, n_warmup=10, n_runs=200):
    try:
        sess = ort.InferenceSession(ONNX_PATH, providers=provider_list)
    except Exception as e:
        print(f"\n[{label}] FAILED to create session: {e}")
        return None
    active = sess.get_providers()
    if provider_list[0] not in active:
        print(f"\n[{label}] requested provider {provider_list[0]} not active (got {active}) - skipping, likely unsupported on this machine")
        return None
    input_name = sess.get_inputs()[0].name

    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss

    # warmup
    for _ in range(n_warmup):
        sess.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    mem_after = proc.memory_info().rss
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean_lat = latencies.mean()
    fps = 1000.0 / mean_lat

    print(f"\n=== [{label}] n_warmup={n_warmup} n_runs={n_runs} active_providers={active} ===")
    print(f"latency ms: p50={p50:.3f} p95={p95:.3f} p99={p99:.3f} mean={mean_lat:.3f}")
    print(f"FPS (from mean latency): {fps:.1f}")
    print(f"process RSS before: {mem_before/1024/1024:.1f} MB, after: {mem_after/1024/1024:.1f} MB, delta: {(mem_after-mem_before)/1024/1024:.1f} MB")

    return {
        'label': label,
        'active_providers': active,
        'n_warmup': n_warmup,
        'n_runs': n_runs,
        'latency_ms': {'p50': float(p50), 'p95': float(p95), 'p99': float(p99), 'mean': float(mean_lat)},
        'fps': float(fps),
        'process_rss_mb': {'before': mem_before/1024/1024, 'after': mem_after/1024/1024, 'delta': (mem_after-mem_before)/1024/1024},
    }

results = {'onnx_file_size_bytes': file_size_bytes, 'onnx_file_size_mb': file_size_bytes/1024/1024}

print("\n--- CPU benchmark ---")
cpu_result = benchmark_provider(['CPUExecutionProvider'], 'CPU')
results['cpu'] = cpu_result

print("\n--- GPU benchmark (CUDAExecutionProvider) ---")
try:
    gpu_result = benchmark_provider(['CUDAExecutionProvider', 'CPUExecutionProvider'], 'GPU (CUDA)')
    results['gpu'] = gpu_result
except Exception as e:
    print("GPU benchmark failed entirely:", e)
    results['gpu'] = None

with open(f'{EXP_DIR}/onnx_benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {EXP_DIR}/onnx_benchmark_results.json")
