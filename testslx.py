import csv
import os
import time
import numpy as np
import sglang as sgl
from sglang.srt.conversation import chat_templates
import psutil
import torch
import gc
from glob import glob
from threading import Thread
import pynvml
from calflops import calculate_flops
from transformers import AutoTokenizer

# Set PyTorch environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure model paths and output directory
# MODEL_PATHS = glob('/root/autodl-tmp/model_save/llama3_8b_sgl/sgl*')
MODEL_PATHS=[]
MODEL_PATHS.append('/root/autodl-tmp/model_save/llama3_8b_sgl/sgl_quant_model_awq_b4')
LOG_DIR = "/root/autodl-tmp/sglang/performance_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_vram_usage(baseline_vram=0.0):
    """Get VRAM usage in MB for the current GPU using pynvml, relative to baseline."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used = mem_info.used / 1024**2  # Convert bytes to MB
        vram_delta = max(0.0, vram_used - baseline_vram)  # Subtract baseline
        print(f"Debug: VRAM used (pynvml): {vram_used:.2f} MB, Delta: {vram_delta:.2f} MB")
        pynvml.nvmlShutdown()
        return vram_delta
    except Exception as e:
        print(f"Debug: Error getting VRAM with pynvml: {e}")
        return 0.0

def estimate_flops(model_stats, completion_time, token_count):
    """Estimate FLOPs based on model parameters and inference time (legacy method)."""
    params = getattr(model_stats, 'params', 8e9)  # Default to 8B params for LLaMA-3 8B
    if isinstance(params, str):
        params = 8e9  # Fallback if params is not numeric
    flops = 2 * params * token_count
    flops_per_second = flops / completion_time if completion_time > 0 else 0
    return flops, flops_per_second

def calculate_model_flops(model, tokenizer, batch_size=1, seq_length=128):
    """Calculate FLOPs using calflops."""
    try:
        flops, macs, params = calculate_flops(
            model=model,
            input_shape=(batch_size, seq_length),
            transformer_tokenizer=tokenizer,
            output_precision=2
        )
        print(f"Debug: Calflops FLOPs: {flops:.2f}, MACs: {macs:.2f}, Params: {params:.2f}")
        return flops
    except Exception as e:
        print(f"Debug: Error calculating FLOPs with calflops: {e}")
        return 0.0

def calculate_metrics(results):
    """Calculate average and standard deviation for metrics."""
    ttfts = [r["ttft"] for r in results]
    completion_times = [r["completion_time"] for r in results]
    tokens_per_sec = [r["tokens_per_second"] for r in results]
    vram_peaks = [r["vram_peak"] for r in results]
    flops = [r["flops"] for r in results]
    flops_per_sec = [r["flops_per_second"] for r in results]
    
    return {
        "avg_ttft": np.mean(ttfts) if ttfts else 0.0,
        "std_ttft": np.std(ttfts) if ttfts else 0.0,
        "avg_completion_time": np.mean(completion_times) if completion_times else 0.0,
        "std_completion_time": np.std(completion_times) if completion_times else 0.0,
        "avg_tokens_per_second": np.mean(tokens_per_sec) if tokens_per_sec else 0.0,
        "std_tokens_per_second": np.std(tokens_per_sec) if tokens_per_sec else 0.0,
        "peak_vram_used_mb": max(vram_peaks) if vram_peaks else 0.0,
        "avg_flops": np.mean(flops) if flops else 0.0,
        "std_flops": np.std(flops) if flops else 0.0,
        "avg_flops_per_second": np.mean(flops_per_sec) if flops_per_sec else 0.0,
        "std_flops_per_second": np.std(flops_per_sec) if flops_per_sec else 0.0
    }

def measure_ttft(llm, prompt, sampling_params, vram_samples, baseline_vram):
    """Measure Time to First Token using streaming and collect VRAM samples."""
    start_time = time.time()
    ttft = 0.0
    
    try:
        # Attempt streaming generation
        stream_iter = llm.generate([prompt], sampling_params, stream=True)
        for chunk in stream_iter:
            print(f"Debug: Stream chunk: {chunk}")
            vram_samples.append(get_vram_usage(baseline_vram))  # Sample during streaming
            if 'text' in chunk and chunk['text']:
                ttft = time.time() - start_time
                break
        if ttft == 0.0:
            print("Debug: No valid text received in stream")
    except Exception as e:
        print(f"Debug: Streaming error: {e}")
        # Fallback to non-streaming
        try:
            output = llm.generate([prompt], sampling_params)[0]
            print(f"Debug: Non-stream output: {output}")
            vram_samples.append(get_vram_usage(baseline_vram))  # Sample during fallback
            ttft = time.time() - start_time
        except Exception as e:
            print(f"Debug: Non-streaming error: {e}")
            return None
    
    return ttft

def monitor_vram(vram_samples, stop_flag, baseline_vram):
    """Continuously monitor VRAM usage in a separate thread."""
    while not stop_flag[0]:
        vram_samples.append(get_vram_usage(baseline_vram))
        time.sleep(0.01)  # Sample every 10ms

def cleanup_processes():
    """Kill residual SGLang-related processes to free GPU memory."""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.pid == current_pid:
                continue
            cmdline = proc.info['cmdline'] or []
            if any('sglang' in str(arg).lower() for arg in cmdline):
                print(f"Terminating residual SGLang process: PID {proc.pid}, Command: {cmdline}")
                proc.terminate()
                proc.wait(timeout=5)  # Wait for graceful termination
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
            print(f"Error terminating process {proc.pid}: {e}")
            continue

if __name__ == "__main__":
    # Define prompts for inference
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Set sampling parameters
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95
    }

    # Load a default tokenizer (fallback for LLaMA models)
    try:
        default_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    except Exception as e:
        print(f"Debug: Error loading default tokenizer: {e}")
        default_tokenizer = None

    # Collect all results for CSV
    all_results = []

    for model_path in MODEL_PATHS:
        # Extract model type from last folder name
        model_type = os.path.basename(model_path)
        print(f"\nRunning experiments for model: {model_path} ({model_type})")
        
        # Measure baseline VRAM before model loading
        baseline_vram = get_vram_usage()
        print(f"Debug: Baseline VRAM before model loading: {baseline_vram:.2f} MB")

        # Initialize the SGLang engine with retry logic
        llm = None
        for attempt in range(3):
            try:
                print(f"Attempt {attempt + 1} to initialize model {model_path}")
                llm = sgl.Engine(model_path=model_path)
                break
            except Exception as e:
                print(f"Error initializing model {model_path} on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    cleanup_processes()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    time.sleep(2)  # Wait before retry
                else:
                    print(f"Failed to initialize model {model_path} after 3 attempts. Skipping.")
                    break
        if llm is None:
            continue

        # Attempt to access the underlying model and tokenizer
        model = None
        tokenizer = default_tokenizer
        try:
            # Access the model (may vary based on SGLang version)
            model = llm.model_runner.model if hasattr(llm, 'model_runner') and hasattr(llm.model_runner, 'model') else None
            # Access the tokenizer (if available)
            tokenizer = llm.tokenizer if hasattr(llm, 'tokenizer') else default_tokenizer
            if model is None or tokenizer is None:
                print("Debug: Could not access model or tokenizer from SGLang engine. Using default tokenizer.")
        except Exception as e:
            print(f"Debug: Error accessing model/tokenizer: {e}")

        # Calculate FLOPs using calflops
        calflops_flops = 0.0
        if model is not None and tokenizer is not None:
            calflops_flops = calculate_model_flops(model, tokenizer, batch_size=1, seq_length=128)
        else:
            print("Debug: Skipping calflops calculation due to missing model or tokenizer.")

        # Run inference and collect metrics
        num_runs = len(prompts)
        valid_results = []

        for prompt in prompts:
            # Track VRAM usage during inference
            vram_samples = [get_vram_usage(baseline_vram)]
            stop_flag = [False]  # Thread-safe flag to stop VRAM monitoring
            vram_thread = Thread(target=monitor_vram, args=(vram_samples, stop_flag, baseline_vram))
            vram_thread.start()

            start_time = time.time()

            # Measure TTFT
            ttft = measure_ttft(llm, prompt, sampling_params, vram_samples, baseline_vram)
            if ttft is None:
                print(f"Error measuring TTFT for prompt '{prompt}': Skipping")
                stop_flag[0] = True
                vram_thread.join()
                continue

            # Generate output (non-streaming for completion time)
            try:
                output = llm.generate([prompt], sampling_params)[0]
                vram_samples.append(get_vram_usage(baseline_vram))  # Sample during generation
            except Exception as e:
                print(f"Error generating output for prompt '{prompt}': {e}")
                stop_flag[0] = True
                vram_thread.join()
                continue
            end_time = time.time()
            vram_samples.append(get_vram_usage(baseline_vram))  # Sample after generation

            # Stop VRAM monitoring
            stop_flag[0] = True
            vram_thread.join()

            # Calculate metrics
            completion_time = end_time - start_time
            token_count = len(output['text'].split())  # Approximate token count
            tokens_per_second = token_count / completion_time if completion_time > 0 else 0
            peak_vram_used_mb = max(vram_samples) if vram_samples else 0.0

            valid_results.append({
                "ttft": ttft,
                "completion_time": completion_time,
                "tokens_per_second": tokens_per_second,
                "vram_peak": peak_vram_used_mb,
                "flops": estimate_flops(None, completion_time, token_count)[0],
                "flops_per_second": estimate_flops(None, completion_time, token_count)[1]
            })

            # Print results
            print("===============================")
            print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
            print(f"Debug: Peak VRAM for prompt: {peak_vram_used_mb:.2f} MB")

        # Calculate aggregated metrics
        metrics = calculate_metrics(valid_results)

        # Prepare results dictionary
        results = {
            "model_type": model_type,
            "avg_ttft": metrics["avg_ttft"],
            "std_ttft": metrics["std_ttft"],
            "avg_completion_time": metrics["avg_completion_time"],
            "std_completion_time": metrics["std_completion_time"],
            "avg_tokens_per_second": metrics["avg_tokens_per_second"],
            "std_tokens_per_second": metrics["std_tokens_per_second"],
            "peak_vram_used_mb": metrics["peak_vram_used_mb"],
            "avg_flops": metrics["avg_flops"],
            "std_flops": metrics["std_flops"],
            "avg_flops_per_second": metrics["avg_flops_per_second"],
            "std_flops_per_second": metrics["std_flops_per_second"],
            "calflops_flops": calflops_flops,
            "successful_runs": len(valid_results),
            "total_runs": num_runs
        }

        all_results.append(results)

        # Clear GPU memory and terminate processes
        try:
            del llm  # Explicitly delete the engine
            llm = None
            cleanup_processes()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()  # Force garbage collection
            print(f"GPU memory and processes cleared for {model_path}")
        except Exception as e:
            print(f"Error clearing GPU memory or processes for {model_path}: {e}")
        
        print(f"Completed experiments for {model_path}")

    # Write all results to a single CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(LOG_DIR, f"performance_batch_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    print(f"All performance metrics written to {csv_file}")