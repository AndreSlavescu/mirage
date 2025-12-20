#!/usr/bin/env python3
"""
PDL Characterization Study:
Find the exact conditions where PDL helps vs hurts.

Key hypothesis: PDL helps when kernel launch overhead is significant
relative to kernel execution time.
"""
import torch
import mirage as mi
import numpy as np
import json
import argparse
from datetime import datetime


def get_gpu_info():
    props = torch.cuda.get_device_properties(0)
    return props.name, props.major * 10 + props.minor


def measure_kernel_launch_overhead():
    """Measure raw CUDA kernel launch overhead."""
    print("\n=== Kernel Launch Overhead Measurement ===")
    
    x = torch.randn(1, 1, device="cuda", dtype=torch.float16)
    
    for _ in range(100):
        _ = x + x
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    iterations = 10000
    start.record()
    for _ in range(iterations):
        _ = x + x
    end.record()
    torch.cuda.synchronize()
    
    total_ms = start.elapsed_time(end)
    per_kernel_us = (total_ms * 1000) / iterations
    
    print(f"Minimal kernel launch overhead: ~{per_kernel_us:.2f} us")
    return per_kernel_us


def sweep_tensor_size(cc: int):
    """Sweep tensor size to find the crossover point."""
    print("\n=== Tensor Size Sweep ===")
    print("Finding where PDL impact changes with tensor size...")
    
    results = []
    
    for size_exp in range(8, 20):
        size = 2 ** size_exp
        elements = size * 32
        
        def create_graph():
            g = mi.new_kernel_graph()
            X = g.new_input(dims=(size, 32), dtype=mi.float16)
            Y = g.new_input(dims=(size, 32), dtype=mi.float16)
            O = g.silu(X)
            O = g.add(O, Y)
            O = g.mul(O, Y)
            g.mark_output(O)
            return g
        
        inputs = [
            torch.randn(size, 32, dtype=torch.float16, device="cuda"),
            torch.randn(size, 32, dtype=torch.float16, device="cuda"),
        ]
        
        try:
            g1 = create_graph()
            g2 = create_graph()
            
            g1.compile(inputs=inputs, target_cc=cc, enable_pdl=False)
            g2.compile(inputs=inputs, target_cc=cc, enable_pdl=True)
            
            for _ in range(50):
                g1(inputs=inputs)
                g2(inputs=inputs)
            torch.cuda.synchronize()
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            iters = 1000
            
            start.record()
            for _ in range(iters):
                g1(inputs=inputs)
            end.record()
            torch.cuda.synchronize()
            lat_no_pdl = start.elapsed_time(end) / iters * 1000
            
            start.record()
            for _ in range(iters):
                g2(inputs=inputs)
            end.record()
            torch.cuda.synchronize()
            lat_pdl = start.elapsed_time(end) / iters * 1000
            
            speedup = (lat_no_pdl - lat_pdl) / lat_no_pdl * 100
            
            results.append({
                "elements": elements,
                "size_exp": size_exp,
                "lat_no_pdl_us": lat_no_pdl,
                "lat_pdl_us": lat_pdl,
                "speedup_pct": speedup,
            })
            
            indicator = "+" if speedup > 0.5 else "-" if speedup < -0.5 else "~"
            print(f"  {elements:>10} elements: {lat_no_pdl:.1f}us -> {lat_pdl:.1f}us ({indicator}{abs(speedup):.2f}%)")
            
        except Exception as e:
            print(f"  {elements:>10} elements: FAILED - {e}")
    
    return results


def sweep_chain_length_varying_size(cc: int):
    """Test different chain lengths at different tensor sizes."""
    print("\n=== Chain Length vs Tensor Size ===")
    
    results = []
    
    for size in [256, 1024, 4096, 16384]:
        for chain_len in [2, 4, 8, 16]:
            def create_graph():
                g = mi.new_kernel_graph()
                X = g.new_input(dims=(size, 64), dtype=mi.float16)
                Y = g.new_input(dims=(size, 64), dtype=mi.float16)
                O = X
                for i in range(chain_len):
                    if i % 2 == 0:
                        O = g.silu(O)
                    else:
                        O = g.add(O, Y)
                g.mark_output(O)
                return g
            
            inputs = [
                torch.randn(size, 64, dtype=torch.float16, device="cuda"),
                torch.randn(size, 64, dtype=torch.float16, device="cuda"),
            ]
            
            try:
                g1 = create_graph()
                g2 = create_graph()
                
                g1.compile(inputs=inputs, target_cc=cc, enable_pdl=False)
                g2.compile(inputs=inputs, target_cc=cc, enable_pdl=True)
                
                for _ in range(50):
                    g1(inputs=inputs)
                    g2(inputs=inputs)
                torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                for _ in range(500):
                    g1(inputs=inputs)
                end.record()
                torch.cuda.synchronize()
                lat_no_pdl = start.elapsed_time(end) / 500 * 1000
                
                start.record()
                for _ in range(500):
                    g2(inputs=inputs)
                end.record()
                torch.cuda.synchronize()
                lat_pdl = start.elapsed_time(end) / 500 * 1000
                
                speedup = (lat_no_pdl - lat_pdl) / lat_no_pdl * 100
                
                results.append({
                    "size": size,
                    "chain_len": chain_len,
                    "lat_no_pdl_us": lat_no_pdl,
                    "lat_pdl_us": lat_pdl,
                    "speedup_pct": speedup,
                })
                
            except Exception as e:
                pass
    
    print(f"\n{'Size':>8} {'Chain':>6} {'No PDL':>10} {'PDL':>10} {'Speedup':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['size']:>8} {r['chain_len']:>6} {r['lat_no_pdl_us']:>10.1f} {r['lat_pdl_us']:>10.1f} {r['speedup_pct']:>+10.2f}%")
    
    return results


def measure_pure_elementwise_latency(cc: int):
    """Measure latency of pure elementwise ops to understand kernel duration."""
    print("\n=== Individual Kernel Latency ===")
    
    for size in [1024, 4096, 16384, 65536, 262144]:
        x = torch.randn(size, 64, dtype=torch.float16, device="cuda")
        y = torch.randn(size, 64, dtype=torch.float16, device="cuda")
        
        for _ in range(100):
            _ = torch.nn.functional.silu(x)
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(1000):
            _ = torch.nn.functional.silu(x)
        end.record()
        torch.cuda.synchronize()
        
        lat_us = start.elapsed_time(end)
        print(f"  silu({size}x64): {lat_us:.2f}us")


def main():
    parser = argparse.ArgumentParser(description="PDL Characterization Study")
    parser.add_argument("--output", type=str, default="pdl_characterization.json",
                        help="Output JSON file path")
    args = parser.parse_args()
    
    mi.set_gpu_device_id(0)
    gpu_name, cc = get_gpu_info()
    
    print("=" * 60)
    print("PDL Characterization Study".center(60))
    print("=" * 60)
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: SM{cc}")
    
    if cc not in (90, 100):
        print("PDL requires SM90+")
        return
    
    launch_overhead = measure_kernel_launch_overhead()
    measure_pure_elementwise_latency(cc)
    size_results = sweep_tensor_size(cc)
    chain_results = sweep_chain_length_varying_size(cc)
    
    print("\n" + "=" * 60)
    print("CONCLUSIONS".center(60))
    print("=" * 60)
    
    positive = [r for r in size_results if r.get("speedup_pct", 0) > 0.5]
    negative = [r for r in size_results if r.get("speedup_pct", 0) < -0.5]
    
    print(f"\nSize sweep: {len(positive)} positive, {len(negative)} negative")
    
    if positive:
        best = max(positive, key=lambda r: r["speedup_pct"])
        print(f"Best case: {best['elements']} elements, {best['speedup_pct']:.2f}% speedup")
    
    if negative:
        worst = min(negative, key=lambda r: r["speedup_pct"])
        print(f"Worst case: {worst['elements']} elements, {worst['speedup_pct']:.2f}% slowdown")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "gpu": {"name": gpu_name, "compute_capability": cc},
        "launch_overhead_us": launch_overhead,
        "size_sweep": size_results,
        "chain_sweep": chain_results,
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

