#!/usr/bin/env python3
"""
Full Paper Experiment Suite for PDL Characterization

This generates all the data needed for a full research paper on
automatic PDL insertion for tensor program optimization.

Experiments:
1. Microbenchmarks: Characterize when PDL helps
2. Operator patterns: Different ML operator combinations  
3. Model shapes: LLaMA, Qwen, Mistral configurations
4. Baselines: Compare with CUDA Graphs, torch.compile
5. Ablations: Chain length, tensor size, batch size
"""
import torch
import mirage as mi
import numpy as np
import json
import time
import argparse
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import subprocess
import os


@dataclass
class ExperimentConfig:
    name: str
    description: str
    
    
@dataclass  
class BenchResult:
    experiment: str
    config: Dict[str, Any]
    latency_baseline_us: float
    latency_optimized_us: float
    speedup_pct: float
    pdl_chains: int
    pdl_ops: int
    std_baseline_us: float = 0.0
    std_optimized_us: float = 0.0
    

def get_gpu_info():
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": props.major * 10 + props.minor,
        "total_memory_gb": props.total_memory / (1024**3),
        "sm_count": props.multi_processor_count,
    }


def benchmark_graph(
    create_fn: Callable,
    inputs: List[torch.Tensor],
    cc: int,
    warmup: int = 50,
    iterations: int = 200,
    trials: int = 3,
) -> Dict[str, Any]:
    """Run benchmark with statistical significance."""
    
    lat_no_pdl = []
    lat_pdl = []
    pdl_info = None
    
    for _ in range(trials):
        g1 = create_fn()
        g2 = create_fn()
        
        g1.compile(inputs=inputs, target_cc=cc, enable_pdl=False)
        g2.compile(inputs=inputs, target_cc=cc, enable_pdl=True)
        
        if pdl_info is None:
            pdl_info = {
                "chains": g2.pdl_chains_count,
                "ops": g2.pdl_kernels_optimized,
            }
        
        for _ in range(warmup):
            g1(inputs=inputs)
            g2(inputs=inputs)
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            g1(inputs=inputs)
        end.record()
        torch.cuda.synchronize()
        lat_no_pdl.append(start.elapsed_time(end) / iterations * 1000)
        
        start.record()
        for _ in range(iterations):
            g2(inputs=inputs)
        end.record()
        torch.cuda.synchronize()
        lat_pdl.append(start.elapsed_time(end) / iterations * 1000)
    
    mean_no_pdl = np.mean(lat_no_pdl)
    mean_pdl = np.mean(lat_pdl)
    
    return {
        "latency_no_pdl_us": mean_no_pdl,
        "latency_pdl_us": mean_pdl,
        "std_no_pdl_us": np.std(lat_no_pdl),
        "std_pdl_us": np.std(lat_pdl),
        "speedup_pct": (mean_no_pdl - mean_pdl) / mean_no_pdl * 100,
        "pdl_chains": pdl_info["chains"],
        "pdl_ops": pdl_info["ops"],
    }


class Experiment1_TensorSizeCharacterization:
    """Experiment 1: How does tensor size affect PDL benefit?"""
    
    name = "tensor_size_characterization"
    
    @staticmethod
    def run(cc: int) -> List[BenchResult]:
        results = []
        sizes = [2**exp for exp in range(8, 18, 2)]
        
        for i, size in enumerate(sizes):
            print(f"  [{i+1}/{len(sizes)}] Size {size}...", end=" ", flush=True)
                
            def create_graph(s=size):
                g = mi.new_kernel_graph()
                X = g.new_input(dims=(s, 64), dtype=mi.float16)
                Y = g.new_input(dims=(s, 64), dtype=mi.float16)
                O = g.silu(X)
                O = g.add(O, Y)
                O = g.mul(O, Y)
                g.mark_output(O)
                return g
            
            inputs = [
                torch.randn(size, 64, dtype=torch.float16, device="cuda"),
                torch.randn(size, 64, dtype=torch.float16, device="cuda"),
            ]
            
            try:
                stats = benchmark_graph(create_graph, inputs, cc)
                results.append(BenchResult(
                    experiment="tensor_size",
                    config={"size": size, "elements": size * 64},
                    latency_baseline_us=stats["latency_no_pdl_us"],
                    latency_optimized_us=stats["latency_pdl_us"],
                    speedup_pct=stats["speedup_pct"],
                    pdl_chains=stats["pdl_chains"],
                    pdl_ops=stats["pdl_ops"],
                    std_baseline_us=stats["std_no_pdl_us"],
                    std_optimized_us=stats["std_pdl_us"],
                ))
                print(f"{stats['speedup_pct']:+.2f}%", flush=True)
            except Exception as e:
                print(f"FAILED - {e}", flush=True)
        
        return results


class Experiment2_ChainLength:
    """Experiment 2: How does chain length affect PDL benefit?"""
    
    name = "chain_length"
    
    @staticmethod
    def run(cc: int) -> List[BenchResult]:
        results = []
        size = 8192
        chain_lengths = [2, 4, 6, 8, 12]
        
        for i, chain_len in enumerate(chain_lengths):
            print(f"  [{i+1}/{len(chain_lengths)}] Chain {chain_len}...", end=" ", flush=True)
            def create_graph(cl=chain_len):
                g = mi.new_kernel_graph()
                X = g.new_input(dims=(size, 64), dtype=mi.float16)
                Y = g.new_input(dims=(size, 64), dtype=mi.float16)
                O = X
                for i in range(cl):
                    if i % 3 == 0:
                        O = g.silu(O)
                    elif i % 3 == 1:
                        O = g.add(O, Y)
                    else:
                        O = g.mul(O, Y)
                g.mark_output(O)
                return g
            
            inputs = [
                torch.randn(size, 64, dtype=torch.float16, device="cuda"),
                torch.randn(size, 64, dtype=torch.float16, device="cuda"),
            ]
            
            try:
                stats = benchmark_graph(create_graph, inputs, cc)
                results.append(BenchResult(
                    experiment="chain_length",
                    config={"chain_length": chain_len, "size": size},
                    latency_baseline_us=stats["latency_no_pdl_us"],
                    latency_optimized_us=stats["latency_pdl_us"],
                    speedup_pct=stats["speedup_pct"],
                    pdl_chains=stats["pdl_chains"],
                    pdl_ops=stats["pdl_ops"],
                ))
                print(f"{stats['speedup_pct']:+.2f}%", flush=True)
            except Exception as e:
                print(f"FAILED - {e}", flush=True)
        
        return results


class Experiment3_OperatorPatterns:
    """Experiment 3: Different operator pattern benchmarks."""
    
    name = "operator_patterns"
    
    @staticmethod
    def run(cc: int) -> List[BenchResult]:
        results = []
        batch, hidden = 64, 2048
        
        patterns = [
            ("silu_mul", lambda g, x, y: g.mul(g.silu(x), y)),
            ("gelu_add", lambda g, x, y: g.add(g.gelu(x), y)),
            ("residual", lambda g, x, y: g.add(g.silu(x), y)),
        ]
        
        for i, (name, pattern_fn) in enumerate(patterns):
            print(f"  [{i+1}/{len(patterns)}] {name}...", end=" ", flush=True)
            
            def create_graph(pf=pattern_fn):
                g = mi.new_kernel_graph()
                X = g.new_input(dims=(batch, hidden), dtype=mi.float16)
                Y = g.new_input(dims=(batch, hidden), dtype=mi.float16)
                O = pf(g, X, Y)
                g.mark_output(O)
                return g
            
            inputs = [
                torch.randn(batch, hidden, dtype=torch.float16, device="cuda"),
                torch.randn(batch, hidden, dtype=torch.float16, device="cuda"),
            ]
            
            try:
                stats = benchmark_graph(create_graph, inputs, cc)
                results.append(BenchResult(
                    experiment="operator_pattern",
                    config={"pattern": name, "batch": batch, "hidden": hidden},
                    latency_baseline_us=stats["latency_no_pdl_us"],
                    latency_optimized_us=stats["latency_pdl_us"],
                    speedup_pct=stats["speedup_pct"],
                    pdl_chains=stats["pdl_chains"],
                    pdl_ops=stats["pdl_ops"],
                ))
                print(f"{stats['speedup_pct']:+.2f}%", flush=True)
            except Exception as e:
                print(f"FAILED - {e}", flush=True)
        
        return results


class Experiment4_ModelShapes:
    """Experiment 4: Real model configuration shapes."""
    
    name = "model_shapes"
    
    MODELS = [
        ("llama-7b", 4096, 11008),
        ("qwen-7b", 4096, 11008),
    ]
    
    @staticmethod
    def run(cc: int) -> List[BenchResult]:
        results = []
        configs = [(m, b) for m, _, _ in Experiment4_ModelShapes.MODELS for b in [1, 8]]
        
        for i, ((model_name, hidden, intermediate), batch) in enumerate(
            [(m, b) for m in Experiment4_ModelShapes.MODELS for b in [1, 8]]
        ):
            seq = 256
            print(f"  [{i+1}/{len(configs)}] {model_name} batch={batch}...", end=" ", flush=True)
            
            def create_mlp_pattern(b=batch, s=seq, h=hidden, inter=intermediate):
                g = mi.new_kernel_graph()
                X = g.new_input(dims=(b * s, h), dtype=mi.float16)
                W_gate = g.new_input(dims=(h, inter), dtype=mi.float16)
                W_up = g.new_input(dims=(h, inter), dtype=mi.float16)
                W_down = g.new_input(dims=(inter, h), dtype=mi.float16)
                
                gate = g.matmul(X, W_gate)
                gate = g.silu(gate)
                up = g.matmul(X, W_up)
                hidden_states = g.mul(gate, up)
                O = g.matmul(hidden_states, W_down)
                g.mark_output(O)
                return g
            
            inputs = [
                torch.randn(batch * seq, hidden, dtype=torch.float16, device="cuda"),
                torch.randn(hidden, intermediate, dtype=torch.float16, device="cuda"),
                torch.randn(hidden, intermediate, dtype=torch.float16, device="cuda"),
                torch.randn(intermediate, hidden, dtype=torch.float16, device="cuda"),
            ]
            
            try:
                stats = benchmark_graph(create_mlp_pattern, inputs, cc)
                results.append(BenchResult(
                    experiment="model_shape",
                    config={"model": model_name, "batch": batch, "seq": seq},
                    latency_baseline_us=stats["latency_no_pdl_us"],
                    latency_optimized_us=stats["latency_pdl_us"],
                    speedup_pct=stats["speedup_pct"],
                    pdl_chains=stats["pdl_chains"],
                    pdl_ops=stats["pdl_ops"],
                ))
                print(f"{stats['speedup_pct']:+.2f}%", flush=True)
            except Exception as e:
                print(f"FAILED - {e}", flush=True)
        
        return results


class Experiment5_CUDAGraphsComparison:
    """Experiment 5: Compare PDL with CUDA Graphs."""
    
    name = "cuda_graphs_comparison"
    
    @staticmethod
    def run(cc: int) -> List[BenchResult]:
        results = []
        configs = [(4096, 4), (16384, 4), (65536, 4)]
        
        for idx, (size, chain_len) in enumerate(configs):
            print(f"  [{idx+1}/{len(configs)}] size={size}, chain={chain_len}...", end=" ", flush=True)
            
            x = torch.randn(size, 64, dtype=torch.float16, device="cuda")
            y = torch.randn(size, 64, dtype=torch.float16, device="cuda")
            
            for _ in range(20):
                o = x
                for i in range(chain_len):
                    if i % 2 == 0:
                        o = torch.nn.functional.silu(o)
                    else:
                        o = o + y
            torch.cuda.synchronize()
            
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                o = x
                for i in range(chain_len):
                    if i % 2 == 0:
                        o = torch.nn.functional.silu(o)
                    else:
                        o = o + y
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            iterations = 200
            
            start.record()
            for _ in range(iterations):
                o = x
                for i in range(chain_len):
                    if i % 2 == 0:
                        o = torch.nn.functional.silu(o)
                    else:
                        o = o + y
            end.record()
            torch.cuda.synchronize()
            lat_eager = start.elapsed_time(end) / iterations * 1000
            
            start.record()
            for _ in range(iterations):
                cuda_graph.replay()
            end.record()
            torch.cuda.synchronize()
            lat_graph = start.elapsed_time(end) / iterations * 1000
            
            def create_mirage_graph(s=size, cl=chain_len):
                mg = mi.new_kernel_graph()
                X = mg.new_input(dims=(s, 64), dtype=mi.float16)
                Y = mg.new_input(dims=(s, 64), dtype=mi.float16)
                O = X
                for i in range(cl):
                    if i % 2 == 0:
                        O = mg.silu(O)
                    else:
                        O = mg.add(O, Y)
                mg.mark_output(O)
                return mg
            
            inputs = [x, y]
            
            try:
                stats = benchmark_graph(create_mirage_graph, inputs, cc)
                
                results.append(BenchResult(
                    experiment="cuda_graphs_comparison",
                    config={
                        "size": size,
                        "chain_length": chain_len,
                        "pytorch_eager_us": lat_eager,
                        "cuda_graph_us": lat_graph,
                    },
                    latency_baseline_us=stats["latency_no_pdl_us"],
                    latency_optimized_us=stats["latency_pdl_us"],
                    speedup_pct=stats["speedup_pct"],
                    pdl_chains=stats["pdl_chains"],
                    pdl_ops=stats["pdl_ops"],
                ))
                print(f"{stats['speedup_pct']:+.2f}%", flush=True)
            except Exception as e:
                print(f"FAILED - {e}", flush=True)
        
        return results


def run_all_experiments(cc: int, output_dir: str):
    """Run all experiments and save results."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    experiments = [
        Experiment1_TensorSizeCharacterization,
        Experiment2_ChainLength,
        Experiment3_OperatorPatterns,
        Experiment4_ModelShapes,
        Experiment5_CUDAGraphsComparison,
    ]
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "gpu": get_gpu_info(),
        "experiments": {},
    }
    
    for i, exp_cls in enumerate(experiments):
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(experiments)}] Running: {exp_cls.name}", flush=True)
        print(f"{'='*60}", flush=True)
        
        try:
            results = exp_cls.run(cc)
            all_results["experiments"][exp_cls.name] = [asdict(r) for r in results]
            
            positive = [r for r in results if r.speedup_pct > 0.5]
            if positive:
                best = max(positive, key=lambda r: r.speedup_pct)
                print(f"  Best speedup: {best.speedup_pct:.2f}% at {best.config}", flush=True)
            else:
                print(f"  No significant positive speedups", flush=True)
                
        except Exception as e:
            print(f"  Experiment failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            all_results["experiments"][exp_cls.name] = {"error": str(e)}
    
    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return all_results


def generate_latex_tables(results: Dict, output_dir: str):
    """Generate LaTeX tables for the paper."""
    
    if "tensor_size_characterization" in results.get("experiments", {}):
        data = results["experiments"]["tensor_size_characterization"]
        if isinstance(data, list):
            lines = [
                r"\begin{table}[t]",
                r"\centering",
                r"\caption{PDL Speedup vs Tensor Size}",
                r"\label{tab:tensor_size}",
                r"\begin{tabular}{rrrrr}",
                r"\toprule",
                r"Elements & Baseline ($\mu$s) & PDL ($\mu$s) & Speedup (\%) \\",
                r"\midrule",
            ]
            for r in data[:12]:
                lines.append(
                    f"{r['config']['elements']:,} & "
                    f"{r['latency_baseline_us']:.1f} & "
                    f"{r['latency_optimized_us']:.1f} & "
                    f"{r['speedup_pct']:+.2f} \\\\"
                )
            lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
            
            with open(os.path.join(output_dir, "table_tensor_size.tex"), "w") as f:
                f.write("\n".join(lines))
    
    if "cuda_graphs_comparison" in results.get("experiments", {}):
        data = results["experiments"]["cuda_graphs_comparison"]
        if isinstance(data, list):
            lines = [
                r"\begin{table}[t]",
                r"\centering",
                r"\caption{Comparison: PDL vs CUDA Graphs vs PyTorch Eager}",
                r"\label{tab:baseline_comparison}",
                r"\begin{tabular}{rrrrrr}",
                r"\toprule",
                r"Size & Chain & Eager ($\mu$s) & CUDA Graph ($\mu$s) & Mirage+PDL ($\mu$s) \\",
                r"\midrule",
            ]
            for r in data:
                lines.append(
                    f"{r['config']['size']:,} & "
                    f"{r['config']['chain_length']} & "
                    f"{r['config']['pytorch_eager_us']:.1f} & "
                    f"{r['config']['cuda_graph_us']:.1f} & "
                    f"{r['latency_optimized_us']:.1f} \\\\"
                )
            lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
            
            with open(os.path.join(output_dir, "table_baselines.tex"), "w") as f:
                f.write("\n".join(lines))


def main():
    import sys
    
    parser = argparse.ArgumentParser(description="PDL Paper Experiments")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick subset of experiments")
    args = parser.parse_args()
    
    print("Initializing...", flush=True)
    mi.set_gpu_device_id(0)
    gpu_info = get_gpu_info()
    cc = gpu_info["compute_capability"]
    
    print("=" * 70, flush=True)
    print("PDL Paper Experiment Suite".center(70), flush=True)
    print("=" * 70, flush=True)
    print(f"GPU: {gpu_info['name']}", flush=True)
    print(f"Compute Capability: SM{cc}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print("=" * 70, flush=True)
    sys.stdout.flush()
    
    if cc not in (90, 100):
        print("PDL requires SM90+ (H100) or SM100+ (Blackwell)")
        return
    
    results = run_all_experiments(cc, args.output)
    
    generate_latex_tables(results, args.output)
    
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT COMPLETE".center(70), flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()

