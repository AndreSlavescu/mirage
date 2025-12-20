#!/usr/bin/env python3
import torch
import mirage as mi


def get_gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    compute_capability = props.major * 10 + props.minor
    gpu_name = props.name
    return gpu_name, compute_capability


def benchmark_builtin_ops():
    print("\n--- Built-in Ops (add, silu, mul) ---")
    
    def create_chain():
        graph = mi.new_kernel_graph()
        X = graph.new_input(dims=(64, 2048), dtype=mi.float16)
        Y = graph.new_input(dims=(64, 2048), dtype=mi.float16)
        O = graph.add(X, Y)
        O = graph.silu(O)
        O = graph.mul(O, Y)
        O = graph.silu(O)
        O = graph.mul(O, X)
        graph.mark_output(O)
        return graph
    
    input_tensors = [
        torch.randn(64, 2048, dtype=torch.float16, device="cuda"),
        torch.randn(64, 2048, dtype=torch.float16, device="cuda"),
    ]
    
    gpu_name, compute_capability = get_gpu_info()
    
    graph1 = create_chain()
    graph2 = create_chain()
    
    print("  Compiling without PDL...")
    graph1.compile(inputs=input_tensors, target_cc=compute_capability, enable_pdl=False)
    
    print("  Compiling with PDL...")
    graph2.compile(inputs=input_tensors, target_cc=compute_capability, enable_pdl=True)
    
    for _ in range(50):
        graph1(inputs=input_tensors)
        graph2(inputs=input_tensors)
    torch.cuda.synchronize()
    
    iterations = 500
    
    print("  Benchmarking without PDL...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph1(inputs=input_tensors)
    end.record()
    torch.cuda.synchronize()
    latency_no_pdl = start.elapsed_time(end) / iterations
    
    pdl1 = {
        "enabled": graph1.pdl_enabled,
        "chains": graph1.pdl_chains_count,
        "ops": graph1.pdl_kernels_optimized,
    }
    print(f"    Latency: {latency_no_pdl:.4f} ms | PDL: {pdl1}")
    
    print("  Benchmarking with PDL...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph2(inputs=input_tensors)
    end.record()
    torch.cuda.synchronize()
    latency_pdl = start.elapsed_time(end) / iterations
    
    pdl2 = {
        "enabled": graph2.pdl_enabled,
        "chains": graph2.pdl_chains_count,
        "ops": graph2.pdl_kernels_optimized,
    }
    print(f"    Latency: {latency_pdl:.4f} ms | PDL: {pdl2}")
    
    return latency_no_pdl, latency_pdl


def benchmark_long_chain():
    """Benchmark a longer elementwise chain (10 ops)."""
    print("\n--- Long Elementwise Chain (10 ops) ---")
    
    def create_chain():
        graph = mi.new_kernel_graph()
        X = graph.new_input(dims=(64, 4096), dtype=mi.float16)
        Y = graph.new_input(dims=(64, 4096), dtype=mi.float16)
        O = graph.add(X, Y)
        O = graph.silu(O)
        O = graph.mul(O, Y)
        O = graph.gelu(O)
        O = graph.add(O, X)
        O = graph.silu(O)
        O = graph.mul(O, Y)
        O = graph.relu(O)
        O = graph.add(O, X)
        O = graph.silu(O)
        graph.mark_output(O)
        return graph
    
    input_tensors = [
        torch.randn(64, 4096, dtype=torch.float16, device="cuda"),
        torch.randn(64, 4096, dtype=torch.float16, device="cuda"),
    ]
    
    gpu_name, compute_capability = get_gpu_info()
    
    graph1 = create_chain()
    graph2 = create_chain()
    
    print("  Compiling without PDL...")
    graph1.compile(inputs=input_tensors, target_cc=compute_capability, enable_pdl=False)
    
    print("  Compiling with PDL...")
    graph2.compile(inputs=input_tensors, target_cc=compute_capability, enable_pdl=True)
    
    for _ in range(50):
        graph1(inputs=input_tensors)
        graph2(inputs=input_tensors)
    torch.cuda.synchronize()
    
    iterations = 500
    
    print("  Benchmarking without PDL...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph1(inputs=input_tensors)
    end.record()
    torch.cuda.synchronize()
    latency_no_pdl = start.elapsed_time(end) / iterations
    
    pdl1 = {
        "enabled": graph1.pdl_enabled,
        "chains": graph1.pdl_chains_count,
        "ops": graph1.pdl_kernels_optimized,
    }
    print(f"    Latency: {latency_no_pdl:.4f} ms | PDL: {pdl1}")
    
    print("  Benchmarking with PDL...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph2(inputs=input_tensors)
    end.record()
    torch.cuda.synchronize()
    latency_pdl = start.elapsed_time(end) / iterations
    
    pdl2 = {
        "enabled": graph2.pdl_enabled,
        "chains": graph2.pdl_chains_count,
        "ops": graph2.pdl_kernels_optimized,
    }
    print(f"    Latency: {latency_pdl:.4f} ms | PDL: {pdl2}")
    
    return latency_no_pdl, latency_pdl


def benchmark_gated_activation():
    """Benchmark gated activation pattern: silu(x) * y (like in Llama/Qwen MLP)."""
    print("\n--- Gated Activation (silu(x) * y) ---")
    
    def create_chain():
        graph = mi.new_kernel_graph()
        X = graph.new_input(dims=(32, 8192), dtype=mi.float16)
        Y = graph.new_input(dims=(32, 8192), dtype=mi.float16)
        gate = graph.silu(X)
        O = graph.mul(gate, Y)
        graph.mark_output(O)
        return graph
    
    input_tensors = [
        torch.randn(32, 8192, dtype=torch.float16, device="cuda"),
        torch.randn(32, 8192, dtype=torch.float16, device="cuda"),
    ]
    
    gpu_name, compute_capability = get_gpu_info()
    
    graph1 = create_chain()
    graph2 = create_chain()
    
    print("  Compiling without PDL...")
    graph1.compile(inputs=input_tensors, target_cc=compute_capability, enable_pdl=False)
    
    print("  Compiling with PDL...")
    graph2.compile(inputs=input_tensors, target_cc=compute_capability, enable_pdl=True)
    
    for _ in range(50):
        graph1(inputs=input_tensors)
        graph2(inputs=input_tensors)
    torch.cuda.synchronize()
    
    iterations = 500
    
    print("  Benchmarking without PDL...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph1(inputs=input_tensors)
    end.record()
    torch.cuda.synchronize()
    latency_no_pdl = start.elapsed_time(end) / iterations
    
    pdl1 = {
        "enabled": graph1.pdl_enabled,
        "chains": graph1.pdl_chains_count,
        "ops": graph1.pdl_kernels_optimized,
    }
    print(f"    Latency: {latency_no_pdl:.4f} ms | PDL: {pdl1}")
    
    print("  Benchmarking with PDL...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph2(inputs=input_tensors)
    end.record()
    torch.cuda.synchronize()
    latency_pdl = start.elapsed_time(end) / iterations
    
    pdl2 = {
        "enabled": graph2.pdl_enabled,
        "chains": graph2.pdl_chains_count,
        "ops": graph2.pdl_kernels_optimized,
    }
    print(f"    Latency: {latency_pdl:.4f} ms | PDL: {pdl2}")
    
    return latency_no_pdl, latency_pdl


def benchmark_residual_pattern():
    """Benchmark residual connection pattern: x + activation(y)."""
    print("\n--- Residual + Activation (x + silu(gelu(y))) ---")
    
    def create_chain():
        graph = mi.new_kernel_graph()
        X = graph.new_input(dims=(128, 2048), dtype=mi.float16)
        Y = graph.new_input(dims=(128, 2048), dtype=mi.float16)
        O = graph.gelu(Y)
        O = graph.silu(O)
        O = graph.add(X, O)
        graph.mark_output(O)
        return graph
    
    input_tensors = [
        torch.randn(128, 2048, dtype=torch.float16, device="cuda"),
        torch.randn(128, 2048, dtype=torch.float16, device="cuda"),
    ]
    
    gpu_name, compute_capability = get_gpu_info()
    
    graph1 = create_chain()
    graph2 = create_chain()
    
    print("  Compiling without PDL...")
    graph1.compile(inputs=input_tensors, target_cc=compute_capability, enable_pdl=False)
    
    print("  Compiling with PDL...")
    graph2.compile(inputs=input_tensors, target_cc=compute_capability, enable_pdl=True)
    
    for _ in range(50):
        graph1(inputs=input_tensors)
        graph2(inputs=input_tensors)
    torch.cuda.synchronize()
    
    iterations = 500
    
    print("  Benchmarking without PDL...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph1(inputs=input_tensors)
    end.record()
    torch.cuda.synchronize()
    latency_no_pdl = start.elapsed_time(end) / iterations
    
    pdl1 = {
        "enabled": graph1.pdl_enabled,
        "chains": graph1.pdl_chains_count,
        "ops": graph1.pdl_kernels_optimized,
    }
    print(f"    Latency: {latency_no_pdl:.4f} ms | PDL: {pdl1}")
    
    print("  Benchmarking with PDL...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph2(inputs=input_tensors)
    end.record()
    torch.cuda.synchronize()
    latency_pdl = start.elapsed_time(end) / iterations
    
    pdl2 = {
        "enabled": graph2.pdl_enabled,
        "chains": graph2.pdl_chains_count,
        "ops": graph2.pdl_kernels_optimized,
    }
    print(f"    Latency: {latency_pdl:.4f} ms | PDL: {pdl2}")
    
    return latency_no_pdl, latency_pdl


def benchmark_scaling_chain():
    """Benchmark scaling chain pattern: mul * mul * mul (common in attention)."""
    print("\n--- Scaling Chain (x * scale1 * scale2 * scale3) ---")
    
    def create_chain():
        graph = mi.new_kernel_graph()
        X = graph.new_input(dims=(64, 4096), dtype=mi.float16)
        S1 = graph.new_input(dims=(1, 4096), dtype=mi.float16)
        S2 = graph.new_input(dims=(1, 4096), dtype=mi.float16)
        S3 = graph.new_input(dims=(1, 4096), dtype=mi.float16)
        O = graph.mul(X, S1)
        O = graph.mul(O, S2)
        O = graph.mul(O, S3)
        graph.mark_output(O)
        return graph
    
    input_tensors = [
        torch.randn(64, 4096, dtype=torch.float16, device="cuda"),
        torch.randn(1, 4096, dtype=torch.float16, device="cuda"),
        torch.randn(1, 4096, dtype=torch.float16, device="cuda"),
        torch.randn(1, 4096, dtype=torch.float16, device="cuda"),
    ]
    
    gpu_name, compute_capability = get_gpu_info()
    
    graph1 = create_chain()
    graph2 = create_chain()
    
    print("  Compiling without PDL...")
    graph1.compile(inputs=input_tensors, target_cc=compute_capability, enable_pdl=False)
    
    print("  Compiling with PDL...")
    graph2.compile(inputs=input_tensors, target_cc=compute_capability, enable_pdl=True)
    
    for _ in range(50):
        graph1(inputs=input_tensors)
        graph2(inputs=input_tensors)
    torch.cuda.synchronize()
    
    iterations = 500
    
    print("  Benchmarking without PDL...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph1(inputs=input_tensors)
    end.record()
    torch.cuda.synchronize()
    latency_no_pdl = start.elapsed_time(end) / iterations
    
    pdl1 = {
        "enabled": graph1.pdl_enabled,
        "chains": graph1.pdl_chains_count,
        "ops": graph1.pdl_kernels_optimized,
    }
    print(f"    Latency: {latency_no_pdl:.4f} ms | PDL: {pdl1}")
    
    print("  Benchmarking with PDL...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph2(inputs=input_tensors)
    end.record()
    torch.cuda.synchronize()
    latency_pdl = start.elapsed_time(end) / iterations
    
    pdl2 = {
        "enabled": graph2.pdl_enabled,
        "chains": graph2.pdl_chains_count,
        "ops": graph2.pdl_kernels_optimized,
    }
    print(f"    Latency: {latency_pdl:.4f} ms | PDL: {pdl2}")
    
    return latency_no_pdl, latency_pdl


def main():
    mi.set_gpu_device_id(0)
    gpu_name, compute_capability = get_gpu_info()
    pdl_supported = compute_capability in (90, 100)

    print("=" * 65)
    print("PDL Infrastructure Benchmark".center(65))
    print("=" * 65)
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: SM{compute_capability}")
    print(f"PDL Supported: {pdl_supported}")
    print("=" * 65)

    results = []
    
    lat1, lat2 = benchmark_builtin_ops()
    results.append(("Built-in ops (5 ops)", lat1, lat2))
    
    lat3, lat4 = benchmark_long_chain()
    results.append(("Long chain (10 ops)", lat3, lat4))
    
    lat5, lat6 = benchmark_gated_activation()
    results.append(("Gated activation", lat5, lat6))
    
    lat7, lat8 = benchmark_residual_pattern()
    results.append(("Residual pattern", lat7, lat8))
    
    lat9, lat10 = benchmark_scaling_chain()
    results.append(("Scaling chain", lat9, lat10))

    print("\n" + "=" * 65)
    print("SUMMARY".center(65))
    print("=" * 65)
    
    for name, no_pdl, with_pdl in results:
        diff_pct = (no_pdl - with_pdl) / no_pdl * 100
        print(f"{name}:")
        print(f"  Without PDL: {no_pdl:.4f} ms")
        print(f"  With PDL:    {with_pdl:.4f} ms")
        print(f"  Difference:  {diff_pct:+.2f}%")
    print()

if __name__ == "__main__":
    main()
