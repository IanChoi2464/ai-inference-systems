# ai-inference-systems
Practical implementations and benchmarks for large-scale AI inference systems — focusing on latency, throughput, and deployment trade-offs.

## 01_minimal-llm-inference-engine
This project is a minimal LLM inference engine built from scratch with a systems-first mindset.<br>

It focuses exclusively on inference — not training — and is designed to
explore real-world performance tradeoffs such as latency vs throughput,
CPU vs GPU execution, and dynamic batching behavior.

The engine supports multiple backends including PyTorch, ONNX Runtime,
and TensorRT, and includes profiling tools and benchmarks to measure
end-to-end inference performance.

*No training code is included by design.*

### Batch size benchmark (what it shows)
We run a controlled benchmark that sends many single-sample requests and varies only the batch size. For each batch size, we measure:
- Throughput (requests/sec)
- Average latency
- P50 and P95 latency (median vs slow tail)

The takeaway is that batching has a “sweet spot.” Too small means poor throughput; too large increases waiting time. In our runs, mid-sized batches (like 4–8) typically balance throughput and latency best.

### How to run
```
python3 01_minimal-llm-inference-engine/benchmarks/batch_size_bench.py \
  --requests 1024 --batch-sizes 1,2,4,8,16 --trials 10
```

This writes a plot to `01_minimal-llm-inference-engine/result/batch_size_bench.png` showing:
- Throughput vs batch size
- Average latency vs batch size
- P50/P95 latency comparison
