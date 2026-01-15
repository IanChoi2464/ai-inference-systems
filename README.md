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
