<div align="center">
  <h1>⚔️ Rinha de Backend 2026 - Submission</h1>
  <p><strong>A "Bare-Metal" C++ solution focused on sub-millisecond latency, SIMD (AVX2) processing, and non-blocking I/O via Epoll.</strong></p>
</div>

<br>

## 🚀 About the Project

This is my submission for the **Rinha de Backend 2026** (Backend Fight 2026). The challenge required building a fraud risk evaluation system capable of handling high throughput under severe resource constraints (**165MB of RAM** and **0.45 CPU** per instance).

To achieve maximum performance and squeeze every CPU cycle and byte of memory, this solution was built entirely from scratch in C++.

### 🔥 Technical Highlights
- **Sub-Millisecond Latency**: The API responds to the stress test in less than `1ms` at the 99th percentile (p99).
- **No Frameworks**: The HTTP server and event loop were hand-crafted using the Linux API (`sys/epoll.h`).
- **SIMD / AVX2**: The K-Nearest Neighbors (KNN) search that determines the fraud score was optimized at the processor level, using 256-bit vector instructions (`__m256`) to calculate the Distance Squared Sum of Errors (DSSE) between vectors simultaneously.
- **Memory-Mapped Files (mmap)**: Since the fraud data/vectors weigh around ~163MB, the Docker memory limit (165MB) didn't allow loading the data into the Heap. The solution maps the static files directly into kernel memory (`MAP_PRIVATE`), avoiding memory copies (*Zero-copy*) and lazy-loading pages on demand.
- **UDS Communication (Unix Domain Sockets)**: Communication between the Load Balancer (HAProxy) and the C++ nodes happens through Unix sockets mapped in a volume, completely bypassing the internal Linux TCP/IP stack and removing internal network latency.
- **Optimized Parsing**: JSON reading with strict search algorithms without using heavy parsing libraries (no pointer allocation on the Heap).
- **HTTP Keep-Alive Pipelining**: A robust Epoll loop designed to read dozens of queued requests through the same TCP connection without bottlenecks.

---

## 🏗️ System Architecture

The topology focuses on pure speed. **HAProxy** balances the load between replicas using *Round-Robin* over physical Socket files (`.sock`), and the API instances query the pre-compiled solid-state databases.

```mermaid
graph TD
    %% Define Styles
    classDef client fill:#e0f7fa,stroke:#006064,stroke-width:2px,color:#000
    classDef haproxy fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef api fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef memory fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef unixsocket fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px,stroke-dasharray: 5 5,color:#000
    
    Client["🤖 Rinha Tester (K6 / Load Generator)"]:::client
    
    subgraph LB["Load Balancing Layer"]
        HAProxy["⚡ HAProxy (Alpine)<br/>Port: 9999<br/>Round-Robin / Keep-Alive"]:::haproxy
    end
    
    subgraph UDS["Unix Domain Sockets (Volume: rinha-sockets)"]
        Sock1[["/sockets/api1.sock"]]:::unixsocket
        Sock2[["/sockets/api2.sock"]]:::unixsocket
    end
    
    subgraph APILayer["API Backend Layer (C++)"]
        API1["🚀 API 1 (ghcr.io/luiziwasaki/rinha-2026-luiz)<br/>Limit: 165MB RAM | 0.45 CPU"]:::api
        API2["🚀 API 2 (ghcr.io/luiziwasaki/rinha-2026-luiz)<br/>Limit: 165MB RAM | 0.45 CPU"]:::api
    end
    
    subgraph APIInternal["API Internals (Event Loop & AVX2)"]
        direction TB
        Epoll{"⚙️ Epoll I/O<br/>Non-blocking"}:::api
        Parser["📝 Zero-copy HTTP Parser"]:::api
        FeatureExt["🧠 Feature Extraction (JSON)"]:::api
        KNN["⚡ AVX2 SIMD KNN Search<br/>(DIMS=14, KNN=5)"]:::api
        
        Epoll --> Parser
        Parser --> FeatureExt
        FeatureExt --> KNN
    end
    
    subgraph MMap["Memory Mapped Storage (mmap)"]
        direction TB
        Vecs[("vectors.bin<br/>(160 MB)")]:::memory
        Cents[("centroids.bin<br/>(112 KB)")]:::memory
        Labels[("labels.bin<br/>(2.9 MB)")]:::memory
    end

    %% Connections
    Client -- "HTTP POST /fraud-score" --> HAProxy
    HAProxy -- "http-reuse always" --> Sock1
    HAProxy -- "http-reuse always" --> Sock2
    Sock1 --> API1
    Sock2 --> API2
    
    API1 -.-> Epoll
    KNN -. "Zero-copy reads" .-> Vecs
    KNN -. "Zero-copy reads" .-> Cents
    KNN -. "Zero-copy reads" .-> Labels
```

---

## 🛠️ Data Lifecycle and Pre-Processing

The data pipeline is divided into **Multiple Docker Stages**:

1. **Builder Stage**: Compiles the C++ executables (`rinha_api` and the `preprocess` tool) with maximum optimization flags (`-O3`, `-march=native` or `-mavx2`).
2. **Pre-processing Stage**: The raw `references.json.gz` file is decompressed and processed by the internal `preprocess` tool, extracting references and calculating centroids (K-Means) at Docker image compile time. The results are highly optimized `.bin` files ready for raw byte reading.
3. **Runtime Stage**: The final Alpine container carries only the clean executable without dependencies and the `.bin` files, which will be consumed by C++'s `mmap()` for direct memory access, eliminating the need to hydrate objects in RAM.

---

## 📊 Memory Management Strategy

To avoid falling victim to the *OOM Killer (Out Of Memory)* within the Spartan `165MB` limit of the Rinha Docker Compose:
- The `MAX_CONN` connection limit was methodically sized to `2048`, pre-allocating exactly one static buffer in non-reclaimable memory (BSS) of around `8 MB`, keeping stack allocation predictable.
- The remainder (~157 MB) is entirely handed over to the Linux Kernel to serve as a *Page Cache* for the binary vectors, practically reducing *page faults* to zero after the first few requests (Warmup).
- The `MAP_POPULATE` flag was purposely avoided to prevent aggressive `ENOMEM` crashes on startup when strict cgroup limits are enforced by the host, relying safely on lazy page-faulting instead.

---

## 💻 How to Run and Test Locally

Make sure the original `references.json.gz` file from the Rinha repository is placed at the correct level (`resources/` folder or following the shell-script copy).

Start and orchestrate the containers using our automated script:
```bash
# Execution permission if needed
chmod +x build.sh

# Runs the multi-stage compilation, spins up the Docker cluster and performs the readiness Health Check
./build.sh
```
