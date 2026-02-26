# 🌌 Sage Kaizen AI

> A Dual-Brain, Local-First AI System for Thinking, Creating, and Controlling the Physical World

Sage Kaizen is a modular, production-oriented local AI architecture designed to run powerful large language models on consumer hardware — while integrating voice interaction, device orchestration (Raspberry Pi agents), retrieval-augmented generation (RAG), and immersive LED visualizations.

It is built around one principle:

> Continuous improvement (Kaizen) guided by wisdom (Sage).

---

🚀 What Is Sage Kaizen?

Sage Kaizen is:

- 🧠 A Dual-Mode AI Stack (Fast Brain + Architect Brain) 🧠
- 🎙 A Voice-Driven Assistant
- 🖥 A Local AI Architect & Code Generator
- 🔦 A Physical World Controller (LED matrices, Pi devices)
- 📚 A Research System with RAG
- ✍️ A Creative Writing Engine
- 🎓 A K–12 Tutor
- 📊 A Self-Documenting Codebase Generator

All running locally, powered by modern GPUs and llama.cpp.

No cloud required.

---

🧠 Dual-Brain Architecture 🧠

Sage Kaizen uses two coordinated model servers:

| Brain  | Purpose  | Default Use |
| ----------- | ----------- | ----------- |
| ⚡ Fast Brain (Q5_K_M) | Responsive chat, voice loop, orchestration | Default |
| 🏗 Architect Brain (Q6_K) | Deep reasoning, research, code generation | On demand |


Both run via llama-server from llama.cpp (CUDA-enabled build).

Key capabilities supported by the current llama-server build include:

- Multi-GPU offloading (--device, --split-mode, --tensor-split)
- Flash Attention (--flash-attn)
- KV cache tuning (--cache-type-k, --cache-type-v)
- Context scaling (RoPE, YaRN)
- Speculative decoding (draft models)
- Continuous batching
- JSON schema constrained generation
- DeepSeek reasoning mode (--reasoning-format deepseek)

(Full flag support verified against current llama-server build llama-server --help)

---

🏗 High-Level Architecture

                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    └─────────┬────────────┘
                              │
                      Router (Fast vs Architect)
                              │
              ┌───────────────┴────────────────┐
              │                                │
      llama-server (Q5_K_M)            llama-server (Q6_K)
        Fast Brain GPU                    Architect GPU
              │                                │
              └───────────────┬────────────────┘
                              │
                        Tool Layer
        ┌────────────┬───────────────┬──────────────┐
        │            │               │              │
     RAG Engine   Pi Agent     Code Generator    Voice System
 (Postgres+pgvector) (ZeroMQ)  (Repo Scanner)  (STT → TTS)

 ---

 🔊 Voice Loop

Sage Kaizen supports a real-time voice interaction pipeline:
`Speech → STT → Fast Brain → Tool Execution → TTS → Audio Output`

Recommended components:
- STT: Vosk or Whisper (local)
- TTS: Coqui XTTS or OpenVoice
- Audio Hardware: Raspberry Pi + WM8960 / Voice Bonnet

 ---

🌠 Physical World Integration

Sage Kaizen is designed to control:
- 64×64 LED matrix cubes
- Multi-panel LED installations
- Audio-reactive equalizers
- Star maps and constellations
- “Cosmic mode” immersive visualizations

Communication uses:
- ZeroMQ transport
- Raspberry Pi agents
- Modular device command protocol

Example command:
`“Set LED mode cosmic.”`

 ---

📚 RAG (Retrieval Augmented Generation)

Sage Kaizen can stay current using:
- Local folder ingestion
- RSS feed ingestion
- Web ingestion
- PostgreSQL + pgvector embeddings
- Re-runnable ingestion pipeline
- Hash-based source tracking

This enables:
- Research workflows
- Context-aware reasoning
- Codebase analysis
- Documentation generation


 ---

🧑‍💻 Self-Documenting Codebase

Sage Kaizen can:
- Scan your repository
- Generate a README
- Create Mermaid diagrams
- Suggest architectural refactors
- Produce ADR (Architecture Decision Records)

It acts as a persistent local AI architect.

 ---

🎨 Creative & Educational Modes

Sage Kaizen is not only technical.

It can:
- Write poems
- Compose stories (“Last night in Atlanta”)
- Explain theology and philosophy
- Teach math, science, and history (Grades 1–12)
- Adapt tone for children or professionals

 ---

🖥 Hardware Profile (Reference Build)

Designed and tested on a high-performance workstation:
- AMD Ryzen 9 9950X3D
- 192GB DDR5
- RTX 5090 (32GB VRAM)
- RTX 5080 (16GB VRAM)
- 40TB storage
- Windows 11 Pro
- Custom CUDA 13.1 llama.cpp build

Also integrates with:
- Raspberry Pi 4 / 5 nodes
- WS2812B LED systems
- WM8960 audio hardware

 ---

🛠 Core Technology Stack
- llama.cpp (CUDA build)
- llama-server
- llama-cpp-python
- Streamlit
- PostgreSQL 18 + pgvector
- ZeroMQ
- Python 3.14+
- Custom GPU tuning
- Multi-GPU split execution

 ---

📦 Modular Design Philosophy

Sage Kaizen is intentionally modular:

| Module  | Replaceable? |
| ----------- | ----------- |
| LLM backend  | Yes |
| Voice engine  | Yes |
| TTS  | Yes |
| Vector DB  | Yes |
| UI layer  | Yes |
| Device transport  | Yes |

The goal is long-term extensibility.

 ---

🔐 Local-First Philosophy

- No mandatory cloud dependency
- No external API required
- Runs fully offline
- Complete data ownership
- Expandable to air-gapped environments

 ---

📌 Project Goals

- Production-ready local AI architecture
- Hardware-integrated AI ecosystem
- Long-context reasoning
- Continuous performance tuning
- Reproducible configuration
- Modular evolution over time

 ---

🧭 Roadmap

Current build stages:

1. Dual llama-server brains
2. Voice loop
3. Pi Agent orchestration
4. RAG v1
5. Auto-documentation system
6. Multimodal expansion
7. Advanced reasoning budgets
8. Distributed node scaling

 ---

🤝 Contributing

This project is evolving.

- Contributions welcome in:
- GPU optimization research
- llama.cpp tuning
- Voice integration
- Embedded systems control
- RAG improvements
- Documentation
- Testing + benchmarking

 ---

📜 Philosophy

Sage Kaizen means:
- Sage → Wisdom, reflection, deep reasoning.
- Kaizen → Continuous improvement.

The system embodies both:
- Fast responses when needed.
- Deep thought when required.
- Continuous tuning and architectural refinement.

 ---

⭐ Why This Matters

Sage Kaizen demonstrates that:

> A powerful, modular, reasoning-capable AI system can run entirely on personal hardware — and control the physical world.

It merges:
- AI
- Hardware
- Creativity
- Education
- Systems engineering
- Continuous improvement

Into a single evolving architecture.

 ---

📣 Final Note

This is not just a chatbot.

It is:

A local AI architect.
A reasoning engine.
A physical-world controller.
A creative partner.
A tutor.
A continuously evolving system.

 ---
