# 🏗️ geniesim_engine — Design Document Index

> **Central entry point** for all design docs in the `genie_sim_engine` ecosystem.
> Read this first, then follow the link that matches your role.

---

## 📚 Document Dependency Graph

```mermaid
flowchart LR
    subgraph Layer1["🥇 Architecture & Organisation"]
        CORE["DESIGN.CORE.md<br/>Unified Backend Architecture<br/>package split · plugin system<br/>migration · test suite"]
    end

    subgraph Layer2["🥈 Interface Contracts"]
        ABI["DESIGN.ABI.md<br/>Common Engine ABI<br/>lifecycle · ROS topics<br/>config · mock strategy"]
    end

    subgraph Layer3["🥉 Implementation Proposals"]
        RL["DESIGN.RL.md<br/>RL Training Architecture<br/>NewtonSimContext · CUDA graphs<br/>ManagerBasedRLEnv"]
        ALIPC["DESIGN.ALIPC.md<br/>AL-IPC Contact Augmentation<br/>two-phase CUDA graphs<br/>Genesis-inspired patterns"]
    end

    CORE -.->|"owns + enforces"| ABI
    CORE -.->|"migrates"| RL
    CORE -.->|"migrates"| ALIPC

    ABI -.->|"Phase B: NewtonPhysicsCore"| RL
    ABI -.->|"not in scope"| ALIPC

    ALIPC -.->|"physics backbone for"| RL

    style Layer1 fill:#E8D5B7
    style Layer2 fill:#ADD8E6
    style Layer3 fill:#D8BFD8
```

---

## 🧭 Reading Guide

| You are... | Read this first | Then | When to read |
|---|---|---|---|
| ✍️ **New backend developer** (adding mujoco, libuipc, etc.) | [`DESIGN.CORE.md`](./DESIGN.CORE.md) — package layout, registration, factory, entry-point template | [`DESIGN.ABI.md`](./DESIGN.ABI.md) — `PhysicsEngine` ABC contracts, state/command/ROS invariants | Before writing the first line of engine code |
| 🧪 **Test / CI engineer** | [`DESIGN.CORE.md §9`](./DESIGN.CORE.md#9-abi-enforcement--test-suite) — `ABITestSuite`, `MockCppBridge`, enforcement hierarchy | [`DESIGN.ABI.md §12`](./DESIGN.ABI.md#13-unit-test-mock-strategy) — mock boundaries, `MockPhysicsEngine` skeleton | Before writing test fixtures |
| 🤖 **RL researcher** | [`DESIGN.RL.md`](./DESIGN.RL.md) — RL training architecture, vectorized envs, CUDA graph management | [`DESIGN.ABI.md §14`](./DESIGN.ABI.md#14-rl-reuse-path) — what RL consumes/reuses from the engine | Before implementing the training loop |
| 🔬 **Physics / solver engineer** (contact, cloth) | [`DESIGN.ALIPC.md`](./DESIGN.ALIPC.md) — AL-IPC inner loop, two-phase CUDA graphs, performance budget | [`DESIGN.CORE.md §4`](./DESIGN.CORE.md#4--genie_sim_engine_-contract) — how to wrap as a `PhysicsEngine` plugin | Before implementing a new solver adapter |
| 🔧 **Package maintainer** (splitting the monolith) | [`DESIGN.CORE.md §8`](./DESIGN.CORE.md#8-migration-path) — phased migration plan, import table, file moves | [`DESIGN.CORE.md §10`](./DESIGN.CORE.md#10-file-inventory--moves) — every file's destination | Before starting Phase 1 of the split |
| 👤 **Newcomer / overview** | This file (`DESIGN.md`) — document landscape, reading guide | — | First thing |

---

## 📋 Quick Reference

| Doc | Lines | Status | Audience | Key sections |
|---|---|---|---|---|
| [`DESIGN.CORE.md`](./DESIGN.CORE.md) | ~800 | ✅ Active | Backend devs, maintainers, test engineers | §1 Architecture · §4 Backend contract · §5 Registration · §9 ABI test suite · §10 File moves |
| [`DESIGN.ABI.md`](./DESIGN.ABI.md) | ~820 | ✅ Active | Engine implementers, RL devs, test engineers | §1 Module diagram · §4 Lifecycle · §5 State readback · §7-8 ROS/C++ bridge · §12 Mock strategy · §14 RL path |
| [`DESIGN.RL.md`](./DESIGN.RL.md) | ~1310 | 📝 Draft | RL researchers | §7 DirectRLEnv · §8 ManagerBasedRLEnv · §9 NewtonSimContext CUDA graphs · §15 Engine restructuring · §16 IsaacLab lessons |
| [`DESIGN.ALIPC.md`](./DESIGN.ALIPC.md) | ~1405 | 📝 Draft | Physics/solver engineers | §4 Architecture · §5 Two-phase CUDA graph · §6 AL-IPC inner loop · §8 CUDA graph tradeoffs · §14 Genesis patterns |

---

## 🔗 Cross-Reference Quickmap

### DESIGN.CORE.md references

| Where | What | Related doc |
|---|---|---|
| §3.1 | Design docs owned by core | ABI, RL, ALIPC |
| §9 | ABI enforcement via `ABITestSuite` | ABI §12 |

### DESIGN.ABI.md references

| Where | What | Related doc |
|---|---|---|
| §2.3 | "Not in scope — covered by" | RL, ALIPC |
| §14.3 | Phase B: NewtonPhysicsCore extraction | RL §15 |

### DESIGN.RL.md references

| Where | What | Related doc |
|---|---|---|
| §1, §4, etc. | Depends on AL-IPC contact | ALIPC |
| §15 | Engine restructuring follows ABI contract | ABI §14 |

### DESIGN.ALIPC.md references

| Where | What | Related doc |
|---|---|---|
| §7 | Integration map: AL-IPC adapter slots into Newton pipeline | RL (consumes) |
| — | No explicit filename references to other DESIGN docs | — |

---

## 🔄 Update Rules

| When this changes... | Update these docs |
|---|---|
| ✨ New backend added (`genie_sim_engine_foo`) | CORE §4 (contract), §10 (file inventory) |
| 🔧 `PhysicsEngine` ABC method signature changes | ABI (that section) + CORE §9 (ABITestSuite) + all backend implementations |
| 🚀 ROS topic name / message type changes | ABI §7-8 |
| 🧪 New test category added | CORE §9 (ABITestSuite) + ABI §12 (mock strategy) |
| 🤖 RL env implementation begins | RL (entire doc) + ABI §14 (RL path) |
| 🔬 AL-IPC adapter implementation begins | ALIPC (entire doc) |

---

## 📄 Document Lifecycle

```mermaid
flowchart LR
    subgraph Stable["✅ Active"]
        CORE["DESIGN.CORE.md<br/>established architecture"]
        ABI["DESIGN.ABI.md<br/>established contracts"]
    end

    subgraph Draft["📝 Design Proposal"]
        RL["DESIGN.RL.md<br/>awaiting implementation"]
        ALIPC["DESIGN.ALIPC.md<br/>awaiting implementation"]
    end

    subgraph History["🗄️ Superseded"]
        ARCH["DESIGN.ARCH.md<br/>→ renamed to CORE"]
    end

    ARCH -->|"renamed"| CORE
    CORE -.->|"when backends land"| RL
    CORE -.->|"when adapter lands"| ALIPC

    RL -.->|"after implementation"| Active
    ALIPC -.->|"after implementation"| Active

    style Stable fill:#90EE90
    style Draft fill:#FFD700
    style History fill:#D3D3D3
```

---

## 🗂️ File locations

All design docs live at the workspace root alongside the source tree they describe:

```
source/geniesim_ros/src/ros_ws/src/genie_sim_engine/
├── DESIGN.md         ← YOU ARE HERE
├── DESIGN.CORE.md    ← Unified backend architecture
├── DESIGN.ABI.md     ← Common engine ABI
├── DESIGN.RL.md      ← RL training architecture
└── DESIGN.ALIPC.md   ← AL-IPC contact augmentation
```

After the core extraction (Phase 1), `DESIGN.CORE.md`, `DESIGN.ABI.md`, and
this file (`DESIGN.md`) move to `genie_sim_engine_core/`. The implementation
proposals (`RL.md`, `ALIPC.md`) stay alongside their respective backends or
move to the core depending on scope. See
[`DESIGN.CORE.md §10`](./DESIGN.CORE.md#10-file-inventory--moves) for the
exact migration table.

---

## Revision History

| Date | Change |
|---|---|
| 2026-07-12 | Initial version. Document index with dependency graph, reading guide, quick-reference table, cross-reference map, and update rules. |
