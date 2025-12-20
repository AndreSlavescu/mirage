/* Copyright 2023-2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mirage {
namespace transpiler {
namespace pdl {

using KernelId = size_t;
using SlotId = size_t;

constexpr uint32_t PDL_SIGNAL_VALUE_READY = 1;

constexpr int SM90 = 90;
constexpr int SM90a = 90;
constexpr int SM100 = 100;
constexpr int SM100a = 100;

enum class PDLMode {
  DISABLED,
  PROGRAMMATIC_STREAM_SERIALIZATION,
  PROGRAMMATIC_DEPENDENT_LAUNCH,
  CUDA_GRAPH_CONDITIONAL
};

enum class PDLArchitecture {
  UNSUPPORTED,
  HOPPER_SM90,
  HOPPER_SM90a,
  BLACKWELL_SM100,
  BLACKWELL_SM100a
};

inline PDLArchitecture get_pdl_architecture(int compute_capability) {
  switch (compute_capability) {
    case 90:
      return PDLArchitecture::HOPPER_SM90;
    case 100:
      return PDLArchitecture::BLACKWELL_SM100;
    default:
      return PDLArchitecture::UNSUPPORTED;
  }
}

inline bool is_pdl_supported_architecture(int compute_capability) {
  return compute_capability == SM90 || compute_capability == SM100;
}

inline char const *get_architecture_name(PDLArchitecture arch) {
  switch (arch) {
    case PDLArchitecture::HOPPER_SM90:
      return "Hopper (SM90)";
    case PDLArchitecture::HOPPER_SM90a:
      return "Hopper (SM90a)";
    case PDLArchitecture::BLACKWELL_SM100:
      return "Blackwell (SM100)";
    case PDLArchitecture::BLACKWELL_SM100a:
      return "Blackwell (SM100a)";
    default:
      return "Unsupported";
  }
}

struct PDLConfig {
  PDLMode mode = PDLMode::DISABLED;
  bool enable_global_barrier = true;
  bool fallback_on_unsupported = true;
  int target_cc = 0;

  bool is_enabled() const {
    return mode != PDLMode::DISABLED && is_pdl_supported_architecture(target_cc);
  }

  PDLArchitecture get_architecture() const {
    return get_pdl_architecture(target_cc);
  }

  static PDLConfig create_for_hopper() {
    PDLConfig cfg;
    cfg.mode = PDLMode::PROGRAMMATIC_STREAM_SERIALIZATION;
    cfg.target_cc = SM90;
    return cfg;
  }

  static PDLConfig create_for_blackwell() {
    PDLConfig cfg;
    cfg.mode = PDLMode::PROGRAMMATIC_DEPENDENT_LAUNCH;
    cfg.target_cc = SM100;
    return cfg;
  }

  static PDLConfig create_for_cc(int compute_capability) {
    PDLConfig cfg;
    cfg.target_cc = compute_capability;

    if (!is_pdl_supported_architecture(compute_capability)) {
      cfg.mode = PDLMode::DISABLED;
      return cfg;
    }

    if (compute_capability == SM100) {
      cfg.mode = PDLMode::PROGRAMMATIC_DEPENDENT_LAUNCH;
    } else if (compute_capability == SM90) {
      cfg.mode = PDLMode::PROGRAMMATIC_STREAM_SERIALIZATION;
    } else {
      cfg.mode = PDLMode::DISABLED;
    }

    return cfg;
  }
};

enum class KernelType {
  ONE_SHOT,
  PERSISTENT,
  UNKNOWN
};

struct KernelInfo {
  KernelId id;
  std::string func_name;
  KernelType type = KernelType::ONE_SHOT;
  size_t grid_size = 0;
  size_t block_size = 0;

  std::vector<size_t> input_tensor_guids;
  std::vector<size_t> output_tensor_guids;

  bool is_pdl_compatible() const {
    return type == KernelType::ONE_SHOT;
  }
};

enum class DependencyType {
  DATA_FLOW,
  CONTROL_FLOW,
  MEMORY_ALIAS
};

struct KernelDependency {
  KernelId producer;
  KernelId consumer;
  DependencyType type = DependencyType::DATA_FLOW;
  std::vector<size_t> shared_tensor_guids;

  bool is_strict() const {
    return type == DependencyType::DATA_FLOW && !shared_tensor_guids.empty();
  }
};

struct PDLSlot {
  SlotId id;
  size_t global_mem_offset = 0;
  uint32_t signal_value = PDL_SIGNAL_VALUE_READY;

  KernelId producer = 0;
  KernelId consumer = 0;
};

struct PDLChain {
  std::vector<KernelId> kernel_sequence;
  std::vector<SlotId> slots;

  bool is_valid() const {
    return kernel_sequence.size() >= 2;
  }

  bool is_fully_allocated() const {
    return is_valid() && slots.size() == kernel_sequence.size() - 1;
  }

  size_t length() const { return kernel_sequence.size(); }
};

struct KernelDAG {
  std::vector<KernelInfo> kernels;
  std::vector<KernelDependency> dependencies;

  std::unordered_map<KernelId, std::vector<KernelId>> successors;
  std::unordered_map<KernelId, std::vector<KernelId>> predecessors;

  std::vector<KernelId> topological_order;

  void add_kernel(KernelInfo const &info) {
    kernels.push_back(info);
    successors[info.id] = {};
    predecessors[info.id] = {};
  }

  void add_dependency(KernelDependency const &dep) {
    dependencies.push_back(dep);
    successors[dep.producer].push_back(dep.consumer);
    predecessors[dep.consumer].push_back(dep.producer);
  }

  KernelInfo const *get_kernel(KernelId id) const {
    for (auto const &k : kernels) {
      if (k.id == id) {
        return &k;
      }
    }
    return nullptr;
  }

  bool has_single_predecessor(KernelId id) const {
    auto it = predecessors.find(id);
    return it != predecessors.end() && it->second.size() == 1;
  }

  bool has_single_successor(KernelId id) const {
    auto it = successors.find(id);
    return it != successors.end() && it->second.size() == 1;
  }

  KernelDependency const *get_dependency(KernelId producer,
                                         KernelId consumer) const {
    for (auto const &dep : dependencies) {
      if (dep.producer == producer && dep.consumer == consumer) {
        return &dep;
      }
    }
    return nullptr;
  }
};

struct PDLPlan {
  PDLConfig config;
  KernelDAG dag;
  std::vector<PDLChain> chains;
  std::vector<PDLSlot> slots;

  size_t total_slots() const { return slots.size(); }

  size_t required_buffer_size() const {
    return slots.size() * sizeof(uint32_t) * 2;
  }

  bool has_chain_for_kernel(KernelId id) const {
    for (auto const &chain : chains) {
      for (auto kid : chain.kernel_sequence) {
        if (kid == id) {
          return true;
        }
      }
    }
    return false;
  }

  PDLSlot const *get_signal_slot(KernelId producer_id) const {
    for (auto const &slot : slots) {
      if (slot.producer == producer_id) {
        return &slot;
      }
    }
    return nullptr;
  }

  PDLSlot const *get_wait_slot(KernelId consumer_id) const {
    for (auto const &slot : slots) {
      if (slot.consumer == consumer_id) {
        return &slot;
      }
    }
    return nullptr;
  }
};

struct PDLAnalysisResult {
  bool success = false;
  std::string error_message;
  PDLPlan plan;

  size_t chains_identified = 0;
  size_t kernels_optimized = 0;
  size_t estimated_latency_reduction_us = 0;
};

}  // namespace pdl
}  // namespace transpiler
}  // namespace mirage

