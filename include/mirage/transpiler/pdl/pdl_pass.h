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

#include "mirage/kernel/graph.h"
#include "mirage/transpiler/pdl/chain_identifier.h"
#include "mirage/transpiler/pdl/dependency_analysis.h"
#include "mirage/transpiler/pdl/pdl_types.h"

namespace mirage {
namespace transpiler {
namespace pdl {

class PDLSlotAllocator {
public:
  explicit PDLSlotAllocator(size_t base_offset = 0) : next_offset_(base_offset) {}

  PDLSlot allocate(KernelId producer, KernelId consumer) {
    PDLSlot slot;
    slot.id = next_slot_id_++;
    slot.global_mem_offset = next_offset_;
    slot.signal_value = PDL_SIGNAL_VALUE_READY;
    slot.producer = producer;
    slot.consumer = consumer;

    next_offset_ += sizeof(uint32_t) * 2;

    return slot;
  }

  size_t total_buffer_size() const { return next_offset_; }

private:
  SlotId next_slot_id_ = 0;
  size_t next_offset_ = 0;
};

class AutoPDLPass {
public:
  explicit AutoPDLPass(kernel::Graph const *graph, PDLConfig const &config)
      : graph_(graph), config_(config) {}

  PDLAnalysisResult run() {
    PDLAnalysisResult result;

    if (!check_hardware_compatibility()) {
      result.success = false;
      result.error_message = "Hardware does not support PDL (requires CC >= 9.0)";
      return result;
    }

    result.plan.config = config_;
    result.plan.dag = analyze_kernel_dependencies(graph_);

    if (result.plan.dag.kernels.empty()) {
      result.success = true;
      result.error_message = "No kernels to optimize";
      return result;
    }

    result.plan.chains = identify_kernel_chains(result.plan.dag, config_);

    if (result.plan.chains.empty()) {
      result.success = true;
      result.error_message = "No strict chains identified";
      return result;
    }

    allocate_slots(result.plan);

    result.success = true;
    result.chains_identified = result.plan.chains.size();

    for (auto const &chain : result.plan.chains) {
      result.kernels_optimized += chain.length();
    }

    result.estimated_latency_reduction_us =
        result.plan.slots.size() * 5;

    return result;
  }

private:
  kernel::Graph const *graph_;
  PDLConfig config_;

  bool check_hardware_compatibility() const {
    return is_pdl_supported_architecture(config_.target_cc);
  }

  void allocate_slots(PDLPlan &plan) {
    PDLSlotAllocator allocator;

    for (auto &chain : plan.chains) {
      chain.slots.clear();

      for (size_t i = 0; i + 1 < chain.kernel_sequence.size(); ++i) {
        KernelId producer = chain.kernel_sequence[i];
        KernelId consumer = chain.kernel_sequence[i + 1];

        PDLSlot slot = allocator.allocate(producer, consumer);
        chain.slots.push_back(slot.id);
        plan.slots.push_back(slot);
      }
    }
  }
};

inline PDLAnalysisResult run_pdl_pass(kernel::Graph const *graph,
                                      PDLConfig const &config) {
  AutoPDLPass pass(graph, config);
  return pass.run();
}

inline PDLAnalysisResult run_pdl_pass(kernel::Graph const *graph,
                                      int target_cc) {
  PDLConfig config = PDLConfig::create_for_cc(target_cc);
  return run_pdl_pass(graph, config);
}

}  // namespace pdl
}  // namespace transpiler
}  // namespace mirage

