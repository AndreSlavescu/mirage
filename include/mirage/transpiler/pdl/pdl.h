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

#include "mirage/transpiler/pdl/chain_identifier.h"
#include "mirage/transpiler/pdl/dependency_analysis.h"
#include "mirage/transpiler/pdl/pdl_codegen.h"
#include "mirage/transpiler/pdl/pdl_integration.h"
#include "mirage/transpiler/pdl/pdl_pass.h"
#include "mirage/transpiler/pdl/pdl_types.h"

namespace mirage {
namespace transpiler {
namespace pdl {

std::string get_pdl_runtime_header();
std::string get_pdl_init_function(size_t num_slots);
std::string generate_pdl_signal_epilogue(size_t slot_id,
                                         std::string const& pdl_buffer_var);
std::string generate_pdl_wait_prologue(size_t slot_id,
                                       std::string const& pdl_buffer_var);

std::string generate_pdl_kernel_launch(std::string const &func_name,
                                       std::string const &grid_dim,
                                       std::string const &block_dim,
                                       std::string const &smem_size,
                                       std::string const &stream,
                                       std::string const &args,
                                       bool is_dependent_kernel,
                                       int target_cc);

std::string generate_pdl_kernel_launch_with_tma(std::string const &func_name,
                                                std::string const &grid_dim,
                                                std::string const &block_dim,
                                                std::string const &smem_size,
                                                std::string const &stream,
                                                std::string const &tma_args,
                                                std::string const &ptr_args,
                                                bool is_dependent_kernel,
                                                int target_cc);

inline bool is_pdl_supported(int compute_capability) {
  return is_pdl_supported_architecture(compute_capability);
}

inline std::string get_pdl_mode_string(PDLMode mode) {
  switch (mode) {
    case PDLMode::DISABLED:
      return "DISABLED";
    case PDLMode::PROGRAMMATIC_STREAM_SERIALIZATION:
      return "PROGRAMMATIC_STREAM_SERIALIZATION";
    case PDLMode::PROGRAMMATIC_DEPENDENT_LAUNCH:
      return "PROGRAMMATIC_DEPENDENT_LAUNCH";
    case PDLMode::CUDA_GRAPH_CONDITIONAL:
      return "CUDA_GRAPH_CONDITIONAL";
    default:
      return "UNKNOWN";
  }
}

inline void print_pdl_analysis_summary(PDLAnalysisResult const &result) {
  if (!result.success) {
    printf("PDL Analysis Failed: %s\n", result.error_message.c_str());
    return;
  }

  printf("=== PDL Analysis Summary ===\n");
  printf("Architecture: %s\n",
         get_architecture_name(result.plan.config.get_architecture()));
  printf("Mode: %s\n", get_pdl_mode_string(result.plan.config.mode).c_str());
  printf("Target CC: SM%d\n", result.plan.config.target_cc);
  printf("PDL Supported: %s\n",
         is_pdl_supported_architecture(result.plan.config.target_cc) ? "Yes" : "No");
  printf("Kernels in DAG: %zu\n", result.plan.dag.kernels.size());
  printf("Dependencies: %zu\n", result.plan.dag.dependencies.size());
  printf("Chains identified: %zu\n", result.chains_identified);
  printf("Kernels optimized: %zu\n", result.kernels_optimized);
  printf("Slots allocated: %zu\n", result.plan.total_slots());
  printf("Buffer size: %zu bytes\n", result.plan.required_buffer_size());
  printf("Est. latency reduction: ~%zu us\n",
         result.estimated_latency_reduction_us);
  printf("============================\n");

  for (size_t i = 0; i < result.plan.chains.size(); ++i) {
    auto const &chain = result.plan.chains[i];
    printf("Chain %zu: ", i);
    for (size_t j = 0; j < chain.kernel_sequence.size(); ++j) {
      auto const *kernel = result.plan.dag.get_kernel(chain.kernel_sequence[j]);
      if (kernel) {
        printf("%s", kernel->func_name.c_str());
      } else {
        printf("K%zu", chain.kernel_sequence[j]);
      }
      if (j + 1 < chain.kernel_sequence.size()) {
        printf(" -> ");
      }
    }
    printf("\n");
  }
}

}  // namespace pdl
}  // namespace transpiler
}  // namespace mirage

