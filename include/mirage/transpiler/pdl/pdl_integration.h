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
#include "mirage/transpiler/pdl/pdl_codegen.h"
#include "mirage/transpiler/pdl/pdl_pass.h"
#include "mirage/transpiler/structs.h"
#include "mirage/transpiler/utils.h"

namespace mirage {
namespace transpiler {
namespace pdl {

class PDLTranspilerIntegration {
public:
  PDLTranspilerIntegration(kernel::Graph const *graph,
                           TranspilerConfig const &config)
      : graph_(graph), config_(config), analysis_done_(false) {}

  bool run_analysis() {
    if (analysis_done_) {
      return result_.success;
    }

    PDLConfig pdl_config;
    pdl_config.target_cc = config_.target_cc;
    pdl_config.enable_global_barrier = config_.pdl_enable_global_barrier;
    pdl_config.fallback_on_unsupported = config_.pdl_fallback_on_unsupported;

    if (config_.target_cc >= 100) {
      pdl_config.mode = PDLMode::PROGRAMMATIC_DEPENDENT_LAUNCH;
    } else if (config_.target_cc >= 90) {
      pdl_config.mode = PDLMode::PROGRAMMATIC_STREAM_SERIALIZATION;
    } else {
      pdl_config.mode = PDLMode::DISABLED;
    }

    result_ = run_pdl_pass(graph_, pdl_config);
    analysis_done_ = true;

    return result_.success;
  }

  std::string generate_pdl_header() const {
    if (!is_pdl_active()) {
      return "";
    }

    PDLCodeGenerator codegen(result_.plan);
    return codegen.generate_header();
  }

  std::string generate_pdl_init_code(std::string const &pdl_buffer_var) const {
    if (!is_pdl_active()) {
      return "";
    }

    CodeKeeper code;
    code.e("// Initialize PDL synchronization slots");
    code.e("init_pdl_slots($);", pdl_buffer_var);
    return code.to_string();
  }

  std::string inject_kernel_signal(size_t op_index,
                                   std::string const &pdl_buffer_var) const {
    if (!is_pdl_active()) {
      return "";
    }

    KernelId kernel_id = op_index_to_kernel_id(op_index);
    if (kernel_id == static_cast<KernelId>(-1)) {
      return "";
    }

    PDLCodeGenerator codegen(result_.plan);
    return codegen.generate_signal_code(kernel_id, pdl_buffer_var);
  }

  std::string inject_kernel_wait(size_t op_index,
                                 std::string const &pdl_buffer_var) const {
    if (!is_pdl_active()) {
      return "";
    }

    KernelId kernel_id = op_index_to_kernel_id(op_index);
    if (kernel_id == static_cast<KernelId>(-1)) {
      return "";
    }

    PDLCodeGenerator codegen(result_.plan);
    return codegen.generate_wait_code(kernel_id, pdl_buffer_var);
  }

  std::string generate_kernel_launch(size_t op_index,
                                     std::string const &kernel_name,
                                     std::string const &grid_dim,
                                     std::string const &block_dim,
                                     std::string const &smem_size,
                                     std::string const &stream,
                                     std::string const &args) const {
    if (!is_pdl_active()) {
      return fmt("$<<<$, $, $, $>>>($);",
                 kernel_name, grid_dim, block_dim, smem_size, stream, args);
    }

    KernelId kernel_id = op_index_to_kernel_id(op_index);
    if (kernel_id == static_cast<KernelId>(-1)) {
      return fmt("$<<<$, $, $, $>>>($);",
                 kernel_name, grid_dim, block_dim, smem_size, stream, args);
    }

    PDLCodeGenerator codegen(result_.plan);
    return codegen.generate_kernel_launch_ex(
        kernel_name, grid_dim, block_dim, smem_size, stream, args, kernel_id);
  }

  bool is_pdl_active() const {
    return analysis_done_ && result_.success &&
           result_.plan.chains.size() > 0 && config_.enable_pdl;
  }

  size_t get_pdl_buffer_size() const {
    if (!is_pdl_active()) {
      return 0;
    }
    return result_.plan.required_buffer_size();
  }

  void populate_transpile_result(TranspileResult &result) const {
    result.pdl_enabled = is_pdl_active();
    if (is_pdl_active()) {
      result.pdl_chains_count = result_.plan.chains.size();
      result.pdl_kernels_optimized = result_.kernels_optimized;
      result.pdl_buffer_size = result_.plan.required_buffer_size();
    }
  }

  PDLAnalysisResult const &get_analysis_result() const { return result_; }

private:
  kernel::Graph const *graph_;
  TranspilerConfig config_;
  PDLAnalysisResult result_;
  bool analysis_done_;

  KernelId op_index_to_kernel_id(size_t op_index) const {
    size_t kernel_count = 0;
    size_t current_index = 0;

    for (auto const *op : graph_->operators) {
      if (op->op_type == type::KN_INPUT_OP ||
          op->op_type == type::KN_OUTPUT_OP) {
        current_index++;
        continue;
      }

      if (current_index == op_index) {
        return static_cast<KernelId>(kernel_count);
      }

      kernel_count++;
      current_index++;
    }

    return static_cast<KernelId>(-1);
  }
};

}  // namespace pdl
}  // namespace transpiler
}  // namespace mirage

