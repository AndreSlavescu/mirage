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

#include "mirage/transpiler/pdl/pdl.h"

namespace mirage {
namespace transpiler {
namespace pdl {

std::string get_pdl_runtime_header() {
  CodeKeeper code;
  code.e("// True PDL uses cudaLaunchKernelEx - no device-side code needed");
  return code.to_string();
}

std::string get_pdl_init_function(size_t num_slots) {
  return "";
}

std::string generate_pdl_signal_epilogue(size_t slot_id,
                                         std::string const &pdl_buffer_var) {
  return "";
}

std::string generate_pdl_wait_prologue(size_t slot_id,
                                       std::string const &pdl_buffer_var) {
  return "";
}

std::string generate_pdl_kernel_launch(std::string const &func_name,
                                       std::string const &grid_dim,
                                       std::string const &block_dim,
                                       std::string const &smem_size,
                                       std::string const &stream,
                                       std::string const &args,
                                       bool is_dependent_kernel,
                                       int target_cc) {
  CodeKeeper code;

  if (!is_dependent_kernel) {
    code.e("$<<<$, $, $, $>>>($);", func_name, grid_dim, block_dim, smem_size,
           stream, args);
    return code.to_string();
  }

  code.e("{");
  code.e("  cudaLaunchAttribute pdl_attrs[1];");

  if (target_cc >= 100) {
    code.e("  pdl_attrs[0].id = cudaLaunchAttributeProgrammaticEvent;");
    code.e("  pdl_attrs[0].val.programmaticEvent.triggerAtBlockStart = 0;");
    code.e("  pdl_attrs[0].val.programmaticEvent.flags = 0;");
  } else {
    code.e("  pdl_attrs[0].id = "
           "cudaLaunchAttributeProgrammaticStreamSerialization;");
    code.e("  pdl_attrs[0].val.programmaticStreamSerializationAllowed = 1;");
  }

  code.e("  cudaLaunchConfig_t pdl_config;");
  code.e("  pdl_config.gridDim = $;", grid_dim);
  code.e("  pdl_config.blockDim = $;", block_dim);
  code.e("  pdl_config.dynamicSmemBytes = $;", smem_size);
  code.e("  pdl_config.stream = $;", stream);
  code.e("  pdl_config.attrs = pdl_attrs;");
  code.e("  pdl_config.numAttrs = 1;");

  code.e("  void* pdl_args[] = {$};", args);
  code.e("  cudaLaunchKernelEx(&pdl_config, $, pdl_args);", func_name);
  code.e("}");

  return code.to_string();
}

std::string generate_pdl_kernel_launch_with_tma(std::string const &func_name,
                                                std::string const &grid_dim,
                                                std::string const &block_dim,
                                                std::string const &smem_size,
                                                std::string const &stream,
                                                std::string const &tma_args,
                                                std::string const &ptr_args,
                                                bool is_dependent_kernel,
                                                int target_cc) {
  CodeKeeper code;

  if (!is_dependent_kernel) {
    code.e("$<<<$, $, $, $>>>($ $);", func_name, grid_dim, block_dim, smem_size,
           stream, tma_args, ptr_args);
    return code.to_string();
  }

  code.e("{");
  code.e("  cudaLaunchAttribute pdl_attrs[1];");

  if (target_cc >= 100) {
    code.e("  pdl_attrs[0].id = cudaLaunchAttributeProgrammaticEvent;");
    code.e("  pdl_attrs[0].val.programmaticEvent.triggerAtBlockStart = 0;");
    code.e("  pdl_attrs[0].val.programmaticEvent.flags = 0;");
  } else {
    code.e("  pdl_attrs[0].id = "
           "cudaLaunchAttributeProgrammaticStreamSerialization;");
    code.e("  pdl_attrs[0].val.programmaticStreamSerializationAllowed = 1;");
  }

  code.e("  cudaLaunchConfig_t pdl_config;");
  code.e("  pdl_config.gridDim = $;", grid_dim);
  code.e("  pdl_config.blockDim = $;", block_dim);
  code.e("  pdl_config.dynamicSmemBytes = $;", smem_size);
  code.e("  pdl_config.stream = $;", stream);
  code.e("  pdl_config.attrs = pdl_attrs;");
  code.e("  pdl_config.numAttrs = 1;");

  code.e("  void* pdl_args[] = {$ $};", tma_args, ptr_args);
  code.e("  cudaLaunchKernelEx(&pdl_config, $, pdl_args);", func_name);
  code.e("}");

  return code.to_string();
}

}  // namespace pdl
}  // namespace transpiler
}  // namespace mirage
