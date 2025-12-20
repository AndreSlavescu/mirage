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

#include "mirage/transpiler/pdl/pdl_types.h"
#include "mirage/transpiler/utils.h"

#include <sstream>

namespace mirage {
namespace transpiler {
namespace pdl {

class PDLCodeGenerator {
public:
  explicit PDLCodeGenerator(PDLPlan const &plan) : plan_(plan) {}

  std::string generate_header() const {
    CodeKeeper code;

    code.e("namespace mirage_pdl {");
    code.e("");
    code.e("struct PDLSyncSlot {");
    code.e("  uint32_t signal_value;");
    code.e("  uint32_t counter;");
    code.e("};");
    code.e("");
    code.e("constexpr uint32_t PDL_READY = $;", PDL_SIGNAL_VALUE_READY);
    code.e("constexpr uint32_t PDL_NOT_READY = 0;");
    code.e("");

    code.e("__device__ __forceinline__ void pdl_global_barrier(");
    code.e("    uint32_t* counter_ptr, uint32_t grid_size) {");
    code.e("  __shared__ int is_last_block;");
    code.e("  __threadfence();");
    code.e("  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {");
    code.e("    uint32_t old = atomicAdd(counter_ptr, 1);");
    code.e("    is_last_block = (old == grid_size - 1);");
    code.e("  }");
    code.e("  __syncthreads();");
    code.e("}");
    code.e("");

    code.e("__device__ __forceinline__ void pdl_signal(");
    code.e("    uint32_t* signal_ptr, uint32_t value) {");
    code.e("  __threadfence_system();");
    code.e("  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {");
    code.e("    atomicExch(signal_ptr, value);");
    code.e("  }");
    code.e("}");
    code.e("");

    code.e("__device__ __forceinline__ void pdl_wait(");
    code.e("    uint32_t const* signal_ptr, uint32_t expected_value) {");
    code.e("  while (atomicAdd((uint32_t*)signal_ptr, 0) != expected_value) {");
    code.e("    __nanosleep(100);");
    code.e("  }");
    code.e("  __threadfence_system();");
    code.e("}");
    code.e("");

    code.e("} // namespace mirage_pdl");
    code.e("");

    return code.to_string();
  }

  std::string generate_slot_initialization() const {
    CodeKeeper code;

    code.e("void init_pdl_slots(void* pdl_buffer) {");
    code.e("  mirage_pdl::PDLSyncSlot* slots = ");
    code.e("      reinterpret_cast<mirage_pdl::PDLSyncSlot*>(pdl_buffer);");

    for (auto const &slot : plan_.slots) {
      code.e("  slots[$].signal_value = mirage_pdl::PDL_NOT_READY;", slot.id);
      code.e("  slots[$].counter = 0;", slot.id);
    }

    code.e("}");
    code.e("");

    return code.to_string();
  }

  std::string generate_signal_code(KernelId producer_id,
                                   std::string const &pdl_buffer_var) const {
    auto const *slot = plan_.get_signal_slot(producer_id);
    if (!slot) {
      return "";
    }

    CodeKeeper code;

    auto const *kernel = plan_.dag.get_kernel(producer_id);
    if (kernel && plan_.config.enable_global_barrier) {
      code.e("{");
      code.e("  mirage_pdl::PDLSyncSlot* slots = ");
      code.e("      reinterpret_cast<mirage_pdl::PDLSyncSlot*>($);",
             pdl_buffer_var);
      code.e("  uint32_t* counter = &slots[$].counter;", slot->id);
      code.e("  __shared__ int _pdl_is_last_block;");
      code.e("  __threadfence();");
      code.e("  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {");
      code.e("    uint32_t old = atomicAdd(counter, 1);");
      code.e("    _pdl_is_last_block = (old == gridDim.x * gridDim.y * gridDim.z - 1);");
      code.e("  }");
      code.e("  __syncthreads();");
      code.e("  if (_pdl_is_last_block && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {");
      code.e("    mirage_pdl::pdl_signal(&slots[$].signal_value, $);",
             slot->id, slot->signal_value);
      code.e("  }");
      code.e("}");
    }

    return code.to_string();
  }

  std::string generate_wait_code(KernelId consumer_id,
                                 std::string const &pdl_buffer_var) const {
    auto const *slot = plan_.get_wait_slot(consumer_id);
    if (!slot) {
      return "";
    }

    CodeKeeper code;

    code.e("{");
    code.e("  mirage_pdl::PDLSyncSlot* slots = ");
    code.e("      reinterpret_cast<mirage_pdl::PDLSyncSlot*>($);",
           pdl_buffer_var);
    code.e("  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&");
    code.e("      threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {");
    code.e("    mirage_pdl::pdl_wait(&slots[$].signal_value, $);",
           slot->id, slot->signal_value);
    code.e("  }");
    code.e("  __syncthreads();");
    code.e("}");

    return code.to_string();
  }

  std::string generate_host_launch_attributes(KernelId kernel_id,
                                              std::string const &attr_var) const {
    auto const *wait_slot = plan_.get_wait_slot(kernel_id);
    if (!wait_slot) {
      return "";
    }

    CodeKeeper code;

    if (plan_.config.mode == PDLMode::PROGRAMMATIC_STREAM_SERIALIZATION) {
      code.e("cudaLaunchAttribute $[1];", attr_var);
      code.e("$[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;",
             attr_var);
      code.e("$[0].val.programmaticStreamSerializationAllowed = 1;", attr_var);
    } else if (plan_.config.mode == PDLMode::PROGRAMMATIC_DEPENDENT_LAUNCH) {
      code.e("cudaLaunchAttribute $[1];", attr_var);
      code.e("$[0].id = cudaLaunchAttributeProgrammaticEvent;", attr_var);
      code.e("$[0].val.programmaticEvent.triggerAtBlockStart = 0;", attr_var);
      code.e("$[0].val.programmaticEvent.flags = 0;", attr_var);
    }

    return code.to_string();
  }

  std::string generate_kernel_launch_ex(std::string const &kernel_name,
                                        std::string const &grid_dim,
                                        std::string const &block_dim,
                                        std::string const &smem_size,
                                        std::string const &stream,
                                        std::string const &args,
                                        KernelId kernel_id) const {
    auto const *wait_slot = plan_.get_wait_slot(kernel_id);

    CodeKeeper code;

    if (wait_slot) {
      std::string attr_var = fmt("pdl_attr_$", kernel_id);
      code.e(generate_host_launch_attributes(kernel_id, attr_var));

      code.e("cudaLaunchConfig_t pdl_config_$;", kernel_id);
      code.e("pdl_config_$.gridDim = $;", kernel_id, grid_dim);
      code.e("pdl_config_$.blockDim = $;", kernel_id, block_dim);
      code.e("pdl_config_$.dynamicSmemBytes = $;", kernel_id, smem_size);
      code.e("pdl_config_$.stream = $;", kernel_id, stream);
      code.e("pdl_config_$.attrs = $;", kernel_id, attr_var);
      code.e("pdl_config_$.numAttrs = 1;", kernel_id);

      code.e("void* pdl_args_$[] = { $ };", kernel_id, args);
      code.e("cudaLaunchKernelEx(&pdl_config_$, $, pdl_args_$);",
             kernel_id, kernel_name, kernel_id);
    } else {
      code.e("$<<<$, $, $, $>>>($);",
             kernel_name, grid_dim, block_dim, smem_size, stream, args);
    }

    return code.to_string();
  }

  size_t get_required_buffer_size() const {
    return plan_.required_buffer_size();
  }

private:
  PDLPlan const &plan_;
};

}  // namespace pdl
}  // namespace transpiler
}  // namespace mirage

