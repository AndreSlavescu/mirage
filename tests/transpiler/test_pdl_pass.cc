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

#include "mirage/kernel/graph.h"
#include "mirage/transpiler/pdl/pdl.h"

#include <cassert>
#include <cstdio>

using namespace mirage;
using namespace mirage::transpiler::pdl;

void test_pdl_types() {
  printf("Testing PDL types...\n");

  PDLConfig config = PDLConfig::create_for_hopper();
  assert(config.target_cc == SM90);
  assert(config.is_enabled());
  assert(config.mode == PDLMode::PROGRAMMATIC_STREAM_SERIALIZATION);
  assert(config.get_architecture() == PDLArchitecture::HOPPER_SM90);

  PDLConfig blackwell_config = PDLConfig::create_for_blackwell();
  assert(blackwell_config.target_cc == SM100);
  assert(blackwell_config.mode == PDLMode::PROGRAMMATIC_DEPENDENT_LAUNCH);
  assert(blackwell_config.get_architecture() == PDLArchitecture::BLACKWELL_SM100);

  PDLConfig disabled_config;
  disabled_config.target_cc = 80;
  assert(!disabled_config.is_enabled());
  assert(!is_pdl_supported_architecture(80));

  PDLConfig auto_config = PDLConfig::create_for_cc(90);
  assert(auto_config.is_enabled());
  assert(auto_config.mode == PDLMode::PROGRAMMATIC_STREAM_SERIALIZATION);

  PDLConfig unsupported = PDLConfig::create_for_cc(86);
  assert(!unsupported.is_enabled());
  assert(unsupported.mode == PDLMode::DISABLED);

  printf("  PDL types: PASSED\n");
}

void test_kernel_dag() {
  printf("Testing Kernel DAG...\n");

  KernelDAG dag;

  KernelInfo k0;
  k0.id = 0;
  k0.func_name = "kernel_A";
  k0.type = KernelType::ONE_SHOT;
  k0.output_tensor_guids = {100};

  KernelInfo k1;
  k1.id = 1;
  k1.func_name = "kernel_B";
  k1.type = KernelType::ONE_SHOT;
  k1.input_tensor_guids = {100};
  k1.output_tensor_guids = {101};

  KernelInfo k2;
  k2.id = 2;
  k2.func_name = "kernel_C";
  k2.type = KernelType::ONE_SHOT;
  k2.input_tensor_guids = {101};

  dag.add_kernel(k0);
  dag.add_kernel(k1);
  dag.add_kernel(k2);

  KernelDependency dep01;
  dep01.producer = 0;
  dep01.consumer = 1;
  dep01.type = DependencyType::DATA_FLOW;
  dep01.shared_tensor_guids = {100};

  KernelDependency dep12;
  dep12.producer = 1;
  dep12.consumer = 2;
  dep12.type = DependencyType::DATA_FLOW;
  dep12.shared_tensor_guids = {101};

  dag.add_dependency(dep01);
  dag.add_dependency(dep12);

  assert(dag.kernels.size() == 3);
  assert(dag.dependencies.size() == 2);
  assert(dag.has_single_successor(0));
  assert(dag.has_single_predecessor(1));
  assert(dag.has_single_successor(1));
  assert(dag.has_single_predecessor(2));

  printf("  Kernel DAG: PASSED\n");
}

void test_chain_identification() {
  printf("Testing chain identification...\n");

  KernelDAG dag;

  for (int i = 0; i < 4; ++i) {
    KernelInfo k;
    k.id = i;
    k.func_name = "kernel_" + std::to_string(i);
    k.type = KernelType::ONE_SHOT;
    if (i > 0) {
      k.input_tensor_guids = {static_cast<size_t>(100 + i - 1)};
    }
    if (i < 3) {
      k.output_tensor_guids = {static_cast<size_t>(100 + i)};
    }
    dag.add_kernel(k);
  }

  for (int i = 0; i < 3; ++i) {
    KernelDependency dep;
    dep.producer = i;
    dep.consumer = i + 1;
    dep.type = DependencyType::DATA_FLOW;
    dep.shared_tensor_guids = {static_cast<size_t>(100 + i)};
    dag.add_dependency(dep);
  }

  dag.topological_order = {0, 1, 2, 3};

  PDLConfig config = PDLConfig::create_for_hopper();
  auto chains = identify_kernel_chains(dag, config);

  assert(chains.size() == 1);
  assert(chains[0].kernel_sequence.size() == 4);
  assert(chains[0].kernel_sequence[0] == 0);
  assert(chains[0].kernel_sequence[3] == 3);

  printf("  Chain identification: PASSED\n");
}

void test_slot_allocation() {
  printf("Testing slot allocation...\n");

  PDLSlotAllocator allocator;

  PDLSlot slot0 = allocator.allocate(0, 1);
  PDLSlot slot1 = allocator.allocate(1, 2);
  PDLSlot slot2 = allocator.allocate(2, 3);

  assert(slot0.id == 0);
  assert(slot1.id == 1);
  assert(slot2.id == 2);

  assert(slot0.global_mem_offset == 0);
  assert(slot1.global_mem_offset == 8);
  assert(slot2.global_mem_offset == 16);

  assert(allocator.total_buffer_size() == 24);

  printf("  Slot allocation: PASSED\n");
}

void test_code_generation() {
  printf("Testing code generation...\n");

  PDLPlan plan;
  plan.config = PDLConfig::create_for_hopper();

  KernelInfo k0, k1;
  k0.id = 0;
  k0.func_name = "kernel_A";
  k1.id = 1;
  k1.func_name = "kernel_B";

  plan.dag.add_kernel(k0);
  plan.dag.add_kernel(k1);

  PDLSlot slot;
  slot.id = 0;
  slot.producer = 0;
  slot.consumer = 1;
  slot.signal_value = PDL_SIGNAL_VALUE_READY;
  plan.slots.push_back(slot);

  PDLChain chain;
  chain.kernel_sequence = {0, 1};
  chain.slots = {0};
  plan.chains.push_back(chain);

  PDLCodeGenerator codegen(plan);

  std::string header = codegen.generate_header();
  assert(!header.empty());
  assert(header.find("pdl_signal") != std::string::npos);
  assert(header.find("pdl_wait") != std::string::npos);

  std::string signal_code = codegen.generate_signal_code(0, "pdl_buf");
  assert(!signal_code.empty());

  std::string wait_code = codegen.generate_wait_code(1, "pdl_buf");
  assert(!wait_code.empty());

  printf("  Code generation: PASSED\n");
}

void test_pdl_pass() {
  printf("Testing full PDL pass...\n");

  kernel::Graph graph;

  std::vector<int> dims = {1, 64, 64};
  std::vector<size_t> strides = {64 * 64, 64, 1};
  auto input = graph.new_input(dims, strides, type::DT_FLOAT16,
                               layout::DmemRowMajor);

  auto exp_result = graph.exp(input);
  auto silu_result = graph.silu(exp_result);

  graph.mark_output(silu_result);

  printf("  Testing with SM90 (Hopper)...\n");
  PDLConfig hopper_config = PDLConfig::create_for_hopper();
  auto hopper_result = run_pdl_pass(&graph, hopper_config);
  assert(hopper_result.success);
  printf("    Chains found: %zu\n", hopper_result.chains_identified);

  printf("  Testing with SM100 (Blackwell)...\n");
  PDLConfig blackwell_config = PDLConfig::create_for_blackwell();
  auto blackwell_result = run_pdl_pass(&graph, blackwell_config);
  assert(blackwell_result.success);
  printf("    Chains found: %zu\n", blackwell_result.chains_identified);

  printf("  Testing with unsupported SM86...\n");
  auto unsupported_result = run_pdl_pass(&graph, 86);
  assert(!unsupported_result.success || unsupported_result.chains_identified == 0);
  printf("    PDL correctly disabled for unsupported arch\n");

  print_pdl_analysis_summary(hopper_result);

  printf("  Full PDL pass: PASSED\n");
}

int main() {
  printf("=== PDL Pass Unit Tests ===\n\n");

  test_pdl_types();
  test_kernel_dag();
  test_chain_identification();
  test_slot_allocation();
  test_code_generation();
  test_pdl_pass();

  printf("\n=== All PDL Tests PASSED ===\n");
  return 0;
}

