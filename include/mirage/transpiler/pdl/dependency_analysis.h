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
#include "mirage/transpiler/pdl/pdl_types.h"

#include <algorithm>
#include <queue>
#include <unordered_set>

namespace mirage {
namespace transpiler {
namespace pdl {

class DependencyAnalyzer {
public:
  explicit DependencyAnalyzer(kernel::Graph const *graph) : graph_(graph) {}

  KernelDAG analyze() {
    KernelDAG dag;

    build_kernel_info(dag);
    find_data_dependencies(dag);
    compute_topological_order(dag);

    return dag;
  }

private:
  kernel::Graph const *graph_;

  std::unordered_map<size_t, KernelId> tensor_producer_;
  std::unordered_map<size_t, std::vector<KernelId>> tensor_consumers_;

  void build_kernel_info(KernelDAG &dag) {
    KernelId kernel_id = 0;

    for (auto const *op : graph_->operators) {
      if (op->op_type == type::KN_INPUT_OP ||
          op->op_type == type::KN_OUTPUT_OP) {
        continue;
      }

      KernelInfo info;
      info.id = kernel_id++;
      info.func_name = get_kernel_func_name(op);
      info.type = determine_kernel_type(op);

      if (op->op_type == type::KN_CUSTOMIZED_OP) {
        auto const *custom_op =
            dynamic_cast<kernel::KNCustomizedOp const *>(op);
        info.grid_size = static_cast<size_t>(custom_op->bgraph.grid_dim.x) *
                         custom_op->bgraph.grid_dim.y *
                         custom_op->bgraph.grid_dim.z;
        info.block_size = static_cast<size_t>(custom_op->bgraph.block_dim.x) *
                          custom_op->bgraph.block_dim.y *
                          custom_op->bgraph.block_dim.z;
      }

      for (auto const &tensor : op->input_tensors) {
        info.input_tensor_guids.push_back(tensor.guid);
        tensor_consumers_[tensor.guid].push_back(info.id);
      }

      for (auto const &tensor : op->output_tensors) {
        info.output_tensor_guids.push_back(tensor.guid);
        tensor_producer_[tensor.guid] = info.id;
      }

      dag.add_kernel(info);
    }
  }

  void find_data_dependencies(KernelDAG &dag) {
    for (auto const &kernel : dag.kernels) {
      for (size_t input_guid : kernel.input_tensor_guids) {
        auto prod_it = tensor_producer_.find(input_guid);
        if (prod_it != tensor_producer_.end()) {
          KernelId producer_id = prod_it->second;

          auto const *existing = dag.get_dependency(producer_id, kernel.id);
          if (existing) {
            continue;
          }

          KernelDependency dep;
          dep.producer = producer_id;
          dep.consumer = kernel.id;
          dep.type = DependencyType::DATA_FLOW;
          dep.shared_tensor_guids.push_back(input_guid);

          dag.add_dependency(dep);
        }
      }
    }
  }

  void compute_topological_order(KernelDAG &dag) {
    std::unordered_map<KernelId, int> in_degree;
    for (auto const &k : dag.kernels) {
      in_degree[k.id] = 0;
    }
    for (auto const &dep : dag.dependencies) {
      in_degree[dep.consumer]++;
    }

    std::queue<KernelId> ready_queue;
    for (auto const &k : dag.kernels) {
      if (in_degree[k.id] == 0) {
        ready_queue.push(k.id);
      }
    }

    dag.topological_order.clear();
    while (!ready_queue.empty()) {
      KernelId current = ready_queue.front();
      ready_queue.pop();
      dag.topological_order.push_back(current);

      auto succ_it = dag.successors.find(current);
      if (succ_it != dag.successors.end()) {
        for (KernelId succ : succ_it->second) {
          in_degree[succ]--;
          if (in_degree[succ] == 0) {
            ready_queue.push(succ);
          }
        }
      }
    }
  }

  std::string get_kernel_func_name(kernel::KNOperator const *op) const {
    switch (op->op_type) {
      case type::KN_MATMUL_OP:
        return "matmul_kernel";
      case type::KN_CUSTOMIZED_OP:
        return "custom_kernel";
      case type::KN_EXP_OP:
        return "exp_kernel";
      case type::KN_SILU_OP:
        return "silu_kernel";
      case type::KN_ADD_OP:
        return "add_kernel";
      case type::KN_MUL_OP:
        return "mul_kernel";
      default:
        return "unknown_kernel";
    }
  }

  KernelType determine_kernel_type(kernel::KNOperator const *op) const {
    return KernelType::ONE_SHOT;
  }
};

inline KernelDAG analyze_kernel_dependencies(kernel::Graph const *graph) {
  DependencyAnalyzer analyzer(graph);
  return analyzer.analyze();
}

}  // namespace pdl
}  // namespace transpiler
}  // namespace mirage

