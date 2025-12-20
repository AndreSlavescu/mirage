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

#include <unordered_set>

namespace mirage {
namespace transpiler {
namespace pdl {

class ChainIdentifier {
public:
  explicit ChainIdentifier(KernelDAG const &dag, PDLConfig const &config)
      : dag_(dag), config_(config) {}

  std::vector<PDLChain> identify_chains() {
    std::vector<PDLChain> chains;
    std::unordered_set<KernelId> visited;

    for (KernelId start_id : dag_.topological_order) {
      if (visited.count(start_id)) {
        continue;
      }

      if (!is_chain_start_candidate(start_id)) {
        continue;
      }

      PDLChain chain = build_chain_from(start_id, visited);
      if (chain.is_valid()) {
        chains.push_back(chain);
      }
    }

    return chains;
  }

private:
  KernelDAG const &dag_;
  PDLConfig const &config_;

  bool is_chain_start_candidate(KernelId id) const {
    auto const *kernel = dag_.get_kernel(id);
    if (!kernel || !kernel->is_pdl_compatible()) {
      return false;
    }

    if (!dag_.has_single_successor(id)) {
      return false;
    }

    auto succ_it = dag_.successors.find(id);
    if (succ_it == dag_.successors.end() || succ_it->second.empty()) {
      return false;
    }

    KernelId successor = succ_it->second[0];
    if (!dag_.has_single_predecessor(successor)) {
      return false;
    }

    auto const *dep = dag_.get_dependency(id, successor);
    if (!dep || !dep->is_strict()) {
      return false;
    }

    return true;
  }

  bool can_extend_chain(KernelId current, KernelId next) const {
    auto const *current_kernel = dag_.get_kernel(current);
    auto const *next_kernel = dag_.get_kernel(next);

    if (!current_kernel || !next_kernel) {
      return false;
    }

    if (!next_kernel->is_pdl_compatible()) {
      return false;
    }

    if (!dag_.has_single_predecessor(next)) {
      return false;
    }

    auto pred_it = dag_.predecessors.find(next);
    if (pred_it == dag_.predecessors.end() ||
        pred_it->second.size() != 1 ||
        pred_it->second[0] != current) {
      return false;
    }

    auto const *dep = dag_.get_dependency(current, next);
    if (!dep || !dep->is_strict()) {
      return false;
    }

    return true;
  }

  PDLChain build_chain_from(KernelId start,
                             std::unordered_set<KernelId> &visited) {
    PDLChain chain;
    chain.kernel_sequence.push_back(start);
    visited.insert(start);

    KernelId current = start;
    while (true) {
      auto succ_it = dag_.successors.find(current);
      if (succ_it == dag_.successors.end() || succ_it->second.empty()) {
        break;
      }

      if (succ_it->second.size() != 1) {
        break;
      }

      KernelId next = succ_it->second[0];

      if (visited.count(next)) {
        break;
      }

      if (!can_extend_chain(current, next)) {
        break;
      }

      chain.kernel_sequence.push_back(next);
      visited.insert(next);
      current = next;
    }

    return chain;
  }
};

inline std::vector<PDLChain> identify_kernel_chains(KernelDAG const &dag,
                                                     PDLConfig const &config) {
  ChainIdentifier identifier(dag, config);
  return identifier.identify_chains();
}

}  // namespace pdl
}  // namespace transpiler
}  // namespace mirage

