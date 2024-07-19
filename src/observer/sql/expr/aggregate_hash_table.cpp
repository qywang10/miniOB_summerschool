/* Copyright (c) 2021 OceanBase and/or its affiliates. All rights reserved.
miniob is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

#include "sql/expr/aggregate_hash_table.h"

// ----------------------------------StandardAggregateHashTable------------------

RC StandardAggregateHashTable::add_chunk(Chunk &groups_chunk, Chunk &aggrs_chunk)
{
  for(int i = 0;i < groups_chunk.rows();i ++) {
    std::vector<Value> group_vals, aggrs_vals;
    for(int j = 0;j < groups_chunk.column_num();j ++) {
      group_vals.push_back(groups_chunk.get_value(j, i));
    }
    for(int j = 0;j < aggrs_chunk.column_num();j ++) {
      // std::cout << aggrs_chunk.column_num() << " " << aggrs_chunk.rows() << " " << j << " " << i << std::endl;
      aggrs_vals.push_back(aggrs_chunk.get_value(j, i));
    }

    // do aggregation
    if(aggr_values_.count(group_vals) != 0) {
      // aggregate the aggrs_vals here
      for (size_t aggr_idx = 0; aggr_idx < aggrs_chunk.column_num(); aggr_idx++) {
        if(aggr_values_[group_vals].at(aggr_idx).attr_type() == AttrType::INTS) {
          int old_val = aggr_values_[group_vals].at(aggr_idx).get_int();
          // std::cout << "old val: " << old_val << " " << aggrs_vals[aggr_idx].get_float() << std::endl;
          aggr_values_[group_vals].at(aggr_idx).set_int(aggrs_vals[aggr_idx].get_int() + old_val);
        } else if(aggr_values_[group_vals].at(aggr_idx).attr_type() == AttrType::FLOATS) {
          float old_val = aggr_values_[group_vals].at(aggr_idx).get_float();
          // std::cout << "old val: " << old_val << " " << aggrs_vals[aggr_idx].get_float() << std::endl;
          aggr_values_[group_vals].at(aggr_idx).set_float(aggrs_vals[aggr_idx].get_float() + old_val);
        }
      }
    } else {
      aggr_values_[group_vals] = aggrs_vals;
    }
  }
  return RC::SUCCESS;
}

void StandardAggregateHashTable::Scanner::open_scan()
{
  it_  = static_cast<StandardAggregateHashTable *>(hash_table_)->begin();
  end_ = static_cast<StandardAggregateHashTable *>(hash_table_)->end();
}

RC StandardAggregateHashTable::Scanner::next(Chunk &output_chunk)
{
  if (it_ == end_) {
    return RC::RECORD_EOF;
  }
  while (it_ != end_ && output_chunk.rows() <= output_chunk.capacity()) {
    auto &group_by_values = it_->first;
    auto &aggrs           = it_->second;
    for (int i = 0; i < output_chunk.column_num(); i++) {
      auto col_idx = output_chunk.column_ids(i);
      if (col_idx >= static_cast<int>(group_by_values.size())) {
        output_chunk.column(i).append_one((char *)aggrs[col_idx - group_by_values.size()].data());
      } else {
        output_chunk.column(i).append_one((char *)group_by_values[col_idx].data());
      }
    }
    it_++;
  }
  if (it_ == end_) {
    return RC::SUCCESS;
  }

  return RC::SUCCESS;
}

size_t StandardAggregateHashTable::VectorHash::operator()(const vector<Value> &vec) const
{
  size_t hash = 0;
  for (const auto &elem : vec) {
    hash ^= std::hash<string>()(elem.to_string());
  }
  return hash;
}

bool StandardAggregateHashTable::VectorEqual::operator()(const vector<Value> &lhs, const vector<Value> &rhs) const
{
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (rhs[i].compare(lhs[i]) != 0) {
      return false;
    }
  }
  return true;
}

// ----------------------------------LinearProbingAggregateHashTable------------------
#ifdef USE_SIMD
template <typename V>
RC LinearProbingAggregateHashTable<V>::add_chunk(Chunk &group_chunk, Chunk &aggr_chunk)
{
  if (group_chunk.column_num() != 1 || aggr_chunk.column_num() != 1) {
    LOG_WARN("group_chunk and aggr_chunk size must be 1.");
    return RC::INVALID_ARGUMENT;
  }
  if (group_chunk.rows() != aggr_chunk.rows()) {
    LOG_WARN("group_chunk and aggr _chunk rows must be equal.");
    return RC::INVALID_ARGUMENT;
  }
  add_batch((int *)group_chunk.column(0).data(), (V *)aggr_chunk.column(0).data(), group_chunk.rows());
  return RC::SUCCESS;
}

template <typename V>
void LinearProbingAggregateHashTable<V>::Scanner::open_scan()
{
  capacity_   = static_cast<LinearProbingAggregateHashTable *>(hash_table_)->capacity();
  size_       = static_cast<LinearProbingAggregateHashTable *>(hash_table_)->size();
  scan_pos_   = 0;
  scan_count_ = 0;
}

template <typename V>
RC LinearProbingAggregateHashTable<V>::Scanner::next(Chunk &output_chunk)
{
  if (scan_pos_ >= capacity_ || scan_count_ >= size_) {
    return RC::RECORD_EOF;
  }
  auto linear_probing_hash_table = static_cast<LinearProbingAggregateHashTable *>(hash_table_);
  while (scan_pos_ < capacity_ && scan_count_ < size_ && output_chunk.rows() <= output_chunk.capacity()) {
    int key;
    V   value;
    RC  rc = linear_probing_hash_table->iter_get(scan_pos_, key, value);
    if (rc == RC::SUCCESS) {
      output_chunk.column(0).append_one((char *)&key);
      output_chunk.column(1).append_one((char *)&value);
      scan_count_++;
    }
    scan_pos_++;
  }
  return RC::SUCCESS;
}

template <typename V>
void LinearProbingAggregateHashTable<V>::Scanner::close_scan()
{
  capacity_   = -1;
  size_       = -1;
  scan_pos_   = -1;
  scan_count_ = 0;
}

template <typename V>
RC LinearProbingAggregateHashTable<V>::get(int key, V &value)
{
  RC  rc          = RC::SUCCESS;
  int index       = (key % capacity_ + capacity_) % capacity_;
  int iterate_cnt = 0;
  while (true) {
    if (keys_[index] == EMPTY_KEY) {
      rc = RC::NOT_EXIST;
      break;
    } else if (keys_[index] == key) {
      value = values_[index];
      break;
    } else {
      index += 1;
      index %= capacity_;
      iterate_cnt++;
      if (iterate_cnt > capacity_) {
        rc = RC::NOT_EXIST;
        break;
      }
    }
  }
  return rc;
}

template <typename V>
RC LinearProbingAggregateHashTable<V>::iter_get(int pos, int &key, V &value)
{
  RC rc = RC::SUCCESS;
  if (keys_[pos] == LinearProbingAggregateHashTable<V>::EMPTY_KEY) {
    rc = RC::NOT_EXIST;
  } else {
    key   = keys_[pos];
    value = values_[pos];
  }
  return rc;
}

template <typename V>
void LinearProbingAggregateHashTable<V>::aggregate(V *value, V value_to_aggregate)
{
  if (aggregate_type_ == AggregateExpr::Type::SUM) {
    *value += value_to_aggregate;
  } else {
    ASSERT(false, "unsupported aggregate type");
  }
}

template <typename V>
void LinearProbingAggregateHashTable<V>::resize()
{
  capacity_ *= 2;
  std::vector<int> new_keys(capacity_);
  std::vector<V>   new_values(capacity_);

  for (size_t i = 0; i < keys_.size(); i++) {
    auto &key   = keys_[i];
    auto &value = values_[i];
    if (key != EMPTY_KEY) {
      int index = (key % capacity_ + capacity_) % capacity_;
      while (new_keys[index] != EMPTY_KEY) {
        index = (index + 1) % capacity_;
      }
      new_keys[index]   = key;
      new_values[index] = value;
    }
  }

  keys_   = std::move(new_keys);
  values_ = std::move(new_values);
}

template <typename V>
void LinearProbingAggregateHashTable<V>::resize_if_need()
{
  if (size_ >= capacity_ / 2) {
    resize();
  }
}

template <typename V>
void LinearProbingAggregateHashTable<V>::add_batch(int *input_keys, V *input_values, int len)
{
  // your code here
  exit(-1);

  // inv (invalid) 表示是否有效，inv[i] = -1 表示有效，inv[i] = 0 表示无效。
  // key[SIMD_WIDTH],value[SIMD_WIDTH] 表示当前循环中处理的键值对。
  // off (offset) 表示线性探测冲突时的偏移量，key[i] 每次遇到冲突键，则off[i]++，如果key[i] 已经完成聚合，则off[i] = 0，
  // i = 0 表示selective load 的起始位置。
  // inv 全部初始化为 -1
  // off 全部初始化为 0

  // for (; i + SIMD_WIDTH <= len;) {
    // 1: 根据 `inv` 变量的值，从 `input_keys` 中 `selective load` `SIMD_WIDTH` 个不同的输入键值对。
    // 2. 计算 i += |inv|, `|inv|` 表示 `inv` 中有效的个数 
    // 3. 计算 hash 值，
    // 4. 根据聚合类型（目前只支持 sum），在哈希表中更新聚合结果。如果本次循环，没有找到key[i] 在哈希表中的位置，则不更新聚合结果。
    // 5. gather 操作，根据 hash 值将 keys_ 的 gather 结果写入 table_key 中。
    // 6. 更新 inv 和 off。如果本次循环key[i] 聚合完成，则inv[i]=-1，表示该位置在下次循环中读取新的键值对。
    // 如果本次循环 key[i] 未在哈希表中聚合完成（table_key[i] != key[i]），则inv[i] = 0，表示该位置在下次循环中不需要读取新的键值对。
    // 如果本次循环中，key[i]聚合完成，则off[i] 更新为 0，表示线性探测偏移量为 0，key[i] 未完成聚合，则off[i]++,表示线性探测偏移量加 1。
  // }
  //7. 通过标量线性探测，处理剩余键值对

  int inv[SIMD_WIDTH];
  int off[SIMD_WIDTH];
  memset(inv, -1, sizeof(inv)); // Initialize inv to -1
  memset(off, 0, sizeof(off));  // Initialize off to 0

  std::cout << "cap " << capacity_ << std::endl;

  int i = 0;
  __m256i keys = _mm256_setzero_si256();
  __m256i values = _mm256_setzero_si256();
  __m256i hash_vals = _mm256_setzero_si256();
  __m256i table_keys;

  while (i + SIMD_WIDTH <= len) {
      for (int j = 0; j < SIMD_WIDTH; ++j) {
          if (inv[j] == -1 && i < len) {
              std::cout << "insert pos " << j << " key: " << input_keys[i] << std::endl;
              keys = insert_value(keys, input_keys[i], j);
              values = insert_value(values, input_values[i], j);
              inv[j] = 0;
              ++i;
          }
      }

      // Calculate hash values

      for (int j = 0; j < SIMD_WIDTH; ++j) {
        if(inv[j] != -1) {
          int key = mm256_extract_epi32_var_indx(keys, j);
          int hash_val = hash_function(key + off[j]);
          std::cout << "cal hash vals, pos: " << j << " key: " << key << " hash val: " << hash_val << std::endl;
          hash_vals = insert_value(hash_vals, hash_val, j);
        }
      }

      // aggregate
      for (int j = 0; j < SIMD_WIDTH; ++j) {
          if(inv[j] == -1) continue;
          int key = mm256_extract_epi32_var_indx(keys, j);
          int hash_val = hash_function(key + off[j]);

          if (keys_[hash_val] == key) {
              values_[hash_val] += mm256_extract_epi32_var_indx(values, j);
              inv[j] = -1; // Mark as done
              off[j] = 0;
          }
      }

      // Gather operation
      table_keys = _mm256_i32gather_epi32(keys_.data(), hash_vals, 4);

      for (int j = 0; j < SIMD_WIDTH; ++j) {
          if(inv[j] == -1) continue;
          int key = mm256_extract_epi32_var_indx(keys, j);
          int table_key = mm256_extract_epi32_var_indx(table_keys, j);
          int hash_val = mm256_extract_epi32_var_indx(hash_vals, j);

          if(table_key == EMPTY_KEY) {
              if(keys_[hash_val] != EMPTY_KEY && keys_[hash_val] != key) {
                off[j]++;
              } else if(keys_[hash_val] != EMPTY_KEY && keys_[hash_val] == key) {
                values_[hash_val] += mm256_extract_epi32_var_indx(values, j);
                inv[j] = -1; // Mark as done
                off[j] = 0;
              } else {
                // std::cout << "key " << key << "insert into empty " << hash_val << std::endl;
                keys_[hash_val] = key;
                values_[hash_val] = mm256_extract_epi32_var_indx(values, j);
                inv[j] = -1; // Mark as done
                off[j] = 0;
                size_++;
              }
          } else {
              off[j]++;
          }
      }
  }

  for (int j = 0; j < SIMD_WIDTH; ++j) {
      if(inv[j] == -1) continue;
      int key = mm256_extract_epi32_var_indx(keys, j);
      int hash_val = hash_function(key + off[j]);

      if (keys_[hash_val] == key) {
          values_[hash_val] += mm256_extract_epi32_var_indx(values, j);
          inv[j] = -1; // Mark as done
          off[j] = 0;
      }
  }

  // Process remaining elements
  while (i < len) {
      int key = input_keys[i];
      V value = input_values[i];
      int hash_val = hash_function(key);

      while (keys_[hash_val] != key && keys_[hash_val] != EMPTY_KEY) {
          hash_val = (hash_val + 1) % capacity_;
      }

      if (keys_[hash_val] == key) {
          values_[hash_val] += value;
      } else {
          keys_[hash_val] = key;
          values_[hash_val] = value;
          size_++;
      }
      ++i;
  }

  resize_if_need();
}

template <typename V>
const int LinearProbingAggregateHashTable<V>::EMPTY_KEY = 0xffffffff;
template <typename V>
const int LinearProbingAggregateHashTable<V>::DEFAULT_CAPACITY = 16384;

template class LinearProbingAggregateHashTable<int>;
template class LinearProbingAggregateHashTable<float>;
#endif