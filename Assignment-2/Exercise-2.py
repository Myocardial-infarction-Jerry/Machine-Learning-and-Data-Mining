import math

# 表1的排名结果
ranking = [{
    "rank": 1,
    "docID": 51,
    "graded_relevance": 4,
    "binary_relevance": 1
}, {
    "rank": 2,
    "docID": 501,
    "graded_relevance": 1,
    "binary_relevance": 1
}, {
    "rank": 3,
    "docID": 21,
    "graded_relevance": 0,
    "binary_relevance": 0
}, {
    "rank": 4,
    "docID": 75,
    "graded_relevance": 3,
    "binary_relevance": 1
}, {
    "rank": 5,
    "docID": 321,
    "graded_relevance": 4,
    "binary_relevance": 1
}, {
    "rank": 6,
    "docID": 38,
    "graded_relevance": 1,
    "binary_relevance": 1
}, {
    "rank": 7,
    "docID": 521,
    "graded_relevance": 0,
    "binary_relevance": 0
}, {
    "rank": 8,
    "docID": 412,
    "graded_relevance": 1,
    "binary_relevance": 1
}, {
    "rank": 9,
    "docID": 331,
    "graded_relevance": 0,
    "binary_relevance": 0
}, {
    "rank": 10,
    "docID": 101,
    "graded_relevance": 2,
    "binary_relevance": 1
}]

# (a) 计算P@5和P@10
p_5 = sum(doc["binary_relevance"] for doc in ranking[:5]) / 5
p_10 = sum(doc["binary_relevance"] for doc in ranking[:10]) / 10

# (b) 计算R@5和R@10
relevant_docs = sum(doc["binary_relevance"] for doc in ranking)
r_5 = sum(doc["binary_relevance"] for doc in ranking[:5]) / relevant_docs
r_10 = sum(doc["binary_relevance"] for doc in ranking[:10]) / relevant_docs

# (c) 提供一个最大化P@5的示例排名
max_p_5_ranking = sorted(ranking,
                         key=lambda doc: doc["binary_relevance"],
                         reverse=True)[:5]

# (d) 提供一个最大化P@10的示例排名
max_p_10_ranking = sorted(ranking,
                          key=lambda doc: doc["binary_relevance"],
                          reverse=True)[:10]

# (e) 提供一个最大化R@5的示例排名
max_r_5_ranking = sorted(ranking, key=lambda doc: doc["rank"])[:5]

# (f) 提供一个最大化R@10的示例排名
max_r_10_ranking = sorted(ranking, key=lambda doc: doc["rank"])[:10]

# (g) 在这种情况下，可以使用R-Precision来设置K值。由于有7个相关文档，所以K = 7。

# (h) 计算平均准确率（AP）
ap = sum((sum(doc["binary_relevance"] for doc in ranking[:i + 1]) / (i + 1))
         for i, doc in enumerate(ranking)
         if doc["binary_relevance"] == 1) / relevant_docs

# (i) 提供一个最大化AP的示例排名
max_ap_ranking = sorted(ranking,
                        key=lambda doc: doc["graded_relevance"],
                        reverse=True)

# (j) 计算DCG5
dcg_5 = ranking[0]["graded_relevance"] + sum(
    doc["graded_relevance"] / math.log2(doc["rank"]) for doc in ranking[1:5])

# (k) 计算NDCG5
ideal_ranking = [{
    "rank": 1,
    "graded_relevance": 4
}, {
    "rank": 2,
    "graded_relevance": 4
}, {
    "rank": 3,
    "graded_relevance": 3
}, {
    "rank": 4,
    "graded_relevance": 2
}, {
    "rank": 5,
    "graded_relevance": 1
}]
idcg_5 = ideal_ranking[0]["graded_relevance"] + sum(
    doc["graded_relevance"] / math.log2(doc["rank"])
    for doc in ideal_ranking[1:5])
ndcg_5 = dcg_5 / idcg_5

# (l) 还有其他评估指标可用于评估排名的性能，如F1得分、AP@K和NDCG@K。

# 打印结果
print("(a) P@5:", p_5)
print("(a) P@10:", p_10)
print("(b) R@5:", r_5)
print("(b) R@10:", r_10)
print("(c) 最大化P@5的示例排名:", [doc["docID"] for doc in max_p_5_ranking])
print("(d) 最大化P@10的示例排名:", [doc["docID"] for doc in max_p_10_ranking])
print("(e) 最大化R@5的示例排名:", [doc["docID"] for doc in max_r_5_ranking])
print("(f) 最大化R@10的示例排名:", [doc["docID"] for doc in max_r_10_ranking])
print("(h) 平均准确率（AP）:", ap)
print("(i) 最大化AP的示例排名:", [doc["docID"] for doc in max_ap_ranking])
print("(j) DCG5:", dcg_5)
print("(k) NDCG5:", ndcg_5)