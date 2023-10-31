def compute_AP(rankings, relevant_docs, k):
    num_relevant = 0
    precision_sum = 0.0

    for i in range(k):
        doc = rankings[i]
        if doc in relevant_docs:
            num_relevant += 1
            precision = num_relevant / (i + 1)
            precision_sum += precision

    avg_precision = precision_sum / min(k, len(relevant_docs))
    return avg_precision


def compute_RR(rankings, relevant_docs, k):
    for i in range(k):
        doc = rankings[i]
        if doc in relevant_docs:
            return 1.0 / (i + 1)

    return 0.0


# 查询1的结果
ranking_1 = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10']
relevant_docs_1 = ['d1', 'd3', 'd4', 'd6', 'd7', 'd10']

# 查询2的结果
ranking_2 = ['d3', 'd8', 'd7', 'd1', 'd2', 'd4', 'd5', 'd9', 'd10', 'd6']
relevant_docs_2 = ['d8', 'd9']

# 查询3的结果
ranking_3 = ['d7', 'd6', 'd5', 'd3', 'd2', 'd1', 'd9', 'd10', 'd4', 'd8']
relevant_docs_3 = ['d5', 'd9', 'd8']

# 计算AP@5和AP@10
ap_5_query_1 = compute_AP(ranking_1, relevant_docs_1, 5)
ap_10_query_1 = compute_AP(ranking_1, relevant_docs_1, 10)

ap_5_query_2 = compute_AP(ranking_2, relevant_docs_2, 5)
ap_10_query_2 = compute_AP(ranking_2, relevant_docs_2, 10)

ap_5_query_3 = compute_AP(ranking_3, relevant_docs_3, 5)
ap_10_query_3 = compute_AP(ranking_3, relevant_docs_3, 10)

# 计算RR@5和RR@10
rr_5_query_1 = compute_RR(ranking_1, relevant_docs_1, 5)
rr_10_query_1 = compute_RR(ranking_1, relevant_docs_1, 10)

rr_5_query_2 = compute_RR(ranking_2, relevant_docs_2, 5)
rr_10_query_2 = compute_RR(ranking_2, relevant_docs_2, 10)

rr_5_query_3 = compute_RR(ranking_3, relevant_docs_3, 5)
rr_10_query_3 = compute_RR(ranking_3, relevant_docs_3, 10)

# 计算MAP@5和MAP@10
map_5 = (ap_5_query_1 + ap_5_query_2 + ap_5_query_3) / 3
map_10 = (ap_10_query_1 + ap_10_query_2 + ap_10_query_3) / 3

# 计算MRR@5和MRR@10
mrr_5 = (rr_5_query_1 + rr_5_query_2 + rr_5_query_3) / 3
mrr_10 = (rr_10_query_1 + rr_10_query_2 + rr_10_query_3) / 3

# 打印结果
print("Query 1 AP@5 :", ap_5_query_1)
print("Query 1 AP@10:", ap_10_query_1)
print("Query 1 RR@5 :", rr_5_query_1)
print("Query 1 RR@10:", rr_10_query_1)
print("Query 2 AP@5 :", ap_5_query_2)
print("Query 2 AP@10:", ap_10_query_2)
print("Query 2 RR@5 :", rr_5_query_2)
print("Query 2 RR@10:", rr_10_query_2)
print("Query 3 AP@5 :", ap_5_query_3)
print("Query 3 AP@10:", ap_10_query_3)
print("Query 3 RR@5 :", rr_5_query_3)
print("Query 3 RR@10:", rr_10_query_3)

print("MAP@5 :", map_5)
print("MAP@10:", map_10)
print("MRR@5 :", mrr_5)
print("MRR@10:", mrr_10)