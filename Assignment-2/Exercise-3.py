import matplotlib.pyplot as plt
import os

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

# 计算Precision和Recall
precision = []
recall = []
relevant_docs = sum(doc["binary_relevance"] for doc in ranking)
retrieved_docs = 0
relevant_retrieved_docs = 0

for doc in ranking:
    retrieved_docs += 1
    if doc["binary_relevance"] == 1:
        relevant_retrieved_docs += 1
    precision.append(relevant_retrieved_docs / retrieved_docs)
    recall.append(relevant_retrieved_docs / relevant_docs)

# 绘制PR曲线
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(os.path.dirname(__file__) + '/Exercise-3.png', dpi=300)
plt.show()