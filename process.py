import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import gensim
import numpy as np
import collections
from gensim import corpora
from gensim.models import LdaModel
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')  # 可以根据需要选择其他字体

# 读取CSV文件
df = pd.read_csv('巴以冲突.csv')

# 查看数据前几行
print(df.head())

# 文本清洗函数
def clean_text(text):
    # 移除所有的非中文字符（保持中文字符）
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 去除多余的空白字符和换行符
    text = text.strip()
    return text

# 定义分词函数
def jieba_cut(text):
    return " ".join(jieba.cut(text))

# 停用词加载函数
def load_stopwords(file_path):
    stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

# 定义去停用词的函数
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stopwords])

# 应用到整个文本列
df['cleaned_content'] = df['微博正文'].apply(lambda x: clean_text(str(x)))

# 查看清洗后的文本
print(df['cleaned_content'].head())

# 应用分词
df['segmented_content'] = df['cleaned_content'].apply(lambda x: jieba_cut(x))

# 查看分词结果
print(df['segmented_content'].head())

# 假设停用词列表文件为 'stopwords.txt'
stopwords = load_stopwords('hit_stopwords.txt')

# 应用去停用词
df['filtered_content'] = df['segmented_content'].apply(lambda x: remove_stopwords(x))

# 查看去停用词后的文本
print(df['filtered_content'].head())

# 初始化 TF-IDF 向量化器
tfidf_vectorizer = TfidfVectorizer()

# 对清洗后的文本进行 TF-IDF 转换
tfidf_matrix = tfidf_vectorizer.fit_transform(df['filtered_content'])

# 查看生成的TF-IDF矩阵（稀疏矩阵格式）
print(tfidf_matrix.shape)

# 也可以存储 TF-IDF 矩阵为一个文件（可选择稀疏矩阵或密集矩阵）
sp.save_npz('tfidf_matrix.npz', tfidf_matrix)

# 先从 TF-IDF 矩阵中获取词典
tfidf_matrix_dense = tfidf_matrix.toarray()  # 将稀疏矩阵转为密集矩阵
vocab = list(tfidf_vectorizer.get_feature_names_out())  # 获取词汇表

# 创建 Gensim 词典
dictionary = corpora.Dictionary([vocab])

# 将 TF-IDF 转换为 Gensim 格式的文档词袋表示
corpus = []
for doc in tfidf_matrix_dense:
    doc_str = [vocab[i] for i in range(len(doc)) if doc[i] > 0]
    corpus.append(dictionary.doc2bow(doc_str))

# 检查是否存在已有的模型lda_model.gensim文件,有的话直接加载到lda_model,否则再新建一个
if os.path.exists('lda_model.gensim'):
    lda_model = LdaModel.load('lda_model.gensim')
else:
    # 创建 LDA 模型（OLDA 是 LDA 的一种扩展，Gensim 也支持增量学习）
    lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=10)
    # 保存训练好的 LDA 模型
    lda_model.save('lda_model.gensim')

# 输出每个主题的前几个关键词
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

# 获取每篇文档的主题分布
topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]

# 将稀疏表示转换为密集向量
dense_topic_distributions = np.array([[prob for _, prob in doc] for doc in topic_distributions])

print(dense_topic_distributions.shape)  # 输出 (文档数量, 主题数量)

# 定义 K-means 模型
num_clusters = 3  # 聚类数目
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# 拟合 K-means 模型
kmeans.fit(dense_topic_distributions)

# 获取每个文档的聚类标签
labels = kmeans.labels_

# 输出每个文档的聚类标签
print(labels)

# 获取聚类中心
cluster_centers = kmeans.cluster_centers_

# 打印每个簇的中心
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}: {center}")

# 统计每个簇中的文档数
cluster_counts = collections.Counter(labels)
print("Cluster Counts:", cluster_counts)

# 将聚类标签加入到原始数据框中
df['cluster_label'] = labels

# 查看每个聚类中的示例文档
for cluster in range(num_clusters):
    print(f"\nCluster {cluster} Examples:")
    print(df[df['cluster_label'] == cluster]['微博正文'].head())

# 将结果存储为新的 CSV 文件
df.to_csv('processed_data.csv', index=False)

# PCA 降维到 2D
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(dense_topic_distributions)

# 绘制聚类结果
plt.figure(figsize=(10, 6))
for cluster in range(num_clusters):
    cluster_points = reduced_data[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")

plt.legend()
plt.title("K-means Clustering of Topics")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig('kmeans_clusters.png')
plt.close()

# 打印每个聚类的热点话题关键词
for i, center in enumerate(cluster_centers):
    topic_indices = np.argsort(-center)[:5]  # 获取前 5 个最重要的主题
    print(f"\nCluster {i} Hot Topics:")
    for topic_idx in topic_indices:
        print(f"  Topic {topic_idx}: {lda_model.print_topic(topic_idx, topn=5)}")
    # 汇总每个簇的关键词
    cluster_keywords = [lda_model.print_topic(topic_idx, topn=5) for topic_idx in topic_indices]
    # 处理汇总后的关键词
    cluster_keywords = [re.findall(r'"(.*?)"', keyword) for keyword in cluster_keywords]
    # 展平关键词列表
    flat_keywords = [item for sublist in cluster_keywords for item in sublist]
    # 生成词云
    wordcloud = WordCloud(font_path='msyh.ttc', background_color='white').generate(' '.join(flat_keywords))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"话题{i}的热门标题",fontproperties=font,fontsize=20)
    plt.savefig(f'cluster_{i}_wordcloud.png')
    plt.close()
