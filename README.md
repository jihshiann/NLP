# NLP學習日誌
## W2基礎文本處理與分類
1. 詞幹提取（Stemming）和詞形還原（Lemmatization）: nltk 庫中的 PorterStemmer 和 WordNetLemmatizer <br>
1. scikit-learn 的 CountVectorizer 將文本轉換為數值向量，並利用這些向量訓練一個樸素貝葉斯分類器。

## W4文本表示與推薦系統
1. TF-IDF (Term Frequency-Inverse Document Frequency)進行向量化，並透過餘弦相似度（cosine similarity）來尋找相似的電影進行推薦
1. 詞嵌入（Word Embedding): 預訓練的 Word2Vec 模型

## W6&8序列模型與密碼學應用
1. 二階馬可夫模型（second-order Markov model）來生成類似詩歌的文本
1. 一階馬可夫模型計算詞語的初始和轉移機率，然後用樸素貝葉斯方法進行分類和評估
1. 馬可夫模型和基因演算法來破解替換式密碼。它將解碼後的文本的對數機率作為適應度分數，透過演算法迭代地尋找最優解碼方案

## W9非監督式學習與文本摘要
1. TF-IDF 和多項式樸素貝葉斯來實作情感分析
1. TF-IDF 和多項式樸素貝葉斯來進行垃圾郵件檢測
1. 計算句子中詞語的 TF-IDF 權重，來選出最重要的句子作為文章的摘要

## W11主題模型
1. 潛在語義分析 (LSA)，透過奇異值分解從文本中提取主題
1. 潛在狄利克雷分配 (LDA)，一個機率模型，來發現文本中隱藏的主題
1. 非負矩陣分解 (NMF)，一種適合詞頻數據的演算法，來分解出主題

## W12神經網路基礎
1. 淺層神經網路和 TF-IDF 進行文本分類
1. 序列資料上的應用: 卷積神經網路（CNN）和預訓練的 GloVe 詞嵌入來進行文本分類
1. CBOW (Continuous Bag of Words) 詞嵌入模型

## W13序列標註與 RNN 
1. 隱馬可夫模型 (HMM) 實現詞性標註
1. 使用 spaCy 函式庫進行命名實體識別 (NER)
1. 循環神經網路 (RNN) 來對文本進行分類

## W14-16大型語言模型 (LLMs) 
1. Hugging Face 的 BERT 模型對推文進行情感分析
1. BERT 進行更細緻的情緒分類任務
1. Hugging Face 的 GPT 模型進行文本生成，並示範如何控制生成過程
1. SentenceTransformer 模型建立向量資料庫，並利用 FAISS 實現語義搜尋，應用於電影推薦和聖經經文檢索
1. 向量資料庫概念擴展到多模態領域，使用 CLIP 模型將文字和圖片嵌入到同一向量空間，實現跨模態搜尋
1. 利用 Llama-2 等 LLM 進行資料增強，以擴充訓練資料集
1. LLM 微調(finetuning)，分別使用 OpenAI API 和 Gradient 平台來客製化 GPT-3.5-Turbo 和 Llama-2-7b-chat 等模型
1. LLM 的量化技術，減少模型的大小和記憶體需求
