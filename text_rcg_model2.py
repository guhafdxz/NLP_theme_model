# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy

from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import models
from gensim import corpora

from collections import defaultdict
from collections import Counter
import jieba.posseg as jp
import jieba.analyse
import pyLDAvis
import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()


k1=0.5  #名词 'n' 
k2=0.4  #动词 'v' 
k3=0.3  #形容词/副词 'a' 'd' 
k4=0.2  # 其他词性

delta_k=0.1
λ=1
delta_λ=1
flag_en2cn = {
    'a': '形容词', 'ad': '副形词', 'ag': '形语素', 'an': '名形词', 'b': '区别词',
    'c': '连词', 'd': '副词', 'df': '不要', 'dg': '副语素',
    'e': '叹词', 'f': '方位词', 'g': '语素', 'h': '前接成分',
    'i': '成语', 'j': '简称略语', 'k': '后接成分', 'l': '习用语',
    'm': '数词', 'mg': '数语素', 'mq': '数量词',
    'n': '名词', 'ng': '名语素', 'nr': '人名', 'nrfg': '古代人名', 'nrt': '音译人名',
    'ns': '地名', 'nt': '机构团体', 'nz': '其他专名',
    'o': '拟声词', 'p': '介词', 'q': '量词',
    'r': '代词', 'rg': '代语素', 'rr': '代词', 'rz': '代词',
    's': '处所词', 't': '时间词', 'tg': '时间语素',
    'u': '助词', 'ud': '得', 'ug': '过', 'uj': '的', 'ul': '了', 'uv': '地', 'uz': '着',
    'v': '动词', 'vd': '副动词', 'vg': '动语素', 'vi': '动词', 'vn': '名动词', 'vq': '动词',
    'x': '非语素字', 'y': '语气词', 'z': '状态词', 'zg': '状态语素',
}

n_list=['n', 'ng', 'nr', 'nrfg', 'nrt','ns', 'nt', 'nz']  #广义名词词性
v_list=['v', 'vd', 'vg', 'vi', 'vn', 'vq'] #广义动词词性
a_list=['a','ad','ag','an']  #广义形容词词性
d_list=['d','df','dg']  #广义副词词性


mark_punctuate=['，','。','？','……','”','“'] #去除语料标点
stop_words=[]  #停用词
with open('./中文停用词.txt','r',encoding='utf-8') as f:
    stop_words.append(f.readlines())
stop_words=stop_words[0]


"语料预处理"
def preprocess_str(te):
    text=te
    if ":" in text:
        text=text.split(":")[1]
    if '：' in text:
        text=text.split('：')[1]
    words=[w.word for w in jp.lcut(text) if w.word not in mark_punctuate and '\u4e00' <= w.word <= '\u9fff' and w.word not in stop_words] #Jieba中文分词处理
    word_class=[w.flag for w in jp.lcut(text) if w.word not in mark_punctuate and '\u4e00' <= w.word <= '\u9fff' and w.word not in stop_words] 
    
    return zip(words,word_class)

"统计词频"
def frequency_stats(zip_word):
    frequency=defaultdict(int)
    word_list=list(zip_word)
    for word in word_list:
        frequency[word]+=1
    return frequency

"词性权重"
def weight(word_class,delta_k=0):
    if word_class in n_list:
        return k1+delta_k
    elif word_class in v_list:
        return k2+delta_k
    elif word_class in a_list or word_class in d_list:
        return k3+delta_k
    else:
        return k4+delta_k


def plot_KL(lda,n_topics=20):
    "计算主题方差KL散度"
    divergence=[]
    for k in range(2,n_topics+1):
        var=d=0
        topics_terms=lda.state.get_lambda()
        topics_terms_proba = np.apply_along_axis(lambda x: x/x.sum(),1,topics_terms)
        # 计算主题方差
        for t in range(k):
                for c in range(t, k):
                    var += scipy.stats.entropy(topics_terms_proba[t], topics_terms_proba[c])
                    d += 1
        var_std = round(var / (k*d), 2)
        divergence.append(var_std)
    plt.figure(figsize=(15, 10))
    plt.plot(list(range(2, n_topics+1)), divergence, color='green')
    plt.scatter(list(range(2, n_topics+1)), divergence, color='red')
    plt.xticks(list(range(2, n_topics+1)))
    plt.xlabel('主题数')
    plt.ylabel('主题方差')    
    return divergence

"--------------------------------------------LDA/TFIDF+LDA模型实验-------------------------------------"

"导入语料"
text=pd.read_excel("./参考语料库.xlsx")
corpus=[]  #分词
corpus_class=[] #词性标注
for line in text['评论内容'].tolist():
    if type(line)==str :
       zip_word=preprocess_str(line)
       frequency=frequency_stats(zip_word)
       corpus.append([key[0] for key in frequency if frequency[key] >=1 and len(key[0])>1])
       corpus_class.append([key[1] for key in frequency if frequency[key] >=1 and len(key[0])>1])

corpus=[word_list for word_list in corpus if len(word_list)>1]
corpus_class=[class_list for class_list in corpus_class if len(class_list)>1]
corpus_pair=[list(zip(corpus[i],corpus_class[i])) for i in range(len(corpus))]
"词袋模型"
dictionary=corpora.Dictionary(corpus)
word2id=dictionary.token2id #所有分词和对应索引
id2word={list(word2id.values())[i]:list(word2id.keys())[i] for i in range(len(list(word2id.keys())))} #键值互换
bow_corpus=[dictionary.doc2bow(text) for text in corpus]
"关键词词频和给定语义ωt计算"
corpus_df=pd.DataFrame({'cut_word':np.array(corpus)})
corpus_df['cut_word']=corpus_df['cut_word'].apply(lambda x:" ".join(x))
jieba.analyse.set_stop_words('./中文停用词.txt')
te=''
for i in range(len(corpus_df['cut_word'])):
    if len(corpus_df['cut_word'][i])>0:
     te+=corpus_df['cut_word'][i]+'\n'
c=Counter()
stat=0
for word in te.split():
    c[word]+=1
for word in c:
    if c[word]==1:
        stat+=1
t=int((math.sqrt(1+8*stat)-1)/2)
keyword_list=jieba.analyse.extract_tags(te,topK=100,withWeight=True) #关键词词频
word_t=keyword_list[t-1]  #第t个关键词
word_tid=word2id[word_t[0]]  #第t个关键词索引
"语料训练tf-idf模型TFDIF"
tfidf=models.TfidfModel(bow_corpus)  
corpus_tfidf=[tfidf[doc] for doc in bow_corpus]
"corpus_tfidf词性加权处理"

def get_weighted_word_tfidf(corpus_tfidf,delta_k):
    word_t_tfidf=[]  #统计第t个关键词tfidf
    for i in range(len(corpus_tfidf)):
        for m in range(len(corpus_tfidf[i])):
            id_tfidf=corpus_tfidf[i][m]
            word=id2word[id_tfidf[0]]
            word_class=corpus_pair[i][corpus[i].index(word)][1]
            corpus_tfidf[i][m]=(id_tfidf[0],id_tfidf[1]*weight(word_class,delta_k))     
            if id_tfidf[0]==word_tid:
                word_t_tfidf.append(id_tfidf[1]*weight(word_class,delta_k))  #找到第t个关键词对应的加权tfidf值
    word_t_tfidf=np.sum(word_t_tfidf)     #给定第t个词加权tfidf值
    return word_t_tfidf
word_t_tfidf=get_weighted_word_tfidf(corpus_tfidf,delta_k)
            
"LDA建模"
np.random.seed(1234)
lda=LdaModel(corpus=bow_corpus,id2word=dictionary,num_topics=20,passes=100)
ldacm=CoherenceModel(model=lda,texts=corpus,dictionary=dictionary,coherence='u_mass')
lda_tfidf=LdaModel(corpus=corpus_tfidf,id2word=dictionary,num_topics=20,passes=100)
ldacm_tfidf=CoherenceModel(model=lda_tfidf,texts=corpus,dictionary=dictionary,coherence='u_mass')
"主题连贯性计算"
coherence_lda=ldacm.get_coherence()
coherence_lda_tfidf=ldacm_tfidf.get_coherence()
doc_topic_lda = [a for a in lda[bow_corpus]]
doc_topic_lda_tfidf = [a for a in lda_tfidf[bow_corpus]]
topics_r_lda = lda.print_topics(num_topics = 20, num_words =5)
topics_r_lda_tfidf = lda_tfidf.print_topics(num_topics = 20, num_words =5)
"LDA模型可视化"
data=pyLDAvis.gensim_models.prepare(lda,bow_corpus,dictionary)
pyLDAvis.save_html(data,'online_lda.html')
data=pyLDAvis.gensim_models.prepare(lda_tfidf,bow_corpus,dictionary)
pyLDAvis.save_html(data,'online_lda_tfidf.html')
"计算主题方差KL散度"

divergence_lda=plot_KL(lda,20)
divergence_lda_tfidf=plot_KL(lda_tfidf,20)

"--------------------------------------------BTM主题模型实验----------------------------------"
"Gibbs采样——BTM"
def gibbs_BTM(It, V, B,num_topics, b, alpha=1., beta=0.1):
  print("Biterm model ------ ")
  print("Corpus length: " + str(len(b)))
  print("Number of topics: " + str(num_topics))
  print("alpha: " + str(alpha) + " beta: " + str(beta))

  Z =  np.zeros(B)
  Nwz = np.zeros((V, num_topics))
  Nz = np.zeros(num_topics)

  theta = np.random.dirichlet([alpha]*num_topics, 1)
  #gibbs采样初始化
  for ibi, bi in enumerate(b):
     
      topics = np.random.choice(num_topics, 1, p=theta[0,:])[0]
      Nwz[bi[0], topics] += 1
      Nwz[bi[1], topics] += 1
      Nz[topics] += 1
      Z[ibi] = topics
  Z=Z.astype('int32')
  #gibbs采样
  for it in range(It):
      print("Iteration: " + str(it))
      Nzold = np.copy(Nz)
      for ibi, bi in enumerate(b):
          Nwz[bi[0], Z[ibi]] -= 1
          Nwz[bi[1], Z[ibi]] -= 1
          Nz[Z[ibi]] -= 1
          pz = (Nz + alpha)*(Nwz[bi[0],:]+beta)*(Nwz[bi[1],:]+beta)/(Nwz.sum(axis=0)+beta*V)**2
          pz = pz/pz.sum()
          Z[ibi] = np.random.choice(num_topics, 1, p=pz)
          Nwz[bi[0], Z[ibi]] += 1
          Nwz[bi[1], Z[ibi]] += 1
          Nz[Z[ibi]] += 1
      print("Variation between iterations:  " + str(np.sqrt(np.sum((Nz-Nzold)**2))))
  return Nz, Nwz, Z


"Gibbs采样——TFIDF-BTM"
def gibbs_TFIDF_BTM(It, V, B,dictionary,word_tfidf,word_t_tfidf,num_topics, b, alpha=1., beta=0.1,λ=1):
    print("Biterm model ------ ")
    print("Corpus length: " + str(len(b)))
    print("Number of topics: " + str(num_topics))
    print("alpha: " + str(alpha) + " beta: " + str(beta))

    Z =  np.zeros(B)
    Nwz = np.zeros((V, num_topics))
    Nz = np.zeros(num_topics)
    q=0 #统计增加词对的数量
    theta = np.random.dirichlet([alpha]*num_topics, 1)
    #gibbs采样初始化
    for ibi, bi in enumerate(b): 
        word=dictionary[bi[0]]
        topics = np.random.choice(num_topics, 1, p=theta[0,:])[0]
        if word_tfidf[word]>word_t_tfidf: #ωa＞ωt
          n_wz[bi[0],topics]+=1+word_tfidf[word]*λ
          n_wz[bi[1],topics]+=1+word_tfidf[word]*λ
          n_z[topics] += 1+word_tfidf[word]*λ
          q+=1  #统计增加词对的数量
        else:
            Nwz[bi[0], topics] += 1
            Nwz[bi[1], topics] += 1
            Nz[topics] += 1
        Z[ibi] = topics
    Z=Z.astype('int32')
    #gibbs采样
    for it in range(It):
        print("Iteration: " + str(it))
        Nzold = np.copy(Nz)
        for ibi, bi in enumerate(b):
            Nwz[bi[0], Z[ibi]] -= 1
            Nwz[bi[1], Z[ibi]] -= 1
            Nz[Z[ibi]] -= 1
            word=dictionary[bi[0]]
            if word_tfidf[word]>word_t_tfidf: #ωa＞ωt 依据ωa与给定的语义距离值ωt之间的距离关系
              pz = (Nz + alpha)*(Nwz[bi[0],:]+beta+word_tfidf[word]*λ)*(Nwz[bi[1],:]+beta+word_tfidf[word]*λ)/(Nwz.sum(axis=0)+beta*V+2*word_tfidf[word]*λ)**2
              pz = pz/pz.sum()
              Z[ibi] = np.random.choice(num_topics, 1, p=pz)
              Nwz[bi[0], Z[ibi]] += 1
              Nwz[bi[1], Z[ibi]] += 1
              Nz[Z[ibi]] += 1
            else:
                pz = (Nz + alpha)*(Nwz[bi[0],:]+beta)*(Nwz[bi[1],:]+beta)/(Nwz.sum(axis=0)+beta*V)**2
                pz = pz/pz.sum()
                Z[ibi] = np.random.choice(num_topics, 1, p=pz)
                Nwz[bi[0], Z[ibi]] += 1
                Nwz[bi[1], Z[ibi]] += 1
                Nz[Z[ibi]] += 1
        print("Variation between iterations:  " + str(np.sqrt(np.sum((Nz-Nzold)**2))))
    return Nz, Nwz, Z,q

def pbd(doc,names):
    ret = []
    retnames = []
    for term1 in set(doc):
        cnts = 0.
        for term2 in doc:
            if term1 == term2:
                cnts +=1.
        ret.append(cnts/len(doc))
        retnames.append(term1)
    if names:
        return retnames
    else:
        return ret
#计算单词词性加权tfidf
def  weighted_tfidf(texts,dictionary,corpus_pair,delta_k):
    count={}
    sentence={}
    #统计词频
    for i in range(len(dictionary)):
        word=dictionary[i]
        count[word]=0
        sentence[word]=0
        for text in texts:
          if word in text:
              sentence[word]+=1
          for w in text:
                if word==w:
                    count[word]+=1
   
    freq_word={word:count[word]/sum(count.values()) for word in count}
    idf_word={word:math.log(len(texts)/ (1+sentence[word])) for word in sentence}
    tfidf_word={word: freq_word[word] * idf_word[word] for word in freq_word }
    corpus_pair_list=[]
    for pair in corpus_pair:
        corpus_pair_list.extend(pair)
    word_type_dict=dict(corpus_pair_list)
    
    weighted_word_tfidf={word:tfidf_word[word]* weight(word_type_dict[word],delta_k) for word in tfidf_word}
    
    return weighted_word_tfidf

def topTopic(Nwz,model,q,word_tfidf,λ=1,num_topics=20):
      topics =  [[dictionary[ident] for ident in np.argsort(-Nwz[:,k])[0:10]] for k in range(num_topics)]
      print("TOP 10 words per topic")
      topic_tfidf=[]
      for topic in topics:
          tfidf=0
          print(topic)
          print ("  ---- ")
          for word in topic:
              tfidf+=word_tfidf[word]*λ  #K个主题中单词wa的tfidf值与λ的乘积累加
          topic_tfidf.append(tfidf)
      topic_tfidf=np.array(topic_tfidf) 
      if  model=='btm':
        thetaz = (Nz + alpha)/(B + num_topics*alpha)
        phiwz = (Nwz + beta)/np.tile((Nwz.sum(axis=0)+V*beta),(V,1))
      else:  
          thetaz = (Nz + alpha+topic_tfidf)/(B + num_topics*alpha+q)
          phiwz = (Nwz + beta+topic_tfidf)/np.tile((Nwz.sum(axis=0)+V*beta+2*topic_tfidf),(V,1))

pzb = [[list(thetaz*phiwz[term[0],:]*phiwz[term[1],:]/(thetaz*phiwz[term[0],:]*phiwz[term[1],:]).sum()) for term in set(doc)] for doc in btmp]

pdz = []
for idoc, doc in enumerate(pzb):
    aux = 0
    for iterm, term in enumerate(doc):
        aux += np.array(term) * pbd_cts[idoc][iterm]
    pdz.append(aux)

pdz = np.array(pdz)

topics = [[tweets[ident] for ident in np.argsort(-pdz[:,k])[0:5]] for k in xrange(num_topics)]
print("TOP 5 tweets per topic")
for topic in topics:
    for tweet in topic:
        print tweet
    print (" ---- ")



texts=corpus_df.cut_word.tolist() #导入文本生成词组列表
N = len(texts)
texts=[text.split() for text in texts]
N = len(texts)
dictionary = np.array(list(set([word for text in texts for word in text]))) 
word_tfidf= weighted_tfidf(texts,dictionary,corpus_pair,delta_k) #计算词组词性加权tfidf

V = len(dictionary)
alpha = 1.
beta = 0.1

btmp = [[(np.where(dictionary==word1)[0][0], np.where(dictionary==word2)[0][0])
         for iword1, word1 in enumerate(text) for iword2, word2 in enumerate(text)
         if iword1 < iword2] for text in texts] #生成biterms
aux = []
for bi in btmp:
    aux.extend(bi)
b = aux  #biterm数组
B = len(b)
bset = set(b)
num_topics = 20
pbd_cts = [pbd(doc, False) for doc in btmp]
pbd_names = [pbd(doc, True) for doc in btmp]




Nz, Nwz, Z = gibbs_BTM(It=20, V=V, B=B, num_topics=num_topics, b=b, alpha=alpha, beta=beta) #btm
Nz_td,Nwz_td,Z_td=gibbs_TFIDF_BTM(20, V, B,dictionary, word_tfidf,word_t_tfidf,num_topics, b, alpha=1., beta=0.1,λ=1)  #tfidf+btm
















