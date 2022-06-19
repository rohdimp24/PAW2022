import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import PorterStemmer
import gensim, logging
import re
import pickle
import scipy as sp
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import mysql.connector
from sklearn.manifold import TSNE
from nltk import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud


def getConnectionString(dbName):
    conn=mysql.connector.connect(user='rohit', password='xjkrrbrWpAsKKAGf',host='eagleeye-sd.ckjulgq8jktj.ap-south-1.rds.amazonaws.com',port='3306',database=dbName)
    return conn


def getStopWords():
    # lst=['box','carton','kgx','line','line_carton','main_line','pouch','red','whole','black','blue','bottle','btl','can','carton','elect','elect_pr','elect_pr_label','free','fresh','gift','green','hp','jar','label','label_tze','ml','ml_mrp','mrp','na','np','offer','others','pack','premium','pt','pt_cc','shrink','tb','tb_hs','tze','white','advertising_materials','diff','na','fl','fr','free','fw','gr','set','tl','adult','baby','bc','body','care','cool','de','fr','free','frs_free_item','ir','pack','premium','reg','regular','set','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','gms','xg','pcs','q','r','s','t','u','v','w','x','y','z','ml','rs','kg','gm','pc','mrp','new','old','more','ltr','pouch','carton','bizom','shrink','line','gms','gm','jar','xg','kgx','pcs','red','pp','bottle','gold','super','mono','golden','premium','free','white','nw','pack','np','black','sachet','mr','ctn','ig','xkg','nf','fresh','lit','box','green','bag','rb','offer','lb','lt','mini','km','btl','promo','gxn','classic','pk','pkt','cbb','mlx','big','delite','special','solar','lp','ch','tb','ic','small','xgm','regular','xx','hs','gxx','sp','liner','supreme','nirmal','zone','kewal','kgs','opening','closing','shelves','godown','shelf','fsu','floor','stack','gx','townbus','mm','rbm','cp','empty','nd','value','hdpe','ltrx','gmx','pj','bulk','fun','exchange','stb','bl','delight','extra','platinum','select','bopppp','three','gr','csd','pure','full','vp','xml','nat','str','combi','lm','rgb','pt','ltrs','ab','agmark','poly','‚äì','mansion','house','excellence','s¬†','flavor','scheme','shubh','stick','rich','case','asap','one','label','xxgm','indian','high','original','bot','litre','cut','xxg','bottles','canz','test','packs','jarxx','diff','amp','lbs']
    # return lst
    #standard english
    englishStopWords = set(['a','in','about','ain','all','am','an','and','any','are','aren',"aren't",'as','at','be','because','been','being','both','by','can','d','did','during','each','few','for','from','further','he','her','here','hers','herself','him','himself','his','how','i','if','is','isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",'my','myself','needn','o','of','once','only','or','other','our','ours','ourselves','own','re','s','same','shan',"shan't",'she',"she's",'should',"should've",'shouldn',"shouldn't",'so','some','such','t','than','that',"that'll",'the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too','until','ve','very','was','wasn',"wasn't",'we','were','weren',"weren't",'what','when','where','which','while','who','whom','why','will','with','won',"won't",'wouldn',"wouldn't",'y','you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves'])
    englishStopWords.remove('can')
    #domainspecific..later on should come from db
    domain_words=set(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','gms','xg','pcs','q','r','s','t','u','v','w','x','y','z','ml','rs','kg','gm','pc','mrp','new','old','more','ltr'])
    #domain_words_extra=set(['pouch','carton','bizom','shrink','line','gms','gm','jar','pwd','pet','xg','kgx','pcs','bopp','tin','red','cc','pp','bottle','gold','super','mono','golden','premium','free','white','nw','pack','np','black','sachet','mr','ctn','tetra','ig','xkg','lit','box','ice','green','bag','rb','offer','lb','lt','mini','km','btl','promo','gxn','classic','pk','pkt','cbb','mlx','big','delite','special','solar','lp','ch','tb','ic','small','xgm','regular','xx','hs','gxx','sp','liner','supreme','nirmal','zone','kewal','kgs','opening','closing','shelves','godown','shelf','fsu','floor','stack','gx','townbus','mm','rbm','cp','empty','nd','value','hdpe','ltrx','gmx','plastic','pj','bulk','fun','exchange','stb','bl','delight','extra','platinum','select','bopppp','three','gr','csd','pure','full','vp','xml','nat','str','combi','lm','rgb','pt','ltrs','ab','agmark','poly','krishna','‚äì','mansion','house','tub','excellence','olein','s¬†','flavor','scheme','shubh','stick','rich','case','asap','one','label','xxgm','indian','high','original','bot','litre','cut','xxg','bottles','canz','test','packs','jarxx','diff','amp','lbs'])
    #domain_words=domain_words.union(domain_words_extra)
    stop_words=englishStopWords.union(domain_words)
    return list(stop_words)


#cleaup of the digits
def removeDigits(rawText):
    #https://stackoverflow.com/questions/12851791/removing-numbers-from-string/12856384
    from string import digits
    remove_digits = str.maketrans('', '', digits)
    res = rawText.translate(remove_digits)
    return res

#remove the punctuations
def removePunctuations(rawText):
    case=rawText
    case = case.strip();
    case = re.sub('/[^A-Za-z0-9 _\-\+\&\,\#]/', '', case)
    case = case.replace('"', ' ')
    case = case.replace('\"', ' ')
    case = case.replace('>', ' ')
    case = case.replace('@', ' ')
    case = case.replace('<', ' ')
    case = case.replace(':', ' ')
    case = case.replace('.', ' ')
    case = case.replace('(', ' ')
    case = case.replace(')', ' ')
    case = case.replace('[', ' ')
    case = case.replace(']', ' ')
    case = case.replace('_', ' ')
    case = case.replace(',', ' ')
    case = case.replace('#', ' ')
    case = case.replace('-', ' ')
    case = case.replace('/', ' ')
    case = case.replace('"', ' ')
    case = case.replace('\n', ' ')
    case = case.replace('\r', ' ')
    case = case.replace('~', ' ')
    case = case.replace('%', ' ')
    case = case.replace('$', ' ')
    case = case.replace('&', ' ')
    case = case.replace('!', ' ')
    case = case.replace('*', ' ')
    case = case.replace('+', ' ')
    case = case.replace('?', ' ')
    case = case.replace(';', ' ')
    case = case.replace('_', ' ')
    case = case.replace('\'', ' ')
    case = case.replace('{', ' ')
    case = case.replace('}', ' ')
    case = case.replace('=', ' ')
    case = case.replace(':', ' ')
    case = case.replace('`', ' ')
    
                        
    
    return case



def preprocessText(rawText):
    #remove the digits
    removedDigitsText=removeDigits(rawText)
    #remove the punctuations
    removePunctuationText=removePunctuations(removedDigitsText)
    cleanedUpText=removePunctuationText
    return cleanedUpText

# remove the stopwords from the list of the tokens . It expects the skuname string as a list of tokens
def removeStopWords(text_tokens,lstStopWords):
    tokens_without_sw = [word for word in text_tokens if not word in lstStopWords]
    return tokens_without_sw



def cleanupSKUName(skuName):
    removedDigitsText=removeDigits(skuName)
    #remove the punctuations
    removePunctuationText=removePunctuations(removedDigitsText)
    return removePunctuationText

def getNgramString(inputString):
    inputString=inputString.lower()
    lstStopwords=getStopWords()
    removePunctuationText=cleanupSKUName(inputString)
    tokenizedString=nltk.word_tokenize(removePunctuationText.lower())
    tokens_without_sw=removeStopWords(tokenizedString,lstStopwords)
    res=list(set(tokens_without_sw))
    #print(res)
    #print(nltk.word_tokenize(initalString))
    finalWordToken=[word for word in nltk.word_tokenize(removePunctuationText.lower()) if  word in res]
    domainStr=' '.join(finalWordToken)
    domainTokens = nltk.word_tokenize(domainStr)
    domain_unigrams=ngrams(domainTokens,1)
    domain_bigrams = ngrams(domainTokens,2)
    domain_trigrams = ngrams(domainTokens,3)
    unigramStringLst=[]
    bigramStringLst=[]
    trigramStringLst=[]

    for unigram in domain_unigrams:
        unigramStringLst.append('_'.join(unigram))

    for bigram in domain_bigrams:
        bigramStringLst.append('_'.join(bigram))

    for trigram in domain_trigrams:
        trigramStringLst.append('_'.join(trigram))


    unigramString=','.join(unigramStringLst)
    bigramString=','.join(bigramStringLst)
    trigramString=','.join(trigramStringLst)

    #print(inputString+"##"+unigramString+"=>"+bigramString+"=>"+trigramString)
    return str(unigramString)+","+str(bigramString)+","+str(trigramString)




## this is to generate the wordcloud. 
### look at the example in http://localhost:8888/notebooks/FindUnigramBigramRelationships.ipynb
def createWordCloud(combinedDict):
    wordcloud = WordCloud(background_color="white", max_words=2000,max_font_size=40, random_state=42)
    wordcloud.generate_from_frequencies(frequencies=combinedDict)
    plt.figure(figsize=[10, 5])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()




def generalDBquery(sqlQuery):
    conn = getConnectionString()
    cursor = conn.cursor()
    sqlSelect=sqlQuery
    df=pd.read_sql_query(sqlSelect,conn)
    return df



###LAmbda fucntions
which = lambda lst:list(np.where(lst)[0])