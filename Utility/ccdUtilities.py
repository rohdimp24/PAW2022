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
import contractions
from bs4 import BeautifulSoup
import numpy as np
import re
import tqdm
import unicodedata
from collections import Counter
import pandas as pd
import itertools


def getConnectionString():
    conn=mysql.connector.connect(user='rohit', password='xjkrrbrWpAsKKAGf',host='eagleeye-sd.ckjulgq8jktj.ap-south-1.rds.amazonaws.com',port='3306',database='athenaLogs')
    return conn


# In[ ]:


def generalDBquery(sqlQuery):
    conn = getConnectionString()
    cursor = conn.cursor()
    sqlSelect=sqlQuery
    df=pd.read_sql_query(sqlSelect,conn)
    return df




def generateWordFrequencies(domain_unigrams,domain_bigrams,domain_trigrams,domain_quadgrams,outfile):

    with open(outfile, encoding='utf-8-sig', mode='w') as fp:
        threshold=0
        delimitPattern='^^'
        finalList={}
        cntObj=Counter(domain_unigrams)
        sortedCounterList=cntObj.most_common()
        for tag, count in sortedCounterList:
            unigramString=tag[0]
            #print(unigramString,count)
            if(unigramString.find(delimitPattern) == -1):
                if(count>threshold):
        #             unigramKey=''.unigramString[0]
                    finalList[unigramString]=count
                    fp.write('{}|{}|{}\n'.format("unigram",unigramString, count))



        cntObj=Counter(domain_bigrams)
        sortedCounterList=cntObj.most_common()
        for tag, count in sortedCounterList:
            bigramString=tag[0]+"_"+tag[1]
            #print(bigramString,count)
            if(bigramString.find(delimitPattern) == -1):
                if(count>threshold):
                    finalList[bigramString]=count
                    fp.write('{}|{}|{}\n'.format("bigram",bigramString, count))


        cntObj=Counter(domain_trigrams)
        sortedCounterList=cntObj.most_common()
        for tag, count in sortedCounterList:
            trigramString=tag[0]+"_"+tag[1]+"_"+tag[2]
            #print(trigramString,count)
            if(trigramString.find(delimitPattern) == -1):
                if(count>threshold):
                    finalList[trigramString]=count
                    fp.write('{}|{}|{}\n'.format("trigram",trigramString, count))


        cntObj=Counter(domain_quadgrams)
        sortedCounterList=cntObj.most_common()
        for tag, count in sortedCounterList:
            quadString=tag[0]+"_"+tag[1]+"_"+tag[2]+"_"+tag[3]
            #print(trigramString,count)
            if(quadString.find(delimitPattern) == -1):
                if(count>threshold):
                    finalList[quadString]=count
                    fp.write('{}|{}|{}\n'.format("quad",quadString, count))

    return finalList


























# In[ ]:


def getStopWords():
    #standard english
    englishStopWords = set(['a','in','about','ain','all','am','an','and','any','are','aren',"aren't",'as','at','be','because','been','being','both','by','can','d','did','during','each','few','for','from','further','he','her','here','hers','herself','him','himself','his','how','i','if','is','isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",'my','myself','needn','o','of','once','only','or','other','our','ours','ourselves','own','re','s','same','shan',"shan't",'she',"she's",'should',"should've",'shouldn',"shouldn't",'so','some','such','t','than','that',"that'll",'the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too','until','ve','very','was','wasn',"wasn't",'we','were','weren',"weren't",'what','when','where','which','while','who','whom','why','will','with','won',"won't",'wouldn',"wouldn't",'y','you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves'])
    englishStopWords.remove('can')
    #domainspecific..later on should come from db
    domain_words=set(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','gms','xg','pcs','q','r','s','t','u','v','w','x','y','z','ml','rs','kg','gm','pc','mrp','new','old','more','ltr'])
    #domain_words_extra=set(['pouch','carton','bizom','shrink','line','gms','gm','jar','pwd','pet','xg','kgx','pcs','bopp','tin','red','cc','pp','bottle','gold','super','mono','golden','premium','free','white','nw','pack','np','black','sachet','mr','ctn','tetra','ig','xkg','lit','box','ice','green','bag','rb','offer','lb','lt','mini','km','btl','promo','gxn','classic','pk','pkt','cbb','mlx','big','delite','special','solar','lp','ch','tb','ic','small','xgm','regular','xx','hs','gxx','sp','liner','supreme','nirmal','zone','kewal','kgs','opening','closing','shelves','godown','shelf','fsu','floor','stack','gx','townbus','mm','rbm','cp','empty','nd','value','hdpe','ltrx','gmx','plastic','pj','bulk','fun','exchange','stb','bl','delight','extra','platinum','select','bopppp','three','gr','csd','pure','full','vp','xml','nat','str','combi','lm','rgb','pt','ltrs','ab','agmark','poly','krishna','‚äì','mansion','house','tub','excellence','olein','s¬†','flavor','scheme','shubh','stick','rich','case','asap','one','label','xxgm','indian','high','original','bot','litre','cut','xxg','bottles','canz','test','packs','jarxx','diff','amp','lbs'])
    #domain_words=domain_words.union(domain_words_extra)
    stop_words=englishStopWords.union(domain_words)
    return list(stop_words)


# In[ ]:



# In[ ]:


def generateNgramsLists(dfsubset):
    domainSpecificStr=''
    for i in range(0,dfsubset.shape[0]):
        #print(lstSKUNames[i])
        lstSKUNames=dfsubset.iloc[i]['normalized']
        #removePunctuationText=cu.cleanupSKUName(lstSKUNames)
        tokenizedString=nltk.word_tokenize(lstSKUNames)
        #tokens_without_sw=cu.removeStopWords(tokenizedString,getStopWords())
        tokens_without_sw=tokenizedString
        res=list(set(tokens_without_sw))
        if len(res)==0:
            res=tokenizedString
        domainSpecificStr+=' '.join(res)
        domainSpecificStr+=' ^^ '

    #print(domainSpecificStr)

    #generate the ngrams
    domainTokens = nltk.word_tokenize(domainSpecificStr)
    domain_unigrams=ngrams(domainTokens,1)
    domain_bigrams = ngrams(domainTokens,2)
    domain_trigrams = ngrams(domainTokens,3)
    domain_quadgrams = ngrams(domainTokens,4)
    return domain_unigrams,domain_bigrams,domain_trigrams,domain_quadgrams


def getAngleInRadian(cos_sim):
    # This was already calculated on the previous step, so we just use the value
    angle_in_radians = math.acos(float(cos_sim))
    return (math.degrees(angle_in_radians))


# In[ ]:


#domain_bigrams



''' Read the data from the database '''
def getDfSubsetFromDatabase(subcategory):
    sqlSuperQuery="SELECT Subject,`Ticket Id`, `Ticket Description`,CONCAT(Subject,' ',`Ticket Description`) as Ticket from CCDTickets2 where `Sub Category`='"+subcategory+"'"
    df=generalDBquery(sqlSuperQuery)
    dfsubset=df.replace(np.nan, '', regex=True)
    return dfsubset


def getListSubcategories():
    sqlSuperQuery="SELECT `Sub Category` as subcategory , count(*) from CCDTickets2 group by `Sub Category` order by count(*) DESC"
    df=generalDBquery(sqlSuperQuery)
    dfsubset=df.replace(np.nan, '', regex=True)
    return dfsubset

def getNgramDictionary(dfsubset):
    domain_unigrams,domain_bigrams,domain_trigrams,domain_quadgrams=generateNgramsLists(dfsubset)
    finalList=generateWordFrequencies(domain_unigrams,domain_bigrams,domain_trigrams,domain_quadgrams)
    #sort the dictionary
    newDict={k: v for k, v in sorted(finalList.items(), key=lambda item: item[1],reverse=True)}
    return newDict




#### text cleanup
def removeDigits(rawText):
    #https://stackoverflow.com/questions/12851791/removing-numbers-from-string/12856384
    from string import digits
    remove_digits = str.maketrans('', '', digits)
    res = rawText.translate(remove_digits)
    return res

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
    return case



def generateStopWordsFromEmails(): 
    sqlQueryEmails="Select distinct ticketEmail from CCDTickets3"
    df=generalDBquery(sqlQueryEmails)
    #print(df)
    stopwords=[]
    import re
    for i in range(df.shape[0]):
        try:
            #print(df.iloc[i]['ticketEmail'])
            emailName,domainName=df.iloc[i]['ticketEmail'].split('@')
            lstEmailPart=list(emailName.split('.'))
            [stopwords.append(preprocessText(item)) for item in lstEmailPart]

            lstDomainPast=list(domainName.split('.'))
            [stopwords.append(preprocessText(item)) for item in lstDomainPast]

        except:
            pass



    stopwords=list(set(stopwords))
    #print(stopwords)
    newStopWords=[]
    for stw in stopwords:
        #print(stw)
        lstStopwordsPart=stw.split(' ')
        #if(len(lstStopwordsPart)>1):
        [newStopWords.append(item) for item in lstStopwordsPart]


    #for w in ['purchase','not','bizom','connect', 'sfa','mobisy','in','dm','dms']:
    #   try:
    #        stopwords.remove(w)
    #    except:
    #        print(w,"not found")
    newStopWords.append("nan")
    newStopWords.append("re")
    newStopWords.append("fwd")
    newStopWords.append("fw")
    newStopWords.append("inc")


    stopwords=list(set(newStopWords))
    return stopwords


#def getStopWordsAsDict(lstStopwords):



def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text


def removeStopWords(text,stopwords):
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text



def preprocessText(doc,stopwords,shouldRemoveSW=True):
    #remove the digits
    doc=doc.lower()
    doc=removeDigits(doc)

    #remove the punctuations
    doc=removePunctuations(doc)
    #remove the accented words like '
    doc=remove_accented_chars(doc)
    #remove additional space in between the words
    doc = re.sub(' +', ' ', doc)
    # create the full words like 'll to will
    doc = contractions.fix(doc)
    #remove the stop words as per the stopwords dictionalry
    if(shouldRemoveSW==True):
        doc = removeStopWords(doc,stopwords)
    return doc



def flattenList(lst):
    list_flat = list(itertools.chain(*lst))
    list_flat_unique=list(set(list_flat))
    return list_flat_unique
