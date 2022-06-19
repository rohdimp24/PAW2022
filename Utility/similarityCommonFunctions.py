"""
this file is to provide the functionality for the support of FAISS
"""
import nltk
from nltk.corpus import webtext 
from nltk import FreqDist
from nltk.util import ngrams   
# use to find bigrams, which are pairs of words 
from nltk.collocations import BigramCollocationFinder 
from nltk.metrics import BigramAssocMeasures 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
import MySQLdb
import pandas as pd
import mysql.connector
import sqlalchemy 
from sqlalchemy import create_engine
import psycopg2
import numpy as np
import pandas as pd
import mysql.connector
import glob
import os
from urllib.parse import urlparse
from urllib.parse import urlparse, parse_qs
import statistics
import math
from nltk.util import ngrams
from collections import Counter
#from Utility import commonUtilities as cu
import logging
import re
from Utility import ccdUtilities as cu
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Used to create and store the Faiss index.
import faiss
import numpy as np
import pickle
from pathlib import Path

'''
Function to get the meta information about the various cases
'''
def getCCDData():
    sqlCompany="SELECT companyId,uid,CCDTickets3.ticketId as ticketId,ticket_case_mapping.uid,subject,subCategory,accountName,ticketCreatedTime FROM `CCDTickets3` " \
        "left join ticket_case_mapping on ticket_case_mapping.ticketId=CCDTickets3.ticketId where CCDTickets3.ticketId>87000"
    dfTicketDetails=cu.generalDBquery(sqlCompany)
    dfTicketDetails=dfTicketDetails.dropna()
    dfTicketDetails['ticketId']=dfTicketDetails['ticketId'].astype(int)
    return dfTicketDetails

'''
Functio to get the case descriptuons. This is when we read the 
raw similarity data and then process them
'''
def getCaseDescriptions(casesFilename='ZohoCaseDescDataFinal',seperator="<ROHIT>"):
    dfCases=pd.read_csv(casesFilename,sep=seperator,header=None)
    dfCases.columns=['filename','ticketIdPath','caseDesc']
    dfCases=dfCases.dropna()
    dfCases.reset_index(inplace=True)
    dfCases.head()
    
    ## add the ticket id
    dfCases['ticketId']=dfCases['ticketIdPath']
    dfCases['ticketId']=dfCases['ticketId'].astype(int)
    
    ##get the other data rfrom the main datanase
    dfTicketDetails=getCCDData()
    mergedCases=pd.merge(dfCases,dfTicketDetails,how="left",on="ticketId")
    return mergedCases

def removePunctuations(rawText):
    case=rawText
    case = case.strip();
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

def getWordReplacement(word):
    returnWord=word
    try:
        returnWord=dfPOSTags[dfPOSTags['word']==word]['correctedWord'].values[0]
    except:
        print("word not found",returnWord)
        returnWord="<NF>"+word+"</NF>"
    return returnWord

def removeNonPOSWords(text,allowedWords):
    text = ' '.join([getWordReplacement(word) for word in text.split() if word in allowedWords])
    return text

def checkPresence(pattern,hay):
    pos=hay.find(pattern)
    return pos

def removeWord(word,text):
    text=text.replace(word," ")
    return text

def helperApplyFilters(text,allowedWords):
    #remove the disclaimer information if any
    originalText=text
    text=filterText("disclaimer",text)
#     print("got",text)
    text=removeNonPOSWords(text,allowedWords)
#     print(ticketId,"=> ",originalText," => ",text)
    if(len(text.split(' '))<2):
        print(text,"..Too small")
        return removePunctuations(originalText)
    else:
        return text

def filterText(checkStopWord,inputText):
    if(checkPresence(checkStopWord,inputText)>-1):
        pos=checkPresence(checkStopWord,inputText)
#         print(" found ",checkStopWord,pos)
#         print(">>>>>>>>>>>>>>>>>>>")
#         print(inputText[0:pos])
        return inputText[0:pos]
            #check for the emails
    else:
        return inputText




def generateNgramsListsNew(lstSentences):
    domainSpecificStr=''
    for i in range(0,len(lstSentences)):
        #print(lstSKUNames[i])
        #lstSKUNames=dfsubset.iloc[i]['POSCaseDesc']
        #removePunctuationText=cu.cleanupSKUName(lstSKUNames)
        case=lstSentences[i]
        case = case.replace('<SUBJECT>',' ')
        case = case.replace('<DESC>',' ')
        tokenizedString=nltk.word_tokenize(case)
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
#     domain_trigrams = ngrams(domainTokens,3)
#     domain_quadgrams = ngrams(domainTokens,4)
    return domain_unigrams,domain_bigrams


def generateWordFrequencies(domain_unigrams,domain_bigrams):
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
#                     fp.write('{}|{}|{}\n'.format("unigram",unigramString, count))



    cntObj=Counter(domain_bigrams)
    sortedCounterList=cntObj.most_common()
    for tag, count in sortedCounterList:
        bigramString=tag[0]+"_"+tag[1]
        #print(bigramString,count)
        if(bigramString.find(delimitPattern) == -1):
            if(count>threshold):
                finalList[bigramString]=count
#                     fp.write('{}|{}|{}\n'.format("bigram",bigramString, count))
    return finalList


def getKeywords(lstSentences):
    domainUnigrams,domainBigrams=generateNgramsListsNew(lstSentences)
    finalList=generateWordFrequencies(domainUnigrams,domainBigrams)
    newDict={k: v for k, v in sorted(finalList.items(), key=lambda item: item[1],reverse=True)}
    count=0
    lstDict=list(newDict.keys())
    keys=''
    lstWords=[]
#     if(len(lstDict)>5):
#         count=5

    for i in range(0,len(lstDict)):
        print("checking for",lstDict[i])
        if lstDict[i] in allowedWords:
            count+=1
            if(count<5):
                keys+=lstDict[i]+","
                lstWords.append({lstDict[i]:newDict[lstDict[i]]})
            else:
                break
    return keys,lstWords


which = lambda lst:list(np.where(lst)[0])

##functions related to the FAISS

def vector_search(query, FAISindex, num_results=10):
    """Tranforms query to vector using a pretrained, sentence-level 
    DistilBERT model and finds similar vectors using FAISS.
    """
    BERTModel='distilbert-base-nli-stsb-mean-tokens'
    model = SentenceTransformer(BERTModel)
    # Check if GPU is available and use it
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    print(model.device)
    vector = model.encode(list(query))
    L2Distances,matchingIndexes = FAISindex.search(np.array(vector).astype("float32"), k=num_results)
    return L2Distances,matchingIndexes


def id2details(df, indexes, df_indexingColumn):
#     for ticketId in I[0]:
#         #print(ticketId,"..")
#         print(dfCasesSubset[dfCasesSubset['ticketId']==ticketId]['filterCaseDesc'])
# #     """Returns the paper titles based on the paper index."""
# #     return [list(df[df.id == idx][column]) for idx in I[0]]
    displayDf=df[df[df_indexingColumn].isin(indexes)]
    return displayDf


def generateInputDataBERTEmbeddings(df,column_to_embed,BERTModel='distilbert-base-nli-stsb-mean-tokens'):
    model = SentenceTransformer(BERTModel)
    # Check if GPU is available and use it
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    print(model.device)
    #get the embeddings
    embeddings = model.encode(df[column_to_embed].to_list(), show_progress_bar=True)
    print(f'Shape of the vectorised abstract: {embeddings[0].shape}')
    
    return embeddings,model


def generateFAISSIndex(embeddings, df):
    # Step 1: Change data type
    embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

    # Step 2: Instantiate the index
    index = faiss.IndexFlatL2(embeddings.shape[1])

    # Step 3: Pass the index to IndexIDMap
    index = faiss.IndexIDMap(index)

    # Step 4: Add vectors and their IDs
    index.add_with_ids(embeddings, df.ticketId.values)

    print(f"Number of vectors in the Faiss index: {index.ntotal}")
    return index,embeddings



##case similarityy...which is given a case id find the similar cases
def getSimilarCasesBasisExistingCase(inputTicketId,df,FAISSindex,FAISSembeddings):
    #get the index id for the ticket
    queryRecordIndex=df[df['ticketId']==inputTicketId].index[0]
    
    L2Distances,matchingIndexes = FAISSindex.search(np.array([FAISSembeddings[queryRecordIndex]]), k=10)
    print(f'L2 distance: {L2Distances.flatten().tolist()}\n\nMAG paper IDs: {matchingIndexes.flatten().tolist()}')
    return L2Distances.flatten().tolist(),matchingIndexes.flatten().tolist()

# def getSimilarCasesBasedOnQuery(userQuery,model,df,FAISSindex):
#     lstDistances,lstMatchingIndexes=vector_search([userQuery], model, FAISSindex, num_results=10)
#     return lstDistances,lstMatchingIndexes

def getSimilarCasesBasedOnQuery(userQuery,df,FAISSindex):
    lstDistances,lstMatchingIndexes=vector_search([userQuery], FAISSindex, num_results=10)
    return lstDistances,lstMatchingIndexes




#### FACADES

def getCaseDescDataFrame(caseDescFile="ZohoCaseDescDataFinal",shortCatMappingFile="catMap.csv"):
    dfCasesSubset=getCaseDescriptions(caseDescFile)
    dfCasesSubset['filterCaseDesc']=dfCasesSubset['caseDesc']
    dfCasesSubset['POSCleanedDesc']=dfCasesSubset['caseDesc']
    dfCasesSubset=dfCasesSubset.dropna(subset=['companyId'])
    dfCasesSubset['companyId']=dfCasesSubset['companyId'].astype(int)
    dfCasesSubset['ticketCreatedTime']=pd.to_datetime(dfCasesSubset['ticketCreatedTime'])
    dfCasesSubset=dfCasesSubset.reset_index()


    toBeRemovedIndexes=[]
    
    for i in dfCasesSubset.index:
        try:
            ticketId=dfCasesSubset.iloc[i]['ticketId']
            caseDesc=dfCasesSubset.iloc[i]['caseDesc']
            subject=dfCasesSubset.iloc[i]['subject']
            subject=subject.lower()
            filterCaseDesc=removeWord("PERSON",caseDesc)
            print(ticketId,"=>removing person=>",filterCaseDesc)
            sections=filterCaseDesc.split("<SECTION>")
            numberSections=len(sections)
            if(len(sections[0])<=4):
                filterCaseDesc=sections[1]
                filterCaseDesc=re.sub(' +', ' ', filterCaseDesc)
                print(ticketId, "=>inthere=>", filterCaseDesc)
            else:
                filterCaseDesc=sections[0]
                filterCaseDesc=re.sub(' +', ' ', filterCaseDesc)
                print(ticketId,"=>after section=>",len(sections[0]),"=>",filterCaseDesc)
            finalText="<SUBJECT>"+" "+subject+" "+"<DESC>"+filterCaseDesc
            dfCasesSubset.loc[i,"filterCaseDesc"]=finalText
            dfCasesSubset.loc[i,"POSCleanedDesc"]=""
        except:
            toBeRemovedIndexes.append(i)
            print(ticketId,"=>",caseDesc,"---","NOT ABLE TO PROCESS")

    #remove the cases where subject is nan
    print(toBeRemovedIndexes)
    if(len(toBeRemovedIndexes)>0):
        dfCasesSubset.drop(dfCasesSubset.index[toBeRemovedIndexes],inplace=True)
        dfCasesSubset=dfCasesSubset


    ##update the subcategoru
    dfCasesSubset['shortCategory']=dfCasesSubset['subCategory']
    cat=pd.read_csv("catMap.csv")
    catDict=[]
    for c in cat.index:
        subCat=cat.iloc[c]['subCategory']
        mainCat=cat.iloc[c]['MainCategory']
        catDict.append({'subCat':subCat,'mainCat':mainCat})
    res = {sub['subCat'] : sub['mainCat'] for sub in catDict}   
    dfCasesSubset['shortCategory']=dfCasesSubset['subCategory'].map(res)

    return dfCasesSubset


def createFAISSIndex(dfCasesSubset):

    ### first get the emebeddings 
    BERTembeddings,model=generateInputDataBERTEmbeddings(dfCasesSubset,"filterCaseDesc",'distilbert-base-nli-stsb-mean-tokens')
    FAISSindexFilterCase,_=generateFAISSIndex(BERTembeddings, dfCasesSubset)

    BERTembeddingsOnlySubject,model=generateInputDataBERTEmbeddings(dfCasesSubset,"subject",'distilbert-base-nli-stsb-mean-tokens')
    FAISSindexOnlySubject,_=generateFAISSIndex(BERTembeddingsOnlySubject, dfCasesSubset)

    return FAISSindexFilterCase,FAISSindexOnlySubject

'''
Function to get the similarity scrore based on sentence transformer
'''
def getCaseSimilarityUsingSentenceTransformer(dfCasesSubset):
    # create a list of the caseDesc
    lstCaseDescIds=list(dfCasesSubset[0:]['filterCaseDesc'])
    lstTicketIds=list(dfCasesSubset[0:]['ticketId'])
    lstCompanyIds=list(dfCasesSubset[0:]['companyId'])
    lstSubjectIds=list(dfCasesSubset[0:]['subject'])
    lstCompanyNameIds=list(dfCasesSubset[0:]['accountName'])
    lstSubCategoryIds=list(dfCasesSubset[0:]['subCategory'])
    lstUIDs=list(dfCasesSubset[0:]['uid'])

    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    paraphrases = util.paraphrase_mining(model, lstCaseDescIds)
    dfPara=pd.DataFrame(paraphrases)
    dfPara.columns=['score','originalTicketIndex','matchingTicketIndex']

    ##need to add the additonal fields in the dataframe
    dfPara['originalContent']=dfPara['originalTicketIndex']
    dfPara['matchingContent']=dfPara['matchingTicketIndex']

    dfPara['originalSubject']=dfPara['originalTicketIndex']
    dfPara['matchingSubject']=dfPara['matchingTicketIndex']

    dfPara['originalCompanyId']=dfPara['originalTicketIndex']
    dfPara['matchingCompanyId']=dfPara['matchingTicketIndex']

    dfPara['originalCompanyName']=dfPara['originalTicketIndex']
    dfPara['matchingCompanyName']=dfPara['matchingTicketIndex']

    dfPara['originalSubCategory']=dfPara['originalTicketIndex']
    dfPara['matchingSubCategory']=dfPara['matchingTicketIndex']

    dfPara['originalUID']=dfPara['originalTicketIndex']
    dfPara['matchingUID']=dfPara['matchingTicketIndex']

    ##applying the transformation
    dfPara['originalContent']=dfPara['originalTicketIndex'].apply(lambda x: lstCaseDescIds[x])
    dfPara['matchingContent']=dfPara['matchingTicketIndex'].apply(lambda x: lstCaseDescIds[x])

    dfPara['originalSubject']=dfPara['originalTicketIndex'].apply(lambda x:lstSubjectIds[x])
    dfPara['matchingSubject']=dfPara['matchingTicketIndex'].apply(lambda x:lstSubjectIds[x])

    dfPara['originalCompanyId']=dfPara['originalTicketIndex'].apply(lambda x:lstCompanyIds[x])
    dfPara['matchingCompanyId']=dfPara['matchingTicketIndex'].apply(lambda x:lstCompanyIds[x])

    dfPara['originalCompanyName']=dfPara['originalTicketIndex'].apply(lambda x:lstCompanyNameIds[x])
    dfPara['matchingCompanyName']=dfPara['matchingTicketIndex'].apply(lambda x:lstCompanyNameIds[x])

    dfPara['originalSubCategory']=dfPara['originalTicketIndex'].apply(lambda x:lstSubCategoryIds[x])
    dfPara['matchingSubCategory']=dfPara['matchingTicketIndex'].apply(lambda x:lstSubCategoryIds[x])


    dfPara['originalUID']=dfPara['originalTicketIndex'].apply(lambda x:lstUIDs[x])
    dfPara['matchingUID']=dfPara['matchingTicketIndex'].apply(lambda x:lstUIDs[x])

    dfPara['originalTicketIndex']=dfPara['originalTicketIndex'].apply(lambda x: lstTicketIds[x])
    dfPara['matchingTicketIndex']=dfPara['matchingTicketIndex'].apply(lambda x: lstTicketIds[x])

    return dfPara

'''
Get the percent of duplicate of the cases based on the dimention
mainDimention is the name of the column which decides whose duplicate ypu want like companywise then originalCompanyId
..in case of the subcategory it will be originalSubCategory
supportingDimention is the ticketid in the subdf and maindf. in the subdf it wil be originalTicketIndex
mainDimentionLabel: same as the original dimention but this time it is in the datacasedesc dataframe
'''
def getDuplicateOutletPercent(mainDf,subDf,mainDimention,supportingDimention,mainDimentionLabel):
    subDfGrpLevel1=subDf.groupby([mainDimention,supportingDimention]).size().reset_index()
    subDfGrpLevel2=subDfGrpLevel1.groupby(mainDimention).size().reset_index()
    subDfGrpLevel2.columns=[mainDimentionLabel,'count']
    subDfGrpLevel2.sort_values(by=['count'],ascending=False,inplace=True)

    mainSubDfGrpLevel1=mainDf.groupby([mainDimentionLabel,'ticketId']).size().reset_index()
    mainSubDfGrpLevel2=mainSubDfGrpLevel1.groupby(mainDimentionLabel).size().reset_index()
    mainSubDfGrpLevel2.columns=[mainDimentionLabel,'count']
    mainSubDfGrpLevel2.sort_values(by=['count'],ascending=False,inplace=True)

    finalMergedOutput=pd.merge(mainSubDfGrpLevel2,subDfGrpLevel2,how="left",on=mainDimentionLabel)
    finalMergedOutput['percentDuplicates']=100*(finalMergedOutput['count_y']/finalMergedOutput['count_x'])
    return finalMergedOutput