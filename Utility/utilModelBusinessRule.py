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

def applyBusinessRules(df_res):
    countNC=0
    countNP=0
    countCP=0
    countPP=0
    countNN=0
    countCH=0
    countBP=0

    finalres=[]
    for i in df_res.index:
        row=df_res.iloc[i]
        classificationAlgoOutput=row['classificationAlgoOutput']
        if(classificationAlgoOutput=='store'):
            classificationAlgoOutput='Store'

        majorityAlgoClassification=row['majorityAlgoClassification']
        majorityCompanyProvidedClassification=row['majorityCompanyProvidedClassification']
        labels_predicted=row['labels_predicted']
        if(len(majorityAlgoClassification)>4 and (majorityAlgoClassification !='store' or majorityAlgoClassification!='Store') and (len(classificationAlgoOutput)<4 or classificationAlgoOutput=='Store')):
                classificationAlgoOutput=majorityAlgoClassification

        classificationAlgoOutput=classificationAlgoOutput.strip()

        primaryCompanyClassificationShortName=row['primaryCompanyClassificationShortName']
        #len(primaryCompanyClassificationShortName)<3 means that it is empty

        if(len(primaryCompanyClassificationShortName)<3 and len(majorityCompanyProvidedClassification)>5):
            primaryCompanyClassificationShortName=majorityCompanyProvidedClassification
    # 
    #nameBasedAlgo=companyCategory..which is a high confidence scenario
        if(len(primaryCompanyClassificationShortName)>3 and (primaryCompanyClassificationShortName == classificationAlgoOutput)):
            countNC+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":primaryCompanyClassificationShortName,"rule":"NC"})

        #nameBasedAlgo=predictions...that means prediction matches the algo..this is a success scenrio
        elif(len(classificationAlgoOutput)>4 and (classificationAlgoOutput==labels_predicted) and (primaryCompanyClassificationShortName != classificationAlgoOutput)):
            countNP+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":classificationAlgoOutput,"rule":"NP"})

        #companyCategory=prediction and companyCategory!=namedBasedAlgo
        elif(len(primaryCompanyClassificationShortName)>3 and (primaryCompanyClassificationShortName!=classificationAlgoOutput) and (primaryCompanyClassificationShortName == labels_predicted) ):
            countCP+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":primaryCompanyClassificationShortName,"rule":"CP"})

        #when label != name and company =''
        elif(len(primaryCompanyClassificationShortName)<3 and len(classificationAlgoOutput)>4 and classificationAlgoOutput!='Store' and labels_predicted!='' and classificationAlgoOutput!=labels_predicted):
            countNN+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":classificationAlgoOutput,"rule":"NN1"})

        elif(classificationAlgoOutput=='Chemist'):
            countCH+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":"Chemist","rule":"CH"})

        elif(classificationAlgoOutput=='Bakery/Provision'):
            countBP+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":"Bakery/Provision","rule":"BP"})

        #classification !=company!=Label and company!='' and N!=store then N
        elif(len(primaryCompanyClassificationShortName)>3 and len(classificationAlgoOutput)>4 and classificationAlgoOutput!='Store' and labels_predicted!='' and classificationAlgoOutput!=labels_predicted and classificationAlgoOutput!=primaryCompanyClassificationShortName):
            countNN+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":classificationAlgoOutput,"rule":"NN2"})


        #classification !=company!=Label and company!='' and N=store then c
        elif(len(primaryCompanyClassificationShortName)>3 and len(classificationAlgoOutput)>4 and classificationAlgoOutput=='Store' and labels_predicted!='' and classificationAlgoOutput!=labels_predicted and classificationAlgoOutput!=primaryCompanyClassificationShortName):
            countNN+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":primaryCompanyClassificationShortName,"rule":"NN3"})

        #classification !=company!=Label and company='' and N=store then L also l should not be chemist
        elif(len(primaryCompanyClassificationShortName)<3 and labels_predicted!='Chemist' and len(classificationAlgoOutput)>4 and classificationAlgoOutput=='Store' and labels_predicted!='' and classificationAlgoOutput!=labels_predicted and classificationAlgoOutput!=primaryCompanyClassificationShortName):
            countNN+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":labels_predicted,"rule":"NN4"})

        #company!=Label and company!='' and N='' then C
        elif(len(primaryCompanyClassificationShortName)>3 and len(classificationAlgoOutput)<4 and labels_predicted!='' and primaryCompanyClassificationShortName!=labels_predicted):
            countNN+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":primaryCompanyClassificationShortName,"rule":"NN5"})


        else:
            countPP+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":labels_predicted,"rule":"PP"})



    print("countNC",countNC)
    print("countCP",countCP)
    print("countNP",countNP)
    print("countPP",countPP)
    print("countNN",countNN)
    print("countCH",countCH)
    print("countBP",countBP)
    return finalres



def applyBusinessRules_old(df_res):
    countNC=0
    countNP=0
    countCP=0
    countPP=0
    countNN=0
    countCH=0
    countBP=0

    finalres=[]
    for i in df_res.index:
        row=df_res.iloc[i]
        classificationAlgoOutput=row['classificationAlgoOutput']
        if(classificationAlgoOutput=='store'):
            classificationAlgoOutput='Store'

        majorityAlgoClassification=row['majorityAlgoClassification']
        majorityCompanyProvidedClassification=row['majorityCompanyProvidedClassification']
        labels_predicted=row['labels_predicted']
        if(len(majorityAlgoClassification)>4 and (majorityAlgoClassification !='store' or majorityAlgoClassification!='Store') and (len(classificationAlgoOutput)<4 or classificationAlgoOutput=='Store')):
                classificationAlgoOutput=majorityAlgoClassification

        classificationAlgoOutput=classificationAlgoOutput.strip()

        primaryCompanyClassificationShortName=row['primaryCompanyClassificationShortName']
        #len(primaryCompanyClassificationShortName)<3 means that it is empty

        if(len(primaryCompanyClassificationShortName)<3 and len(majorityCompanyProvidedClassification)>5):
            primaryCompanyClassificationShortName=majorityCompanyProvidedClassification
    # 
    #nameBasedAlgo=companyCategory..which is a high confidence scenario
        if(len(primaryCompanyClassificationShortName)>3 and (primaryCompanyClassificationShortName == classificationAlgoOutput)):
            countNC+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":primaryCompanyClassificationShortName,"rule":"NC"})

        #nameBasedAlgo=predictions...that means prediction matches the algo..this is a success scenrio
        elif((classificationAlgoOutput==labels_predicted) and (primaryCompanyClassificationShortName != classificationAlgoOutput)):
            countNP+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":classificationAlgoOutput,"rule":"NP"})

        #companyCategory=prediction and companyCategory!=namedBasedAlgo
        elif(len(primaryCompanyClassificationShortName)>3 and (primaryCompanyClassificationShortName!=classificationAlgoOutput) and (primaryCompanyClassificationShortName == labels_predicted) ):
            countCP+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":primaryCompanyClassificationShortName,"rule":"CP"})

        #when label != name and company =''
        elif(len(primaryCompanyClassificationShortName)<3 and len(classificationAlgoOutput)>4 and classificationAlgoOutput!='Store' and labels_predicted!='' and classificationAlgoOutput!=labels_predicted):
            countNN+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":classificationAlgoOutput,"rule":"NN1"})

        elif(classificationAlgoOutput=='Chemist'):
            countCH+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":"Chemist","rule":"CH"})

        elif(classificationAlgoOutput=='Bakery/Provision'):
            countBP+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":"Bakery/Provision","rule":"BP"})

        #classification !=company!=Label and company!='' and N!=store then N
        elif(len(primaryCompanyClassificationShortName)>3 and len(classificationAlgoOutput)>4 and classificationAlgoOutput!='Store' and labels_predicted!='' and classificationAlgoOutput!=labels_predicted and classificationAlgoOutput!=primaryCompanyClassificationShortName):
            countNN+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":classificationAlgoOutput,"rule":"NN2"})


        #classification !=company!=Label and company!='' and N=store then c
        elif(len(primaryCompanyClassificationShortName)>3 and len(classificationAlgoOutput)>4 and classificationAlgoOutput=='Store' and labels_predicted!='' and classificationAlgoOutput!=labels_predicted and classificationAlgoOutput!=primaryCompanyClassificationShortName):
            countNN+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":primaryCompanyClassificationShortName,"rule":"NN3"})

        #classification !=company!=Label and company='' and N=store then L
        elif(len(primaryCompanyClassificationShortName)<3 and len(classificationAlgoOutput)>4 and classificationAlgoOutput=='Store' and labels_predicted!='' and classificationAlgoOutput!=labels_predicted and classificationAlgoOutput!=primaryCompanyClassificationShortName):
            countNN+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":labels_predicted,"rule":"NN4"})

        #company!=Label and company!='' and N='' then C
        elif(len(primaryCompanyClassificationShortName)>3 and len(classificationAlgoOutput)<4 and labels_predicted!='' and primaryCompanyClassificationShortName!=labels_predicted):
            countNN+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":primaryCompanyClassificationShortName,"rule":"NN5"})


        else:
            countPP+=1
            finalres.append({"globalId":row['globalId'],"outletId":row['outletId'],"companyId":row['companyId'],"outletName":row['outletName'],"totalProducts":row['totalProducts'],"avgSales":row['avgSales'],"nameBased":classificationAlgoOutput,"predictedLabel":labels_predicted,"companyCategory":primaryCompanyClassificationShortName ,"finalOutput":labels_predicted,"rule":"PP"})



    print("countNC",countNC)
    print("countCP",countCP)
    print("countNP",countNP)
    print("countPP",countPP)
    print("countNN",countNN)
    print("countCH",countCH)
    print("countBP",countBP)
    return finalres