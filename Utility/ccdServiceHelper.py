import matplotlib.pyplot as plt 
import seaborn as sns
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
import json
import glob
from Utility import ccdUtilities as cu
import gensim
import logging
from Utility import ccdUtilities as cu
import re
from sentence_transformers import SentenceTransformer, util

### Custom functions to calulcate the mean and std
def myMean(x):
    ranges=np.quantile(x, [.01,0.90])
    lowerLimit=ranges[0]
    upperLimit=ranges[1]
    validValues=x[(x>=lowerLimit) & (x<=upperLimit)]
    mean=validValues.mean()
    return mean

def myStd(x):
    ranges=np.quantile(x, [.01,0.90])
    lowerLimit=ranges[0]
    upperLimit=ranges[1]
    validValues=x[(x>=lowerLimit) & (x<=upperLimit)]
    std=validValues.std()
    return std


### Function to fetch the data from the database

## get the data to work with...will take all the cases which are available from say 1-jan-2021
def getCaseData():
    sqlData="SELECT ticketId, companyId, subject,ticketDescription, subCategory,ticketOwner "\
    ",ticketCreatedTime,firstResponseTimeInHr,ticketClosedTime,resolutionTimeInHr," \
    "ticketOnHoldTime,totalResponseTimeInHr,ticketStatus,numOfResponses,ticketAge,customerRespondedTime," \
    "isEscalated,isOverdue,numberOfReopen,inProgressTime,numberOfReassign,numberOfComments,bugLink,numberOfOutgoing,numberOfThreads " \
    " FROM `CCDTickets3` where ticketStatus='Closed' and ticketId>'87000'"
    df=cu.generalDBquery(sqlData)
    return df



### functions to convert the column data into correct datatype
#1. updateDateTimeFieldTypes
#2. normalizeResolutionTime
#3. updateDataFrameWithshortCategoryName 

##update the data types of the time related columns
def updateDateTimeFieldTypes(df):
    df['ticketCreatedTime']=pd.to_datetime(df['ticketCreatedTime'])
    df['ticketClosedTime']=pd.to_datetime(df['ticketClosedTime'])
    df['customerRespondedTime']=pd.to_datetime(df['customerRespondedTime'])
    df['ticketCreatedDate']=df['ticketCreatedTime'].dt.date
    df['ticketClosedDate']=df['ticketClosedTime'].dt.date
    df['ticketCreatedYearMonth']=df['ticketCreatedTime'].dt.strftime('%Y-%m')
    df['ticketClosedYearMonth']=df['ticketClosedTime'].dt.strftime('%Y-%m')
    df['resolutionTimeInHr']=df['resolutionTimeInHr'].apply(lambda x:x.split(':')[0])
    df['firstResponseTimeInHr']=df['firstResponseTimeInHr'].apply(lambda x:x.split(':')[0])
    df.loc[df['firstResponseTimeInHr']=='-',"firstResponseTimeInHr"]=10000
    df['firstResponseTimeInHr']=df['firstResponseTimeInHr'].astype(int)
    df['totalClosureTime']=(df['ticketClosedTime']-df['ticketCreatedTime']).astype('timedelta64[h]')
    return df

def normalizeResolutionTime(df):
    df.loc[df['resolutionTimeInHr']=='-',"resolutionTimeInHr"]=0
    df['resolutionTimeInHr']=df['resolutionTimeInHr'].astype(int)
    df['resolutionTimeInHr'].value_counts()
    
    #normalize
    df['resolutionTimeInHrNormalized']="NA"
    df.loc[df['resolutionTimeInHr']<2,"resolutionTimeInHrNormalized"]="<2hr"
    df.loc[(df['resolutionTimeInHr']>2) & (df['resolutionTimeInHr']<=6) ,"resolutionTimeInHrNormalized"]="<6hr"
    df.loc[(df['resolutionTimeInHr']>6) & (df['resolutionTimeInHr']<=12),"resolutionTimeInHrNormalized"]="<12hr"
    df.loc[(df['resolutionTimeInHr']>12) & (df['resolutionTimeInHr']<=24),"resolutionTimeInHrNormalized"]="<24hr"
    df.loc[(df['resolutionTimeInHr']>24) & (df['resolutionTimeInHr']<=48),"resolutionTimeInHrNormalized"]="<2Day"
    df.loc[(df['resolutionTimeInHr']>48) & (df['resolutionTimeInHr']<=72),"resolutionTimeInHrNormalized"]="<3Day"
    df.loc[(df['resolutionTimeInHr']>72) & (df['resolutionTimeInHr']<=96),"resolutionTimeInHrNormalized"]="<4Day"
    df.loc[(df['resolutionTimeInHr']>96) & (df['resolutionTimeInHr']<=120),"resolutionTimeInHrNormalized"]="<5Day"
    df.loc[(df['resolutionTimeInHr']>120) & (df['resolutionTimeInHr']<=144),"resolutionTimeInHrNormalized"]="<5Day"

    df.loc[(df['resolutionTimeInHr']>144) & (df['resolutionTimeInHr']<=288),"resolutionTimeInHrNormalized"]="<2Week"
    df.loc[(df['resolutionTimeInHr']>288) & (df['resolutionTimeInHr']<=432),"resolutionTimeInHrNormalized"]="<3Week"
    df.loc[(df['resolutionTimeInHr']>432) & (df['resolutionTimeInHr']<=576),"resolutionTimeInHrNormalized"]="<1month"
    df.loc[(df['resolutionTimeInHr']>576),"resolutionTimeInHrNormalized"]=">1month"
    
    return df


## add the shot category names
#add a short category that will group the subcategory to some level
#"catMap.csv"
def updateDataFrameWithshortCategoryName(df,catNameFile="../catMap.csv"):
    df['shortCategory']=df['subCategory']
    cat=pd.read_csv(catNameFile)
    catDict=[]
    for c in cat.index:
        subCat=cat.iloc[c]['subCategory']
        mainCat=cat.iloc[c]['MainCategory']
        catDict.append({'subCat':subCat,'mainCat':mainCat})
    res = {sub['subCat'] : sub['mainCat'] for sub in catDict}
    df['shortCategory'].replace(res, inplace=True)
    return df


## functions for analysis
# check for a particular dimention variation like
# 1. getMonthlyTraffic: ticketCreatedYearMonth
# 2. getFirstResponseTimeData: firstResponseTimeInHr
# 3. getResolutionTimeData : resolutionTimeInHr
# 4. getSubCategoryDistribution: subCategory

def get1DimentionAnalysis(df,dim):
    return df[(df['companyId']!=215)][dim].value_counts(),df[(df['companyId']!=215)][dim].value_counts(normalize=True)


# category wise mean and std
# this helps in understanding which type of cases are taking 
# too much of time
def get2DimentionRelationship(df,dim1='subCategory',dim2='resolutionTimeInHr'):
    tt=df.groupby([dim1])[dim2].agg(['count',myMean,myStd])
    tt.sort_values(by=['myMean'],inplace=True,ascending=False)
    tt=pd.DataFrame(tt)
    tt.reset_index(inplace=True)
    tt.columns=[dim1,'count','mean','std']
    return tt

### get the defaulters who are not following the SLA properly
def getSLADefaulters(df):
    # s# return dfResTime.groupby(['companyId','ticketOwner']).size().sort_values(ascending=False).reset_index()[0:20]
    dfResTime=df[df['firstResponseTimeInHr']==10000][['ticketId','companyId','subject','resolutionTimeInHr','ticketOwner','firstResponseTimeInHr']]
    resDf=dfResTime.groupby(['companyId','ticketOwner']).size().sort_values(ascending=False).reset_index()[0:20]
    resDf.columns=['companyId','ticketOwner','numberTicketsDefaulted']
    return resDf 


## fetch the Dataframe
def fetchDataFrame(startDate,endDate,shortCategoryMappingFile):
    df=getCaseData()
    df=updateDateTimeFieldTypes(df)
    df=normalizeResolutionTime(df)
    df=updateDataFrameWithshortCategoryName(df,shortCategoryMappingFile)
    df=df[(df['ticketCreatedYearMonth']>=startDate) & (df['ticketCreatedYearMonth']<=endDate)]
    return df


######################################REOPENS############################
# get the reopens basis of MDM
# number_reopern>5,
# closure_time>100h
# number_thread>20
# subcat=MDM
def getReopenBasisMDM(df):
    resDf_MDM=df[(df['numberOfReopen']>5) & (df['totalClosureTime']>100) & (df['numberOfThreads']>20) & (df['shortCategory']=="MDM")]
    resDf_MDM['reopenReason']="MDM_RULE"
    return resDf_MDM


# get the reopens basis of Attendance
# mostly thins is the example of our mistake and the customer is not happy
def getReopenBasisATT(df):
    resDf_ATT=df[(df['subCategory']=="Attendance changes") & (df['numberOfReopen']>2) & (df['totalClosureTime']>50)]
    resDf_ATT['reopenReason']="ATT_RULE"
    return resDf_ATT

#in general if the number of threads are high ...dont know if we should consider the companyId=215 or not
def getReopenBasisGEN(df):
    resDf_GEN=df[(df['numberOfReopen']>5) & (df['totalClosureTime']>100) & (df['numberOfThreads']>30) & (df['shortCategory']!='MOM')]
    resDf_GEN['reopenReason']="GEN_RULE"
    return resDf_GEN


# get the subcat reopens
# this is the real deal where we have to look into the history files to see 
# which tickets have got multiple subcategory transitions
#get the reopens basis of the subcategory cahnages ..these are the ones which have been changed by the customer

def generateTransitionsFromHistoryData(outputFile,directoryPath="historyData"):
    with open(str(outputFile), encoding='utf-8-sig', mode='w') as fp:
        listing = glob.glob(str(directoryPath)+'/*.json')
        print(len(listing))
        for filename in listing:
            print("opeing the file",filename)
            with open(filename) as f:
                try:
                    data = json.load(f)
                    dict_list=data['data']
                    #getTransitions(filename,dict_list)
                    caseId=filename.split('_')[1]


                    for dictionary in dict_list:
                        for section in dictionary['historys']:
                            #print(section)
                            if('transitionName' in section.keys()):
                                print(filename,":",section['transitionName'],section['HistoryDate'],section['displayTime'])
                                print("\n=====\n")

                            if('operation' in section.keys()):
                                if(section['operation']=="Blueprint Transition"):
                                    continue                    

                            if('ImgClass' in section.keys()):
                                if(section['ImgClass']=="i-tag icon-ticket-reopen"):
                                    print(filename,":", section['HistoryLabel'], 
                                          section['HistoryDate'],section['displayTime'])
                                    print(section['UpdatedFields'][0]['oldValue'],"-->",section['UpdatedFields'][0]['newValue'])
                                    print("\n=====\n")


                                elif(section['ImgClass']=="i-tag icon-update-ticket"):
                                    listUpdateDetailsDict=section['UpdatedFields']
                #                     print("llll",listUpdateDetailsDict)

                                    for dict in listUpdateDetailsDict:
                                        if(dict['fieldLabel']=="Sub Category"):
                                            print(filename,":","subCategoryCahnage",dict['oldValue'],"--->",dict['newValue'],
                                                  " on ",section['HistoryDate'],section['displayTime'])
                                            fp.write('{}|{}|{}|{}|{}|{}\n'.format(filename,caseId,dict['oldValue'],dict['newValue'],section['HistoryDate'],section['displayTime']))
                                else:
                                    pass
                except:
                    print("error in parsing json")
    return "FILE CREATED"


def getReopenBasisSUBCAT(df,outputFile="history_result",directoryPath="historyData"):
#     outputFile="history_result"
#     directoryPath="historyData"
    status=generateTransitionsFromHistoryData(outputFile,directoryPath)
    if(status=="FILE CREATED"):
        historyDataDf=pd.read_csv(outputFile,sep='|',header=None)
        historyDataDf.columns=['filename','caseId','initalCat','changedCat','date','time']
        #create a datetime field and sort the results as  per dateresult ..it will help in finding the transitions
        historyDataDf['datetime']=pd.to_datetime(historyDataDf['date']+" "+historyDataDf['time'])
        historyDataDf=historyDataDf.sort_values(['datetime'],axis=0)


        ##now take all the cases whose occurence is more than 1
        tmpHistoryDataDf=pd.DataFrame(historyDataDf['caseId'].value_counts()>1)
        duplicates=list(tmpHistoryDataDf[tmpHistoryDataDf['caseId']==True].index)
        for i in duplicates:
            subdf=historyDataDf[historyDataDf['caseId']==i].reset_index()
            print("-----------------")
            for j in subdf.index:
                print(subdf.iloc[j]['caseId']," changed from:",subdf.iloc[j]['initalCat'],"to-->",subdf.iloc[j]['changedCat'])

        ##checkput in the actual dataframe
        resDf_HistSubCat=df[df['ticketId'].isin(duplicates) & (df['shortCategory']!='MOM')]
        resDf_HistSubCat['reopenReason']="SUBCAT_RULE"
        return resDf_HistSubCat


# get the reoprn basis the tickets which contains multiple bugs for the same case
# ideally there should be one bug per ticket..but if there are many that is an indicator of unnceesary reipen
def findCasesWithMultipleBugs(outputFile,directoryPath="historyData"):
    listing = glob.glob(str(directoryPath)+'/*.json')
    with open(str(outputFile), encoding='utf-8-sig', mode='w') as fp:
        for filename in listing:
            lstLinks=[]
            caseId=filename.split('_')[1]
            print("openeing",filename)
            with open(filename) as f:
                try:
                    data = json.load(f)
                    dict_list=data['data']
                    link_re = re.compile('https://bugzilla.bizom.in/show_bug.cgi\?id=\d+')
                    links = link_re.findall(str(dict_list))
                    for link in links:
                        link = link.replace('</a>', '')
                        link = link.replace('>', '')
                        link = link.replace('https://bugzilla.bizom.in/show_bug.cgi?id=', '')
                        fp.write('{}|{}|{}\n'.format(filename,caseId,link))
                except:
                    print("something problematic with the file")
    return "FILE CREATED" 


def getReopenBasisBUGS(df,outputFile="history_bugs",directoryPath="historyData"):
#     outputFile="history_bugs"
#     directoryPath="historyData"
    status=findCasesWithMultipleBugs(outputFile,directoryPath)
    if(status=="FILE CREATED"):
        historyDataDf=pd.read_csv(outputFile,sep='|',header=None)
        historyDataDf.columns=['filename','caseId','bugId']
        ##we need to remove the duplicates where the filename and bug id is smae..
        ##prob the same bug has been mentioned multiple time in the file
        historyDataDf.sort_values("caseId", inplace=True)
        historyDataDf.drop_duplicates(subset =['caseId','bugId'],keep="first",inplace=True)

        tmpHistoryDataDf=pd.DataFrame(historyDataDf['caseId'].value_counts()>1)
        defaulters=list(tmpHistoryDataDf[tmpHistoryDataDf['caseId']==True].index)
        ##checkput in the actual dataframe
        resDf_multiBugs=df[df['ticketId'].isin(defaulters) & (df['shortCategory']!='MOM')]
        resDf_multiBugs['reopenReason']="BUGS_RULE"
        return resDf_multiBugs


## to find out the details of teh cases which were responded by the customer but
## we are still holding them
def getCustomerRespondedCasesData():
    #sqlData="SELECT uid,ticketId,ticketStatus from customerRepliedCCDTickets where ticketId>'87000'"
    sqlData="SELECT cr.uid as uid,cr.ticketId as ticketId," \
        "cr.ticketStatus as ticketStatus,cr.subject as subject," \
        "cr.accountName as accountName,ccd.accountCategory as accountCategory," \
        "ccd.ticketOwner as ticketOwner FROM customerRepliedCCDTickets as cr " \
        "left join CCDTickets3 as ccd on cr.ticketId=ccd.ticketId"
    df=cu.generalDBquery(sqlData)
    return df


##given a bug level we have to search in the bugzilla deta base 
def getBugStatus():
    # sqlData="SELECT ccd.uid as uid," \
    #     "cr.ticketId as ticketId,cr.buglink as buglink,cr.subject as subject FROM CCDTickets3 as cr " \
    #     "left join ticket_case_mapping as ccd  on cr.ticketId=ccd.ticketId where  bugLink!='-' and bugLink is not NULL and ccd.ticketId>100000;"
    # df=cu.generalDBquery(sqlData)
    sqlData="SELECT `uid`, CCDTicketBugStatus.ticketId as ticketId, `bugListedIn`, CCDTicketBugStatus.subject as subject, CCDTicketBugStatus.buglink as buglink, `bugId`, `bugStatus`, `bugResolution`, `productName`, `componentName` FROM CCDTicketBugStatus left join CCDTickets3 on CCDTicketBugStatus.ticketId=CCDTickets3.ticketId where CCDTickets3.ticketStatus in ('Open','On hold','In Progress') and bugStatus!='NA'"
    df=cu.generalDBquery(sqlData)
    return df




##get the company data
def getCompanyData():
    sqlCompany="SELECT distinct companyId,accountName FROM `CCDTickets3`"
    dfCompany=cu.generalDBquery(sqlCompany)
    dfCompany=dfCompany.dropna()
    dfCompany['companyId']=dfCompany['companyId'].astype(int)
    dfCompany=dfCompany.set_index('companyId')
    return dfCompany
