from sklearn import datasets
import math
import numpy as np
import ctypes as ct
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
import time
import re
import os
import sys
import matplotlib.pyplot as plt
import multiprocessing

dll=np.ctypeslib.load_library('deepboost_so','.') #for using functions in deepboost.so
bases = ['A', 'C', 'G', 'U']
base_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
bases_len = len(bases)
num_feature = 271 #total number of features

#hyperparameters for the deep boosting algorithm
beta=0
lamda=0.3
depth=5
type=0
round=200

def convert_to_index(str,word_len):
   '''
   convert a sequence 'str' of length 'word_len' into index in 0~4^(word_len)-1
   '''
   output_index = 0
   for i in xrange(word_len):
      output_index = output_index * bases_len + base_dict[str[i]]
   return output_index

def shuffle(trainX,trainY):
   '''
   random shuffle the training data
   '''
   np.random.seed(0)
   train=np.hstack([trainY[:,None],trainX]).astype('float32')
   train=np.random.permutation(train)
   train=train[:120000,:]
   trainX=train[:,1:].astype('float32')
   trainY=train[:,0].astype('float32')
   return trainX,trainY

def leave_out(trainX,trainY):
   '''
   split the data into 70% for training and 30% for testing
   '''
   m=int(0.7*len(trainX))
   testX=trainX[m:,:]
   testY=trainY[m:]
   trainX=trainX[:m,:]
   trainY=trainY[:m]
   return trainX,trainY,testX,testY
   
def deepboost_train(trainX,trainY,beta,lamda,depth,type,round):
    '''
    call the train function in deepboost.so to traing the model
    '''
    trainX=trainX.astype('float32')
    trainY=trainY.astype('float32')
    r,c=trainX.shape
    _model=np.zeros((1,),dtype='int64')
    dll.train(ct.c_int(r),
              ct.c_int(c),
              trainX.ctypes.data_as(ct.c_voidp),
              trainY.ctypes.data_as(ct.c_voidp),
              ct.c_float(beta),
              ct.c_float(lamda),
              ct.c_int(depth),
              ct.c_int(type),
              ct.c_int(round),
              _model.ctypes.data_as(ct.c_voidp))
    return _model


def deepboost_classify(_model,testX):
    '''
    use the trained model '_model' to classify the samples in 'testX'
    '''
    testX=testX.astype('float32')
    r,c=testX.shape
    testY=np.zeros((r,),dtype='float32')
    dll.classify(ct.c_int(r),
                 ct.c_int(c),
                 testX.ctypes.data_as(ct.c_voidp),
                 testY.ctypes.data_as(ct.c_voidp),
                 _model.ctypes.data_as(ct.c_voidp))
    return testY

def delete_model(_model):
    '''
    delete the model '_model' to release memory
    '''
    dll.delete_model(_model.ctypes.data_as(ct.c_voidp))

def extract_features(line):
   '''
   extract features from a sequence of RNA
   '''
   #print line
   line2=line
   for i in 'agctu\n':
      line2 = line2.replace(i, '')
   line = line.upper().rstrip()
   line = line.replace('T', 'U')
   line2 = line2.replace('T','U')
   #line = line.replace('N','')
   #line2 = line2.replace('N','')
   final_output=[]
   for word_len in [1,2,3]:
      output_count_list = [0 for i in xrange(bases_len ** word_len)]
      for i in xrange(len(line) - word_len + 1):
         output_count_list[convert_to_index(line[i: i + word_len],word_len)] += 1
      output_count_list2 = [0 for i in xrange(bases_len ** word_len)]
      for i in xrange(len(line2)-word_len+1):
         output_count_list2[convert_to_index(line2[i:i+word_len],word_len)] +=1
      final_output.extend(output_count_list)
      final_output.extend(output_count_list2)
   for word_len in [4,5,6]:
      output_count_list = [0 for i in xrange(bases_len ** 2)]
      for i in xrange(len(line) - word_len):
         output_count_list[convert_to_index(line[i]+line[i + word_len],2)] += 1
      output_count_list2 = [0 for i in xrange(bases_len ** 2)]
      for i in xrange(len(line2)-word_len):
         output_count_list2[convert_to_index(line2[i]+line2[i+word_len],2)] +=1
      final_output.extend(output_count_list)
      final_output.extend(output_count_list2)
   final_output.append(len(line2))
   final_output.append(math.log(len(line2)))
   final_output.append(int(len(line2) % 3 == 0))
   stop_codons = ['UAG', 'UAA', 'UGA']
   stop_codon_features = [0,0,0,0]
   for stop_codon_num in xrange(len(stop_codons)):
      tmp_arr = [m.start() for m in re.finditer(stop_codons[stop_codon_num], line2)]
      tmp_arr_div3 = [i for i in tmp_arr if i % 3 == 0]
      stop_codon_features[stop_codon_num]=int(len(tmp_arr_div3) > 0)
      stop_codon_features[3]|=stop_codon_features[stop_codon_num]
   final_output.extend(stop_codon_features)
   num_feature = len(final_output)
   return final_output

def load_data(filename,check=False,savecheck='check'):
   '''
   use the extract_features function to extract features for all sequences in the file specified by 'filename'
   '''
   print 'Processing ',filename
   start=time.time()
   total_output=[]
   valid=[]
   for line in open(filename, "r"):
      if line[0] == '>':
         continue
      else:
         if ('n' in line or 'N' in line):
            valid.append(0)
            continue
         else:
            valid.append(1)
            total_output.append(extract_features(line.strip('\n').strip('\r')))
   output_arr=np.array(total_output)
   if (check):
      np.save(savecheck,np.array(valid))
   print output_arr.shape
   end=time.time()
   print 'Finished loading in',end-start,'s\n'
   return output_arr


def training(pathname,dataset):
   '''
   load dataset stored in the directory specified by 'pathname' and then train the model
   '''
   pos_filename=os.path.join(pathname,dataset+'.positives.fa')
   neg_filename=os.path.join(pathname,dataset+'.negatives.fa')
   pos_trainX=load_data(pos_filename)
   pos_trainY=np.ones(len(pos_trainX))
   neg_trainX=load_data(neg_filename)
   neg_trainY=-np.ones(len(neg_trainX))

   trainX=np.vstack([pos_trainX,neg_trainX])
   trainY=np.hstack([pos_trainY,neg_trainY])
   trainX,trainY=shuffle(trainX,trainY)
   #print trainX.shape,trainY.shape,'\n'
   #trainX,trainY,testX,testY=leave_out(trainX,trainY)

   print 'Start training...'
   start=time.time()
   scaler = preprocessing.StandardScaler().fit(trainX.astype(np.float))
   trainX = scaler.transform(trainX.astype(np.float))
   #testX = scaler.transform(testX.astype(np.float))
   _model=deepboost_train(trainX,trainY,beta,lamda,depth,type,round)
   #preds=deepboost_classify(_model,testX)
   #preds=(1+preds)/2
   #roc=roc_auc_score(testY, preds)

   end=time.time()
   print 'Training ends in',end-start,'s'
   #print 'Training roc:',roc,'\n'
   return scaler,_model

def predicting(filename,scaler,_model,savepred=None):
   '''
   predict for sequences in 'filename' using the preprocessing transform 'scaler' and the trained model '_model'
   '''
   print "Predicting",filename
   start=time.time()
   if savepred is not None:
      fout=open(savepred,'w')
   try:
      for line in open(filename):
         if line[0]=='>':
            if savepred is not None:
               fout.write(line)
            continue
         elif ('n' in line or 'N' in line):
            if savepred is not None:
               fout.write('Error!\n')
         else:
            line=line.strip('\n').strip('\r')
            testX=np.array(extract_features(line))[None,:]
            testX=scaler.transform(testX.astype(np.float))
            pred=deepboost_classify(_model,testX)
            if savepred is not None:
               fout.write('%f\n'%float(pred[0]))
   finally:
      if savepred is not None:
         fout.close()
      
def predicting2(filename,scaler,_model,savefile=None):
    '''
    predict for sequences in 'filename' using the preprocessing transform 'scaler' and the trained model '_model'
    '''   
    print "Predicting",filename
    start=time.time()
    testX=load_data(filename)
    testX=scaler.transform(testX.astype(np.float))
    pred=deepboost_classify(_model,testX)
    if savefile is not None:
        np.save(savefile,pred)
    end=time.time()
    print 'Predicting ends in',end-start,'s'
    return pred

def feature_score(_model,num_feature):
   '''
   extract the score for each features from the trained model '_model'
   '''
   score=np.zeros(num_feature,dtype='float64')
   dll.feature_score(_model.ctypes.data_as(ct.c_voidp),score.ctypes.data_as(ct.c_voidp))
   return score

def generate_motif(filename,savefile,score):
    '''
    for generating motifs of length n, 'filename' stores all RNA words of length n,
    'score' is the score of features returned by feature_score function,
    then the score for each word is evaluated and stored in 'savefile'
    ''' 
    fin=open(filename,'r')
    fout=open(savefile,'w')
    for line in fin:
        if line[0]=='>':
            fout.write(line)
        else:
            line=line.strip('\n').strip('\r')
            testX=extract_features(line)
            print >>fout,np.sum(testX*score)
    fin.close()
    fout.close()



if __name__=='__main__':
    sourcedir=sys.argv[1]
    dataset=sys.argv[2]
    testfile=sys.argv[3]
    savepred=sys.argv[4]

    scaler,_model=training(sourcedir,dataset)
    #prin "num_feature:",num_feature
    #score=feature_score(_model,num_feature)
    #generate_motif(testfile,savepred,score)
    predicting(testfile,scaler,_model,savepred)
    
