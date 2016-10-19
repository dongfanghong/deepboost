'''
adapt the original deepboost classification model into a regression model
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from deepboost import training,predicting2,leave_out
from sklearn import preprocessing
from scipy import stats
import gaussianize #preprocessing to transform the distribution of target value into a Gaussian distribution
from multiprocessing import Process,Queue,Pool

def read_RNAcompete_data(filename):
    name_list=[]
    score_list=[]
    for line in open(filename):
        tmp=line.strip('\n').strip('\r').split(' ')
        name_list.append(tmp[0])
        score_list.append(tmp[2])
    name_list=np.array(name_list)
    score_list=np.array(score_list,dtype='float')
    print  name_list.shape,score_list.shape
    return  name_list,score_list[:,None]

def read_RNAcompete_data_2(filename):
    name_list=[]
    score_list=[]
    for line in open(filename):
        tmp=line.strip('\n').strip('\r').split(' ')
        name_list.append(tmp[0])
        score_list.append(tmp[-1].replace(' ',''))
    name_list=np.array(name_list)
    score_list=np.array(score_list,dtype='float')
    print  name_list.shape,score_list.shape
    return  name_list,score_list[:,None]

def shuffle_RNAcompete(score_list,name_list):
    np.random.seed(0)
    tmp=np.hstack([name_list[:,None],score_list])
    np.random.permutation(tmp)[:10000,:]
    score_list=tmp[:,1:].astype('float')
    name_list=tmp[:,0]
    return score_list,name_list

def generate_RNAcompete_training_data(name_list,score_list,filename_positives,filename_negatives):
    np.random.seed(0)
    fout_positives=open(filename_positives,'w')
    fout_negatives=open(filename_negatives,'w')
    for i in xrange(len(name_list)):
        score=score_list[i]
        prob=np.exp(2*score)/(1+np.exp(2*score))
        for j in xrange(1):
            rnd=np.random.rand()
            if rnd<prob:
                print >>fout_positives,'>',name_list[i]
                print >>fout_positives,name_list[i]
            else:
                print >>fout_negatives,'>',name_list[i]
                print >>fout_negatives,name_list[i]
    fout_positives.close()
    fout_negatives.close()
        
if __name__=='__main__':
    filename_train=sys.argv[1]
    filename_test=sys.argv[2]
    output_pathname=sys.argv[3]
    filename_save=sys.argv[4]
    vmin,vmax=-2,2

    def single_process(filename_train,filename_test,filename_save):
        #filename=filename_list[idx]
        #print filename
        name_list_train,score_list_train=read_RNAcompete_data_2(filename_train)
        score_list_train,name_list_train=shuffle_RNAcompete(score_list_train,name_list_train)

        name_list_test,score_list_test=read_RNAcompete_data_2(filename_test)
        score_list_test,name_list_test=shuffle_RNAcompete(score_list_test,name_list_test)
        #score_list_train,name_list_train,score_list_test,name_list_test=leave_out(score_list,name_list)

        preprocess_scaler=gaussianize.Gaussianize(strategy='brute')
        preprocess_scaler.fit(score_list_train)
        score_list_train=preprocess_scaler.transform(score_list_train)
        score_list_train/=2.5
        '''
        plt.hist(score_list_train)#,plt.show()
        plt.savefig('input_'+filename+'.png')
        plt.close()
        '''

        filename_positives=os.path.join(output_pathname,'tmp_'+filename_save+'.positives.fa')
        filename_negatives=os.path.join(output_pathname,'tmp_'+filename_save+'.negatives.fa')
        #print filename_positives,filename_negatives
        generate_RNAcompete_training_data(name_list_train,score_list_train,filename_positives,filename_negatives)
        
        filename_test_tmp=os.path.join(output_pathname,'tmp_'+filename_save+'.test.fa')
        fout_test=open(filename_test_tmp,'w')
        for name in name_list_test:
            print >>fout_test,'>',name
            print >>fout_test,name
        fout_test.close()
          
        scaler,_model=training(output_pathname,'tmp_'+filename_save.replace('.txt',''))
        pred=predicting2(filename_test_tmp,scaler,_model,None)

        #pred=preprocess_scaler.invert(pred)
        #print pred-score_list_test

        score_list_test=preprocess_scaler.transform(score_list_test)
        score_list_test/=2.5
        score_list_test=score_list_test.ravel()

        #print 'ave:',np.abs(pred-score_list_test).mean()
        coef=stats.pearsonr(pred,score_list_test)
        print filename_test,'correlation:',coef
        plt.scatter(score_list_test,pred),plt.plot([vmin,vmax],[vmin,vmax],'r'),plt.xlim(vmin,vmax),plt.ylim(vmin,vmax)#,plt.show()
        plt.savefig(os.path.join(output_pathname,filename_save+'.png'))
        plt.close()

        #save = np.array([pred,score_list_test])
        filename_save=os.path.join(output_pathname,filename_save+'.pred')
        fout_save = open(filename_save,'w')
        for i in xrange(len(name_list_test)):
            print >>fout_save,name_list_test[i],pred[i]
        fout_save.close()

        #print filename_positives
        #print filename_test
        os.remove(filename_positives)
        os.remove(filename_negatives)
        os.remove(filename_test_tmp)

        return coef

    single_process(filename_train,filename_test,filename_save)

    
        