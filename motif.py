import os
import sys
from deepboost import *

sourcedir=sys.argv[1]
dataset=sys.argv[2]
testfile=sys.argv[3]
outputdir=sys.argv[4]
#testfile="dict_motif.fa"
savepred="tmp.fa"
scaler,_model=training(sourcedir,dataset)

score=feature_score(_model,num_feature)
generate_motif(testfile,savepred,score)
predicting(testfile,scaler,_model,savepred)

fin=open("tmp.fa",'r')
fout=open(os.path.join(outputdir,dataset+".motif.fa"),'w')

results=[]
while (True):
	try:
		word=fin.readline().strip('\n').split(' ')[1]
		score=float(fin.readline().strip('\n'))
		results.append([word,score])
	except:
		break

results=sorted(results,key=lambda d:d[1],reverse=True)

cnt=0
for word,score in results:
	print >>fout,'>\n',word
	cnt+=1
	if (cnt>=100):
		break

fin.close()
fout.close()
os.remove(savepred)