import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

pathlist16=['./mbti-eval/16/balancedtrain_classification_cp5000.csv','./mbti-eval/16/balancedtrain_classification_cp17500.csv',
'./mbti-eval/16/baltrain_classification_baleval_cp5500.csv','./mbti-eval/16/initialtrain_classification_old_cp5000.csv',
'./mbti-eval/16/initialtrain_classification_old_cp20000.csv','./mbti-eval/16/initialtrain2_classification_new_cp5334.csv',
'./mbti-eval/16/initialtrain2_classification_new_cp17780.csv','./mbti-eval/16/intialtrain2_classification_baleval_cp17780.csv']

pathlist8=['./mbti-eval/8/first2train_bal-classification_cp8890.csv', './mbti-eval/8/first2train_classification_cp8890.csv', 
'./mbti-eval/8/primtrain_classification_cp8890.csv']

pathlist2=['./mbti-eval/2/IE_classification_2clas_cp8000.csv', './mbti-eval/2/IE_classification_cp32000.csv', 
'./mbti-eval/2/IE_classification_cp44800.csv', './mbti-eval/2/IE_classification_MBTIonly_cp29095.csv', 
'./mbti-eval/2/PJ_classification_cp49784.csv', './mbti-eval/2/SN_classification_cp29095.csv', 
'./mbti-eval/2/TF_classification_cp40733.csv']

for path in pathlist16:
	df = pd.read_csv(path)
	true=df['True'].to_list()
	predicted=df['Predicted'].to_list()
	cross=pd.crosstab([true],[predicted],dropna=False,normalize='index',rownames=['True'],colnames=['Predictions'])
	print(path)
	plt.figure(figsize=(15,15))
	sn.heatmap(cross, annot=True)
	plt.show()
	plt.savefig('cross_tabulation.heatmap.png')#, bbox_inches='tight')
	print(classification_report(true,predicted, zero_division=0.0))
"""

#binary classification of labels
for path in pathlist16:
	df = pd.read_csv(path)
	true=[true[3] for true in df['True'].to_list()] # first part of label I/E
	predicted=[pred[3] for pred in df['Predicted'].to_list()] # first part of label I/E
	cross=pd.crosstab([true],[predicted],dropna=False,normalize='index',rownames=['True'],colnames=['Predictions'])
	print(path)
	plt.figure(figsize=(15,15))
	sn.heatmap(cross, annot=True)
	plt.show()
	print(classification_report(true,predicted, zero_division=0.0))
	#plt.savefig('cross_tabulation.heatmap.png', bbox_inches='tight')


for path in pathlist8:
	df = pd.read_csv(path)
	true=df['True'].to_list()
	predicted=df['Predicted'].to_list()
	cross=pd.crosstab([true],[predicted],dropna=False,normalize='index',rownames=['True'],colnames=['Predictions'])
	print(path)
	plt.figure(figsize=(15,15))
	sn.heatmap(cross, annot=True)
	plt.show()
	print(classification_report(true,predicted, zero_division=0.0))

for path in pathlist2:
	df = pd.read_csv(path)
	true=df['True'].to_list()
	predicted=df['Predicted'].to_list()
	cross=pd.crosstab([true],[predicted],dropna=False,normalize='index',rownames=['True'],colnames=['Predictions'])
	print(path)
	plt.figure(figsize=(15,15))
	sn.heatmap(cross, annot=True)
	plt.show()
	print(classification_report(true,predicted, zero_division=0.0))
"""