import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import os 


df = pd.read_csv(f'other/stats.csv')
# print(df.head())
classes = df.columns[1:].to_list()
# print(classes)

total = df.iloc[:,0].to_list()
total = np.array(total)
# finding index of total
total = np.where(total == 'total')[0][0]
total = df.iloc[total,1:].to_list()
# removing the HER3 column
total = total[:3]+total[4:]
classes = classes[:3]+classes[4:]
# print(total)


plt.figure(figsize=(5,5))
plt.axvspan(-1000,1000,facecolor='#fefbf5')
plt.bar(classes,total,color='#145da0',edgecolor='black',linewidth=0.5)
plt.xlabel('Classes',fontsize=16,fontweight='bold')
plt.ylabel('Number of Images',fontsize=16,fontweight='bold')
plt.title('Number of Images per subtype',fontsize=16,fontweight='bold')
# addinv value above the bar
for i in range(len(classes)):
    plt.text(i,total[i],total[i],ha='center',va='bottom',fontweight='bold',fontsize=14)


plt.xlim(-0.5,len(classes)-0.5)
plt.ylim(0,1000)
plt.xticks(range(len(classes)),[x.upper() for x in classes],fontsize=14,fontweight='bold')
plt.yticks(fontsize=14,fontweight='bold')

plt.tight_layout()
# plt.show()
plt.savefig('other/number_per_subtype.png',dpi=300, transparent=True)
plt.close()

# taking only the tnbc column
tnbc = df.iloc[:,1].to_list()
metrics = df['metrics'].to_list()   

surgery_index = metrics.index('surgery')
biopsy_index = metrics.index('biopsy')
metrics = metrics[1:11]
tnbc = tnbc[1:11]
ck = tnbc[4]+tnbc[5]+tnbc[6]

tnbc = tnbc[:4]+[ck]+tnbc[7:]
ki = tnbc[5]+tnbc[6]
print(ki)
tnbc = tnbc[:5]+[ki]+tnbc[7:]
print(tnbc)
metrics = ['yap', 'vim', 'hne', 'cd31', 'ck5/6', 'ki67', 'egfr']
print(metrics)

plt.figure(figsize=(5,5))
plt.axvspan(-1000,1000,facecolor='#fefbf5')
plt.bar(metrics,tnbc,color='#145da0',edgecolor='black',linewidth=0.5)
plt.xlabel('Stain',fontsize=16,fontweight='bold')
plt.ylabel('Number of Images',fontsize=16,fontweight='bold')
plt.title('No. of Images per Stain for \nTNBC',fontsize=16,fontweight='bold')
# addinv value above the bar
for i in range(len(metrics)):
    plt.text(i,tnbc[i],tnbc[i],ha='center',va='bottom',fontweight='bold',fontsize=14)
plt.xlim(-0.5,len(metrics)-0.5)
plt.ylim(0,110)
plt.xticks(range(len(metrics)),[x.upper() for x in metrics],fontsize=14,fontweight='bold',rotation=45,ha='right')
plt.yticks(fontsize=14,fontweight='bold')
# background color

plt.tight_layout()

# plt.show()
plt.savefig('other/number_per_stain_tnbc.png',dpi=300, transparent=True)
plt.close()