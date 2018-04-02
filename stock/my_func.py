import csv
import numpy as np
def readcsv(f1='鴻海.csv',f2='蘋果電腦.csv'):
    f = open(f1, 'r')
    hon_str = []
    for row in csv.reader(f):
        #print (row)
        hon_str.append(row)
    f.close()
    
    f = open(f2, 'r')
    apple_str = []
    for row in csv.reader(f):
        #print (row)
        apple_str.append(row)
    f.close()
    
    two_list = []
    date = []
    for i in range (1,len(hon_str)):
        for j in range (len(apple_str)):
            if hon_str[i][0]==apple_str[j][0]:
                two_list.append([apple_str[j],hon_str[i]])
                date.append(hon_str[i][0])
                break
    apple = np.zeros((len(two_list),5))
    hon = np.zeros((len(two_list),5))
    
    for i in range (6171):
        apple[i] = [two_list[i][0][1] ,two_list[i][0][2],two_list[i][0][3],two_list[i][0][4],two_list[i][0][5][:-1]]
        hon[i] = [two_list[i][1][1] ,two_list[i][1][2],two_list[i][1][3],two_list[i][1][4],two_list[i][1][5]]
    
    return date,hon,apple

def makedata(hon_hai,apple,date_num=30):
    x = np.zeros((hon_hai.shape[0]-(date_num),date_num*4))
    y = np.zeros((hon_hai.shape[0]-(date_num),2))
    for i in range (0,hon_hai.shape[0]-(date_num)):
        x[i,0:date_num]=hon_hai[i:i+date_num,3]
        x[i,date_num:date_num*2]=hon_hai[i:i+date_num,4]
        x[i,date_num*2:date_num*3]=apple[i:i+date_num,3]
        x[i,date_num*3:date_num*4]=apple[i:i+date_num,4]
        y[i,0]=hon_hai[i+date_num,3]-hon_hai[i+date_num-1,3]
        y[i,1]=apple[i+date_num,3]-apple[i+date_num-1,3]
    return x,y
        
        