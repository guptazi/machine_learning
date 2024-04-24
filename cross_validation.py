from sklearn.model_selection import cross_val_score

def cvDictGen(functions,scr,X_train=X,y_train=y,cv=5,verbose=1):
    cvDict={}
    for func in functions:
        cvScore=cross_val_score(func,X_train,y_train,cv=cv,verbose=verbose,scoring=scr)
        cvDIct[str(func).split('(')[0]]=[cvScore.mean(),cvScore.std()]
    
    return cvDict

def cvDictNormalize(cvDict):
    cvDictNormalized={}
    for key in cvDict.keys():
        for i in cvDict[key]:
            cvDictNormalized[key]=['{0.2f}'.format(cvDict[key][0]/cvDict[cvDict.keys()[0]][0]),
            '{0.2f}'.format((cvDict[key[1]/cvDict[cvDict.keys()[0]][1]]))]

return cvDictNormalize