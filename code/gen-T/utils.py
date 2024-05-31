import numpy as np

def loadListFromTxtFile(listPath): 
    ''' Save list as a text file
    Args:
        listPath: filepath with the stored list
    '''
    savedList = []
    with open(listPath) as f:
        listItems = f.read().splitlines()
        for i in listItems:
            savedList.append(i)
    return savedList
    
def saveListAsTxtFile(listToSave, listPath): 
    ''' Save list as a text file
    Args:
        listToSave to be saved
        listPath: filepath to which the list will be saved
    '''
    with open(listPath, "w") as output:
        for item in listToSave:
            output.writelines(str(item)+'\n')
            
def projectAtts(tableDfs, queryTable):
    ''' For each table, project out the attributes that are in the query table.
        If the table has no shared attributes with the query table, remove
    '''
    projectedDfs = {}
    queryCols = queryTable.columns
    for table, df in tableDfs.items():
        tableCols = df.columns
        projectedTable = df.drop(columns=[c for c in tableCols if c not in queryCols])
        if not projectedTable.empty:
            projectedDfs[table] = projectedTable
    return projectedDfs

def selectKeys(tableDfs, queryTable, primaryKey, foreignKeys):
    ''' For each table, select tuples that contain key value from queryTable
        If the table has no shared keys with the queryTable, remove
    '''    
    selectedDfs = {}
    queryKeyVals = {}
    for key in queryTable.columns:
        queryKeyVals[key] = queryTable[key].tolist()
    commonKey = primaryKey
    for table, df in tableDfs.items():
        dfCols = df.columns.tolist()
        commonKeys = [k for k in [primaryKey]+foreignKeys if k in df.columns]
        if not commonKeys: continue
        
        if len(commonKeys) > 1: 
            conditions = [df[commonKeys[0]].isin(queryKeyVals[commonKeys[0]]).values]
            for commonKey in commonKeys[1:]:
                conditions.append([df[commonKey].isin(queryKeyVals[commonKey]).values])                
            try:
                selectedTuplesDf = df.loc[np.bitwise_and.reduce(conditions)[0]].drop_duplicates().reset_index(drop=True)
            except:
                print(f"Couldn't select tuples in table {table}")
                continue
        elif len(commonKeys) == 1:
            selectedTuplesDf = df[df[commonKey].isin(queryKeyVals[commonKey])]
        if not selectedTuplesDf.empty:
            selectedDfs[table] = selectedTuplesDf
                
    return selectedDfs