import sys
import pandas as pd
import numpy as np
import time
import integration_utils as utils
sys.path.append('../')
from evaluatePaths import bestMatchingTuples, valueSimilarity
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)    
pd.options.mode.chained_assignment = None

class TableIntegration:
    def __init__(self, benchmark, originating_tables, timeout):
        self.benchmark = benchmark
        self.originating_tables = originating_tables # dict {originating table: [matched cols]}
        self.raw_tableDfs = {}
        self.originating_tableDfs = {} # dict {tableName: DataFrame}
        self.algStartTime = None
        self.timeout = timeout
        
        self.source_df = pd.DataFrame()
        self.primary_key = None
        self.foreign_keys = []
        self.tableOpsUsed = []
        self.replaceNull = '*NAN*'
        self.reproducedSourceTable = pd.DataFrame()
        self.expandedSourceTable = pd.DataFrame()


    def MoreEfficientComplementation(self, table):
        partitioned_tuple_list, debug_dict = utils.FineGrainPartitionTuples(table, self.timeout, self.algStartTime)
        if not partitioned_tuple_list: return None, None, None, None, None
        complemented_list = set()
        max_partition_size = 0
        for current_partition_tuples in partitioned_tuple_list:
            current_size = len(current_partition_tuples)
            if current_size > max_partition_size:
                max_partition_size = current_size
            complemented_tuples = utils.ComplementAlgorithm(current_partition_tuples, self.timeout, self.algStartTime)
            if not complemented_tuples: return None, None, None, None, None
            for item in complemented_tuples:
                complemented_list.add(item)
            if (time.time() - self.algStartTime) > self.timeout: return None, None, None, None, None
            
        return complemented_list, len(partitioned_tuple_list), max_partition_size, "full", debug_dict


    def EfficientSubsumption(self, tuple_list):
        #start_time = time.time_ns()
        subsumed_list = []
        m = len(tuple_list[0]) #number of columns
        bucket = dict()
        minimum_null_tuples = dict()
        bucketwise_null_count = dict()
        first_pattern, minimum_nulls = utils.FindCurrentNullPattern(tuple_list[0])
        bucket[first_pattern] = [tuple_list[0]]
        bucketwise_null_count[minimum_nulls] = {first_pattern}
        minimum_null_tuples[minimum_nulls] = [tuple_list[0]]
        for key in tuple_list[1:]:
            current_pattern, current_nulls = utils.FindCurrentNullPattern(key)
            if current_nulls not in bucketwise_null_count:
                bucketwise_null_count[current_nulls] = {current_pattern}
            else:
                bucketwise_null_count[current_nulls].add(current_pattern)
            if current_pattern not in bucket:
                bucket[current_pattern] = [key]
            else:
                bucket[current_pattern].append(key)
            if current_nulls < minimum_nulls:
                minimum_null_tuples[current_nulls] = [key]
                minimum_null_tuples.pop(minimum_nulls)
                minimum_nulls = current_nulls
            elif current_nulls == minimum_nulls:
                minimum_null_tuples[current_nulls].append(key)
            if (time.time() - self.algStartTime) > self.timeout: return None
            
        #output all tuples with k null values
        subsumed_list = minimum_null_tuples[minimum_nulls]
        for i in range(minimum_nulls+1, m):
            if i in bucketwise_null_count:
                related_buckets = bucketwise_null_count[i]
                parent_buckets = set()
                temp = [v for k,v in bucketwise_null_count.items()
                                        if int(k) < i]
                parent_buckets = set([item for sublist in temp for item in sublist])
                
                for each_bucket in related_buckets:
                    #do something
                    current_bucket_tuples = bucket[each_bucket]
                    if len(current_bucket_tuples) == 0:
                        continue
                    non_null_positions = utils.CheckNonNullPositions(each_bucket, m-i)
                    parent_bucket_tuples = set()
                    for each_parent_bucket in parent_buckets:
                        if utils.CheckAncestor(each_bucket, each_parent_bucket) == 1:
                            list_of_parent_tuples = bucket[each_parent_bucket]
                            for every_tuple in list_of_parent_tuples:
                                projected_parent_tuple = utils.GetProjectedTuple(
                                    every_tuple, non_null_positions, m)
                                parent_bucket_tuples.add(projected_parent_tuple)
                    new_bucket_item = []     
                    for each_tuple in current_bucket_tuples:
                        projected_child_tuple = set()
                        for j in range(0,m):
                            if j in non_null_positions:
                                projected_child_tuple.add(each_tuple[j])
                        projected_child_tuple = utils.GetProjectedTuple(
                                    each_tuple, non_null_positions, m)
                        
                        if projected_child_tuple not in parent_bucket_tuples:
                            new_bucket_item.append(each_tuple)
                            subsumed_list.append(each_tuple)
                    bucket[each_bucket] = new_bucket_item
            if (time.time() - self.algStartTime) > self.timeout: return None
        return subsumed_list

    def checkAccuracy(self, old_df, new_df):
        commonCols = [col for col in self.source_df.columns if col in old_df.columns]
        sourceTable = self.source_df[commonCols]
        oldDf = old_df[commonCols]
        newDf = new_df[commonCols]
        unary_operator_applied = True
        
        
        # check if applying complementation / subsumption increases Value Similarity Score 
        original_bestMatchingDf = bestMatchingTuples(sourceTable, oldDf, commonCols[0])
        if original_bestMatchingDf is None: return old_df, False
        original_valueSim = valueSimilarity(sourceTable, original_bestMatchingDf, commonCols[0])
        
        change_bestMatchingDf = bestMatchingTuples(sourceTable, newDf, commonCols[0])
        if change_bestMatchingDf is None: return old_df, False
        change_valueSim = valueSimilarity(sourceTable, change_bestMatchingDf, commonCols[0])
        
        if change_valueSim < original_valueSim:
            return old_df, False
        return new_df, unary_operator_applied

    def FDAlgorithm(self, doReproduce=1, saveTable=1):
        if doReproduce: table_dfs = self.originating_tableDfs
        else: table_dfs = self.raw_tableDfs
        self.algStartTime = time.time()
        m = len(table_dfs)
        null_count = 0
        null_set = set()
        table1Name = list(table_dfs.keys())[0]
        table1 = list(table_dfs.values())[0]
        table1 = table1.reset_index(drop=True)
        if table1.isnull().sum().sum() > 0:
            table1, null_count, current_null_set = utils.ReplaceNulls(table1, null_count)
            null_set = null_set.union(current_null_set)
        
        include_key_vals = [29, 35, 40, 24.0, 14.0, 40]
        
        
        # == BEGIN Outer union
        for tableName, files in table_dfs.items():
                    
            if tableName == table1Name: continue
            if table1.isnull().sum().sum() > 0:
                table1, null_count, current_null_set = utils.ReplaceNulls(table1, null_count)
                null_set = null_set.union(current_null_set)
                
            table2 = files.reset_index(drop=True)
            if table2.isnull().sum().sum() > 0:
                table2, null_count, current_null_set = utils.ReplaceNulls(table2, null_count)
                null_set = null_set.union(current_null_set)
            table1 = pd.concat([table1,table2])
            if (time.time() - self.algStartTime) > self.timeout: return None, None, None
            self.tableOpsUsed.append(f"outer_union:{table1Name},{tableName}")
            #measure time after preprocessing
            start_time = time.time_ns()
            s = table1.shape[0]
            total_cols = table1.shape[1]
            schema = list(table1.columns)
            start_complement_time = time.time_ns()
            complementationResults, complement_partitions, largest_partition_size, partitioning_used, debug_dict = self.MoreEfficientComplementation(table1)
            if not complementationResults and not complement_partitions and not largest_partition_size and not partitioning_used and not debug_dict: return None, None, None
            end_complement_time = time.time_ns()
            complement_time = int(end_complement_time - start_complement_time)/ 10**9
            
            fd_table = pd.DataFrame(complementationResults, columns =schema)

            if len(null_set) > 0:
                fd_table =  utils.AddNullsBack(fd_table, null_set)
            fd_table, unary_operator_applied = self.checkAccuracy(table1, fd_table)
            if unary_operator_applied: self.tableOpsUsed.append("complementation")
            
            old_fd_table = fd_table.copy()
            fd_data = {tuple(x) for x in fd_table.values}
            start_subsume_time = time.time_ns()
            subsumptionResults = self.EfficientSubsumption(list(fd_data))
            if not subsumptionResults: return None, None, None
            end_subsume_time = time.time_ns()
            fd_table = pd.DataFrame(subsumptionResults, columns =schema)

            fd_table, unary_operator_applied = self.checkAccuracy(old_fd_table, fd_table)
            if unary_operator_applied: self.tableOpsUsed.append("subsumption")
            
            table1 = fd_table
            fd_data = [tuple(x) for x in fd_table.values]
            end_time = time.time_ns()
            total_time = int(end_time - start_time)/10**9
            
        if saveTable: self.reproducedSourceTable = table1.drop_duplicates()
        return table1.drop_duplicates()

    def compSubsumInnerUnion(self, table1):
        # == Apply Comp and Subsump on FIRST TABLE
        self.algStartTime = time.time()
        tableOpsApplied = ''
        schema = list(table1.columns)
        complementationResults, complement_partitions, largest_partition_size, partitioning_used, debug_dict = self.MoreEfficientComplementation(table1)
        if not complementationResults and not complement_partitions and not largest_partition_size and not partitioning_used and not debug_dict: return None, None
        fd_table = pd.DataFrame(complementationResults, columns =schema)
                
        fd_table, unary_operator_applied = self.checkAccuracy(table1, fd_table)
        if unary_operator_applied: tableOpsApplied += 'complementation'
    
        old_fd_table = fd_table.copy()
        fd_data = {tuple(x) for x in fd_table.values}
        subsumptionResults = self.EfficientSubsumption(list(fd_data))
        if not subsumptionResults: return None, None
        subsumed_tuples = len(list(fd_data)) - len(subsumptionResults)
        fd_table = pd.DataFrame(subsumptionResults, columns =schema)
        fd_table, unary_operator_applied = self.checkAccuracy(old_fd_table, fd_table)
        if unary_operator_applied: tableOpsApplied += ',subsumption'
        return fd_table, tableOpsApplied
            
    def labelNulls(self):
        for table, df in self.originating_tableDfs.items():
            dfCols = df.columns
            # add empty rows to DF with repeat keys as dummy rows
            nullRows = []
            if self.primary_key in df.columns:
                for keyVal in df[self.primary_key]:
                    nRow = [keyVal]+[np.nan]*(df.shape[1]-1)
                    nullRows.append(nRow)
                commonForeignKeys = [col for col in self.foreign_keys if col in dfCols]
                if commonForeignKeys:
                    for keyVal in df[self.primary_key]:
                        nRows = df[df[self.primary_key]==keyVal].values.tolist()
                        for rIndx, row in enumerate(nRows):
                            for foreignKey in commonForeignKeys:
                                fKeyIndx = list(dfCols).index(foreignKey)
                                row[fKeyIndx] = np.nan
                            nullRows.append(row)
            
            dfRows = df.values.tolist()
            df = pd.DataFrame(dfRows+nullRows, columns=dfCols).drop_duplicates()
            lNullsDf = utils.labelNullsDf(self.source_df, df, [self.primary_key]+self.foreign_keys)
            if lNullsDf is not None: self.originating_tableDfs[table] = lNullsDf
        self.source_df = self.source_df.replace(np.nan, self.replaceNull)
        
    def selectProject(self):
        # ==== PROJECT / SELECT Source Table's Columns / Keys
        projectedTableDfs,project_ops = utils.projectAtts(self.originating_tableDfs, self.source_df)
        self.originating_tableDfs,select_ops = utils.selectKeys(projectedTableDfs, self.source_df, self.primary_key, self.foreign_keys)
        for table, df in self.originating_tableDfs.items():
            df.reset_index(drop=True, inplace=True)
        self.tableOpsUsed += project_ops + select_ops
    
    def loadCandidateTables(self, source_path, datalake_path):
        # Get source table, its primary key and foreign keys
        self.source_df = pd.read_csv(f"{source_path}")
        sourceTableName = source_path.split("/")[-1]
        self.primary_key = self.source_df.columns.tolist()[0]
        self.foreign_keys = []
        
        # Get datalake and originating tables from data lake
        dataLakePath = datalake_path+"/"
        for tableName in self.originating_tables:
            table = dataLakePath+tableName
            if tableName == sourceTableName: continue
            table_df = pd.read_csv(table)
            table_df = table_df.rename(columns=self.originating_tables[tableName])
            
            # check types
            for col in table_df.columns:
                if col in self.source_df:
                    try: table_df[col] = table_df[col].astype(self.source_df[col].dtypes.name)
                    except: 
                        table_df = table_df.dropna()
                        
                        try: table_df[col] = table_df[col].astype(self.source_df[col].dtypes.name)
                        except: table_df = table_df.drop(col, axis=1) # if cannot convert to same type, delete    
                        
            self.raw_tableDfs[tableName] = table_df
        self.originating_tableDfs = self.raw_tableDfs
        

    def integrate_tables(self, source_path, datalake_path):
        self.loadCandidateTables(source_path, datalake_path)
        self.selectProject()        
        noCandidates = False
        timed_out = False
        numOutputVals = 0
        if not self.originating_tableDfs: 
            noCandidates = True
            return timed_out, noCandidates, numOutputVals
        
        self.originating_tableDfs, innerUnionOp = utils.innerUnion(self.originating_tableDfs, self.primary_key, self.foreign_keys)
        self.tableOpsUsed += innerUnionOp
        self.labelNulls()
        
        for table, df in self.originating_tableDfs.items():
            fd_table, tableOps = self.compSubsumInnerUnion(df)
            self.originating_tableDfs[table] = fd_table
            self.tableOpsUsed.append(f"{tableOps}:{table}")
            
        result_FD = list(self.originating_tableDfs.values())[0]
        if result_FD is None: 
            noCandidates = True
            return timed_out, noCandidates, numOutputVals
        numOutputVals = result_FD.shape[0]* result_FD.shape[1]
        
        if len(self.originating_tableDfs) == 1: 
            self.reproducedSourceTable = list(self.originating_tableDfs.values())[0]
        elif len(self.originating_tableDfs) > 1:
            # Integrate using Full Disjunction
            self.FDAlgorithm()
            numOutputVals = self.reproducedSourceTable.shape[0]* self.reproducedSourceTable.shape[1]
        
        if self.reproducedSourceTable is None: 
            noCandidates = True
            timed_out = True
        #save result to hard drive
        commonCols = [col for col in self.source_df.columns if col in self.reproducedSourceTable.columns]
        if not commonCols: 
            noCandidates = True
            return timed_out, noCandidates, numOutputVals
        self.reproducedSourceTable = self.reproducedSourceTable[commonCols]
        
        self.source_df = self.source_df.replace(self.replaceNull, np.nan) 
        self.reproducedSourceTable = self.reproducedSourceTable.replace(self.replaceNull, np.nan)  
        self.reproducedSourceTable = self.reproducedSourceTable.dropna(axis=0, subset=[self.primary_key])
        if self.reproducedSourceTable.empty: 
            noCandidates = True
            return timed_out, noCandidates, numOutputVals
        self.reproducedSourceTable = self.complementationSubsumption(self.reproducedSourceTable)
        
        print("-----x---------x--------x---")
        print(f"Source Table has {self.source_df.shape[0]} rows, {self.source_df.shape[1]} columns")
        print("-----x---------x--------x---")
        print(f"Reclaimed Source Table has {self.reproducedSourceTable.shape[0]} rows, {self.reproducedSourceTable.shape[1]} columns")
        
        return timed_out, noCandidates, numOutputVals


    def complementationSubsumption(self, df):
        schema = list(df.columns)
        complementationResults, complement_partitions, largest_partition_size, partitioning_used, debug_dict = self.MoreEfficientComplementation(df)
        fd_table = pd.DataFrame(complementationResults, columns =schema)
        fd_data = {tuple(x) for x in fd_table.values}
        subsumptionResults = self.EfficientSubsumption(list(fd_data))
        fd_table = pd.DataFrame(subsumptionResults, columns =schema)
        return fd_table
    