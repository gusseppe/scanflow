import pandas as pd
import logging

from sklearn.datasets import make_classification

logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


def generate_data(path, file_system='local', **args):
    """
        n_samples=100,n_features=4,
        class_sep=1.0, n_informative=2,
        n_redundant=2, random_state=rs
    """

    X, y = make_classification(**args)

    
    df = pd.DataFrame(X, columns=['x_'+str(i+1) for i in range(X.shape[1])])
    df = pd.concat([df, pd.DataFrame({'y':y})], axis=1)
    
    if file_system =='local':
        df.to_csv(path, index=False)
        print(df.head())
        logging.info(f'Dataset was generated successfully and saved in {path} ')
    
    elif file_system =='hdfs':
        from pyspark.sql import SQLContext, Row, SparkSession
        
        cluster_manager = 'yarn'
        spark = SparkSession.builder\
        .master(cluster_manager)\
        .appName("myapp")\
        .config("spark.driver.allowMultipleContexts", "true")\
        .getOrCreate()
        
        spark_df = spark.createDataFrame(df)
        spark_df.show(5)
        spark_df.limit(10000).write.mode('overwrite').parquet(path)
        logging.info(f'Dataset was generated successfully and saved in hdfs://{path} ')
    
        
generate_data(path='./raw_data.csv', file_system='local',
                           n_samples=10000,n_features=4,
                            class_sep=1.0, n_informative=2,
                            n_redundant=2, random_state=1)
