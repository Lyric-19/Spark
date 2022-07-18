from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .master("local[6]") \
        .appName("Q2_code") \
        .config("spark.local.dir","/fastdata/acq21rl") \
        .getOrCreate()
        

sc = spark.sparkContext
sc.setLogLevel("WARN")

import pyspark.sql.functions as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyspark.sql.functions import monotonically_increasing_id

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors


# Q2-A
# read the data
ratings = spark.read.load('../Data/ml-25m/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
ratings.show(10,False)
myseed = 6384

# split
splits = ratings.randomSplit([0.2,0.2,0.2,0.2,0.2], 6384)

# define ALS version 1
als1 = ALS(userCol = "userId", rank = 15, itemCol = "movieId", seed = myseed, coldStartStrategy = "drop")
als2 = ALS(userCol = "userId", rank = 17, itemCol = "movieId", blockSize =128 , seed = myseed, coldStartStrategy = "drop")

# define evaluator
evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")


def run_model(_train, _test, _als, _evaluator,i):
    
    # sort the count of each users, find the most 10% users and the least 10% users
    users = _train.select('userId').groupBy('userId').count().sort('count', ascending=False)
    users = users.withColumn('id',monotonically_increasing_id())
    user_n = users.count()
    HotUsers = users.limit(int(user_n/10))
    HotUsers = HotUsers.select('userId').rdd.flatMap(lambda x:x).collect()
    H_train = _train.filter(_train.userId.isin(HotUsers))
    CoolUsers = users.filter(users.id>=int(9*user_n/10))
    CoolUsers = CoolUsers.select('userId').rdd.flatMap(lambda x:x).collect()
    C_train = _train.filter(_train.userId.isin(CoolUsers))
    
    
    model_h = _als.fit(H_train)
    predictions_h = model_h.transform(_test)
    rmse_h = _evaluator.evaluate(predictions_h)
    print(f"No.{i} HotUsers Root-mean-square error = {rmse_h}")
    
    model_c = _als.fit(C_train)
    predictions_c = model_c.transform(_test)
    rmse_c = _evaluator.evaluate(predictions_c)
    print(f"No.{i} CoolUsers Root-mean-square error = {rmse_c}")
    
    return rmse_h,rmse_c


rmse_h = [[0]*5]*2
rmse_c = [[0]*5]*2
for i in range(5):
    test = splits[i]
    train = spark.createDataFrame(spark.sparkContext.emptyRDD(),splits[0].schema)
    for j in range(5):
        if not j==i:
            train = train.union(splits[j])
    rmse_h[0][i],rmse_c[0][i] = run_model(train,test,als1,evaluator,i)
    rmse_h[1][i],rmse_c[1][i] = run_model(train,test,als2,evaluator,i)

x1 = list(range(5))
label = ['split_1','split_2','split_3','split_4','split_5']
plt.title('max and min request of each day of a week')
plt.figure(figsize = (16,8))
plt.subplot(1,2,1)
plt.bar(x1,rmse_h[0],label = 'v1_HotUsers',width = 0.2, tick_label = label, fc = 'orange')
x2 = [x + 0.25 for x in x1]
plt.bar(x2,rmse_c[0],label = 'v1_CoolUsers',width = 0.2, fc = 'purple')
x3 = [x + 0.5 for x in x1]
plt.bar(x3,rmse_h[1],label = 'v2_HotUsers',width = 0.2, fc = 'red')
x4 = [x + 0.75 for x in x1]
plt.bar(x4,rmse_c[1],label = 'v2_CoolUsers',width = 0.2, fc = 'darkblue')
plt.xlabel('Split Number')
plt.ylabel('RMSE')
plt.legend()
plt.savefig("../Output/RMSE for different versions of ALS.png")

# Q2-B

print("********* show the table of tags **************")
tags = spark.read.load('../Data/ml-25m/tags.csv', format = 'csv', inferSchema = "true", header = "true").cache()
tags.show(10,False)

splits = ratings.randomSplit([0.2,0.2,0.2,0.2,0.2], 6384)
splits[0] = splits[0].cache()
splits[1] = splits[1].cache()
splits[2] = splits[2].cache()
splits[3] = splits[3].cache()
splits[4] = splits[4].cache()

als = ALS(userCol = "userId",itemCol = "movieId", seed = myseed, coldStartStrategy = "drop")

evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")

def extract_tag(data):
    model = als.fit(data)
    movie_fac = model.itemFactors
    movie_fac = movie_fac.cache()
    
    kmeans = KMeans().setK(10).setSeed(123)
    model = kmeans.fit(movie_fac) 
    
    prediction = model.transform(movie_fac)  
    #prediction.show(5,False)
    class_num = prediction.groupBy('prediction').count().sort('count', ascending=False)
    #class_num.show(10,False)
    
    select_class = class_num.select('prediction').rdd.flatMap(lambda x:x).collect()[:2]
    print("the largest class in this split is No.",select_class[0])
    print("the second large class in this split is No.",select_class[1])
    
    class1 = prediction.filter(F.col('prediction')==select_class[0])
    class2 = prediction.filter(F.col('prediction')==select_class[1])
    #class1.show(10,False)
    
    class1 = class1.select('id').rdd.flatMap(lambda x:x).collect()
    class2 = class2.select('id').rdd.flatMap(lambda x:x).collect()
    
    class1_tags = tags.filter(tags.movieId.isin(class1))
    class2_tags = tags.filter(tags.movieId.isin(class2))
    
    class1_tags = class1_tags.groupBy('tag').count().sort('count',ascending = False)
    class2_tags = class2_tags.groupBy('tag').count().sort('count',ascending = False)
    
    class1_tag_sort = class1_tags.select('tag').rdd.flatMap(lambda x:x).collect()
    class1_top_tag = class1_tag_sort[0]
    class1_bottom_tag = class1_tag_sort[-1]
    
    class2_tag_sort = class2_tags.select('tag').rdd.flatMap(lambda x:x).collect()
    class2_top_tag = class2_tag_sort[0]
    class2_bottom_tag = class2_tag_sort[-1]
    print("the top tag in the largest class is ",class1_top_tag)
    print("the bottom tag in the largest class is ",class1_bottom_tag)
    print("the top tag in the second large class is ",class2_top_tag)
    print("the bottom tag in the second large class is ",class2_bottom_tag)
    
    return class1_top_tag,class1_bottom_tag,class2_top_tag,class2_bottom_tag

class1_top_tag = [0]*5
class1_bottom_tag = [0]*5
class2_top_tag = [0]*5
class2_bottom_tag = [0]*5

for i in range(5):
    class1_top_tag[i],class1_bottom_tag[i],class2_top_tag[i],class2_bottom_tag[i] = extract_tag(splits[i])
    
print("********* the table of top tag and bottom tag in 5 splits ***********")
column_name = ['split_1','split_2','split_3','split_4','split_5']
class_tag = spark.createDataFrame([class1_top_tag,class1_bottom_tag,class2_top_tag,class2_bottom_tag],column_name)
class_tag.show()

