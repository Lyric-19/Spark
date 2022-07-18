import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql import Row


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Lab 2 Exercise") \
        .config("spark.local.dir","/fastdata/acq21rl") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

# add it to cache, so it can be used in the following steps efficiently    
logFile = spark.read.text("../Data/NASA_access_log_Jul95.gz").cache()    
logFile.show(5, False)


# split into 5 columns using regex and split
# For timestamp, only remain the date information, transform the Jul to number 07 and transform it to date type
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
              .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
              .withColumn('timestamp',F.split('timestamp',':',2).getItem(0).alias('date')) \
              .withColumn('timestamp',F.regexp_replace(F.col('timestamp'),"Jul","07")) \
              .withColumn('timestamp', F.to_date('timestamp', 'dd/MM/yyyy')) \
              .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
              .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
              .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
data.show(5,False)
print(data.count())

# group by date and transform the date to day of week
date = data.select('timestamp').groupBy('timestamp').count()
date = date.withColumn('day',F.dayofweek('timestamp')).dropna(how = 'any')
date.show()

# order the count under each day of week and extract the maximum and minimum one
day_request_max = date.withColumn('sortbyday', F.row_number().over(Window.partitionBy('day').orderBy(F.desc('count'))))
day_request_min = day_request_max.where(F.col('sortbyday')==4)
day_request_min = day_request_min.withColumnRenamed('count','min_count').drop('timestamp','sortbyday')
day_request_max = day_request_max.where(F.col('sortbyday')==1)
day_request_max = day_request_max.withColumnRenamed('count','max_count').drop('timestamp','sortbyday')
day_request = day_request_min.join(day_request_max,'day','inner')  

# to show the data as a table 
# make a new row name from number to literature
dayOfWeek = [(1,'Sunday'),(2,'Monday'),(3,'Tuesday'),(4,'Wednesday'),(5,'Thursday'),(6,'Friday'),(7,'Saturday')]
dayofweek = spark.createDataFrame(dayOfWeek,schema = ['Id','dayofweek'])
dayofweek.show()
day_col = dayofweek.select('dayofweek').collect()
day_request = day_request.join(dayofweek,day_request.day == dayofweek.Id,"inner")
day_request = day_request.select('dayofweek','min_count','max_count')

print("==================== Question 1 ====================")
day_request.show(7,False)
day_request = day_request.toPandas()

############## visualise the data #######################

day = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']    
x1 = list(range(7))
plt.bar(x1,day_request['min_count'],label = 'min',width = 0.4, fc = 'orange')    # min count data
x2 = [x + 0.4 for x in x1]
plt.bar(x2,day_request['max_count'],label = 'max',width = 0.4, tick_label = day ,fc = 'purple')  # max count data
plt.title('max and min request of each day of a week')
plt.xlabel('day of week')
plt.ylabel('count')
plt.legend()
plt.savefig("../Output/max_min_request_of_day.png")

print("===================Question 2=====================")
# filter the request which contains .mpg video and extract the video name using re, then order by the count of each video
# save the most 12 request
mpg_most_request = data.filter(data.request.contains(".mpg")).withColumn('request', F.regexp_extract('request', '.*\/(.*\.mpg).*',1)).groupBy('request').count().sort('count', ascending=False).limit(12).withColumnRenamed('request','videos')
mpg_most_request.show(12,False)

# filter the request which contains .mpg video and extract the video name using re, then order by the count of each video
# save the less 12 request
mpg_less_request = data.filter(data.request.contains(".mpg")).withColumn('request', F.regexp_extract('request', '.*\/(.*\.mpg).*',1)).groupBy('request').count().sort('count', ascending=True).limit(12).withColumnRenamed('request','videos')
mpg_less_request.show(12,False)

mpg_less_request = mpg_less_request.toPandas()
mpg_most_request = mpg_most_request.toPandas()



############## visualise the data #######################

plt.figure(figsize=(16,8))

x1 = list(range(12))

# deal with the y value to make the picture more readable
y1 = -1*np.array(mpg_most_request['count'])
y2 = 1000*np.array(mpg_less_request['count'])

plt.barh(x1,y1,color = 'orange',label = 'most require')
plt.barh(x1,y2 ,color = 'limegreen',label = 'less require')

# make a label to each bar
for x,y,z in zip(x1,y1,mpg_most_request['videos']):
  plt.text(y-1000,x,z,fontsize=12)
for x,y,z in zip(x1,y2,mpg_less_request['videos']):
  plt.text(y,x,z,fontsize=12)

# combine two bar in one x axis
plt.xticks((-4000,-3000,-2000,-1000,0,1000,2000,3000,4000),('4000','3000','2000','1000','0','1','2','3','4'))
plt.yticks(x1,('No.1','No.2','No.3','No.4','No.5','No.6','No.7','No.8','No.9','No.10','No.11','No.12'))
plt.title('Most and Less Require of Videos',fontsize = 16)
plt.xlabel('Count',fontsize=12)
plt.legend(loc = 'upper left')
plt.savefig("../Output/Most_and_Less_Require_of_Videos.png")


