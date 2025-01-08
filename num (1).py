import numpy as np

arr=np.array([2,3,4])
print(arr)
print(type(arr))

""" arr=np.array((10,20,30,40,50))
print(arr[2]+arr[3]) """

""" a = np.array(42)
b = np.array([1,2,3,4,5])
c = np.array([[1,2,3], [4,5,6]])
d = np.array([[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]])
print("a:",a)
print("b:",b)
print("c:",c)
print("d:",d)
print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim) """

""" a=np.array([[[4]]])
print(a.ndim) """

""" arr=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr[1,2:5]) """

""" arr=np.array([1,2,3,4,5])
x=arr.copy()
arr[0]=42
x[1]=10
print(arr)
print(x) """

""" arr=np.array([1,2,3,4,5])
x=arr.view()
arr[0]=42
x[1]=10
print(arr)  
print(x) """

#Shape- Give the shape of the array
""" arr = np.array([[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]]])
print(arr.shape)
#Output: (2, 2, 4) """


#ndim - we can change the dimenstional shape using ndim
""" arr = np.array([1,2,3,4], ndmin=4) 
print(arr)  #[[[[1 2 3 4]]]]
print('shape of array :', arr.shape)
#Output: (1, 1, 1, 4) """


#Reshape -reshape used to share the dimension , no.of elements in array should be equal to axb
""" arr=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
new_arr=arr.reshape(3,4)
print(new_arr) """
#Output
""" [[ 1  2  3  4]  
[ 5  6  7  8]
[ 9 10 11 12]] """


# Unknown dimension= reshape (-1) - (-1) automaticly calculate any one size
""" arr=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
new_arr=arr.reshape(1,3,-1)
print(new_arr) """
#Output
""" [[[ 1  2  3  4]
  [ 5  6  7  8]
  [ 9 10 11 12]]] """

# reshape(-1) 
""" arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
newarr= arr.reshape(-1)
print(newarr) """
#Output: [1 2 3 4 5 6 7 8 9]

#for loop
""" arr=np.array([1,2,3,4,5])
for x in arr:
    print(x) """

""" arr=np.array([[1,2,3],[4,5,6]])
for x in arr:
    print(x) """

""" arr=np.array([[1,2,3],[4,5,6]])
for x in arr:
    for y in x:
        print(y) """

""" arr=np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
for x in arr:
    for y in x:
        for z in y:
            print(z) """

# Instead of using the above code we can go for nditer()
#nditer() used instead of using many for loops
""" arr=np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
for x in np.nditer(arr):
    print(x) """

#concatenate
""" arr1=([1,2,3])
arr2=([4,5,6])
arr=np.concatenate((arr1,arr2))
print(arr) """

#stack()
""" arr1=([1,2,3])
arr2=([4,5,6])
arr=np.stack((arr1,arr2),axis=1)  # axis=0 # axis=1
print(arr) """

#hstack()
""" arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.hstack((arr1, arr2))
print(arr) """

#vstack
""" arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.vstack((arr1, arr2))
print(arr) """


#array_split() -  we can do any number of split in numpy
""" arr1=np.array([1,2,3,4,5,6])
arr=np.array_split(arr1,3) 
print(arr) """

""" arr = np.array([1,2,3,4,5,6,7])
newarr =np.array_split(arr,4)
print(newarr)
print(newarr[0])
print(newarr[1])
print(newarr[2]) """

#hsplit
""" arr1=np.array([[1,2,3],
               [4,5,6]])
new_arr=np.hsplit(arr1,3)
print(new_arr) """

#vsplit 
""" arr1=np.array([[1,2,3],[4,5,6]])
new_arr=np.vsplit(arr1,2)
print(new_arr) """

#where
""" arr=np.array([1,2,3,4,2,5,6,2,8])
x=np.where(arr==2)
print(x) """

""" arr=np.array([1,2,3,4,2,5,6,2,8])
x=np.where(arr%2==0)  #gives the index number where the condition true
print(x) """

#searchsorted - its shows index number where the number should come
""" arr=np.array([1,5,3,7])
x=np.searchsorted(arr,6) 
print(x) """

#sort
""" arr=np.array([1,5,3,7])
x=np.sort(arr) 
print(x) """

""" arr=np.array([[1,5,3,7],[5,0,1,4]])
x=np.sort(arr) 
print(x) """



""" a=np.array([[2,3,4],[6,7,8]])
# for x in np.nditer(a):
for x in np.nditer(a,flags=["buffered"],op_dtypes="S"):  #buffered is a storage-adding storage to store int to str
    print(x) """


#ndenumerate
""" a=np.array([[2,3,4],[6,7,8]])
for x in np.ndenumerate(a):
    print(x) """

""" a=np.array([[2,3,4],[6,7,8]])
for ix,x in np.ndenumerate(a):
    print(ix,x) """

#--------------------------------------------------------------------------------------------
#1
""" temperatures=np.array([22.1,23.4,21.8,24.0,25.2,22.5,21.7,20.9,
                           22.3,23.1,21.6,24.5,23.8,22.0,21.9,22.4,
                           23.7,24.1,22.8,21.5,20.7,22.6,23.0,24.3,
                           25.0,22.2,21.4,23.2,24.6,25.1,22.7,21.3,
                           20.8,23.5,24.2,25.3,22.9,21.2,22.0,23.6])
m1=np.mean(temperatures)
res=np.round(m1,2)
m2=np.median(temperatures)
print("Mean:",res)
print("Median",m2) """

""" rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
res1=np.std(rainfall)
res2=np.var(rainfall)
print(f"Standard deviation:{res1}")
print(f"Variance:{res2}") """

#2
""" temperatures=np.array([22.1,23.4,21.8,24.0,25.2,22.5,21.7,20.9,
                           22.3,23.1,21.6,24.5,23.8,22.0,21.9,22.4,
                           23.7,24.1,22.8,21.5,20.7,22.6,23.0,24.3,
                           25.0,22.2,21.4,23.2,24.6,25.1,22.7,21.3,
                           20.8,23.5,24.2,25.3,22.9,21.2,22.0,23.6])
mx=np.max(temperatures)
mn=np.min(temperatures)
print(f"Maximum temperature:{mx}")
print(f"Minimum temperature:{mn}") """

""" rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
res=np.where(rainfall>1)[0] #0 used to access form 1st element without 0 it returns tuple
print(res) """

#3
""" temperatures=np.array([22.1,23.4,21.8,24.0,25.2,22.5,21.7,20.9,
                           22.3,23.1,21.6,24.5,23.8,22.0,21.9,22.4,
                           23.7,24.1,22.8,21.5,20.7,22.6,23.0,24.3,
                           25.0,22.2,21.4,23.2,24.6,25.1,22.7,21.3,
                           20.8,23.5,24.2,25.3,22.9,21.2,22.0,23.6])
print(temperatures[:7]) """

""" rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
print(rainfall[-5:]) """

#4
""" temperatures=np.array([22.1,23.4,21.8,24.0,25.2,22.5,21.7,20.9,
                           22.3,23.1,21.6,24.5,23.8,22.0,21.9,22.4,
                           23.7,24.1,22.8,21.5,20.7,22.6,23.0,24.3,
                           25.0,22.2,21.4,23.2,24.6,25.1,22.7,21.3,
                           20.8,23.5,24.2,25.3,22.9,21.2,22.0,23.6])
rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
res=np.corrcoef(temperatures,rainfall)
print(res) """

""" temperatures=np.array([22.1,23.4,21.8,24.0,25.2,22.5,21.7,20.9,
                       22.3,23.1,21.6,24.5,23.8,22.0,21.9,22.4,
                       23.7,24.1,22.8,21.5,20.7,22.6,23.0,24.3,
                       25.0,22.2,21.4,23.2,24.6,25.1,22.7,21.3,
                       20.8,23.5,24.2,25.3,22.9,21.2,22.0,23.6])
rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
res1=np.where(temperatures>23)
res2=rainfall[res1]
res=np.average(res2)
print(res) """

#6
""" rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
total=np.sum(rainfall)
avrg=np.average(rainfall)
print(f"Total rainfall:{total}")
print(f"Average rainfall:{avrg}") """

""" temperatures=np.array([22.1,23.4,21.8,24.0,25.2,22.5,21.7,20.9,
                       22.3,23.1,21.6,24.5,23.8,22.0,21.9,22.4,
                       23.7,24.1,22.8,21.5,20.7,22.6,23.0,24.3,
                       25.0,22.2,21.4,23.2,24.6,25.1,22.7,21.3,
                       20.8,23.5,24.2,25.3,22.9,21.2,22.0,23.6])
rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
high_temp=np.argmax(temperatures)
res=temperatures[high_temp]
high_rain=rainfall[high_temp]
print(f"Highest temperature:{res}")
print(f"Rainfall on that day:{high_rain}") """

#7
""" temperatures=np.array([22.1,23.4,21.8,24.0,25.2,22.5,21.7,20.9,
                       22.3,23.1,21.6,24.5,23.8,22.0,21.9,22.4,
                       23.7,24.1,22.8,21.5,20.7,22.6,23.0,24.3,
                       25.0,22.2,21.4,23.2,24.6,25.1,22.7,21.3,
                       20.8,23.5,24.2,25.3,22.9,21.2,22.0,23.6])
new_temp=temperatures[:30]
new_arr=new_temp.reshape(5,6)
print(new_arr) """

""" temperatures=np.array([22.1,23.4,21.8,24.0,25.2,22.5,21.7,20.9,
                       22.3,23.1,21.6,24.5,23.8,22.0,21.9,22.4,
                       23.7,24.1,22.8,21.5,20.7,22.6,23.0,24.3,
                       25.0,22.2,21.4,23.2,24.6,25.1,22.7,21.3,
                       20.8,23.5,24.2,25.3,22.9,21.2,22.0,23.6])
new_temp=temperatures[:30]
new_arr=new_temp.reshape(5,6)
new_mean=new_arr.mean(axis=1)
result=np.round(new_mean,1)
print(result) """

#5
""" temperatures=np.array([22.1, 23.4, 21.8, 24.0, 25.2, 22.5, 21.7, 20.9,
                       22.3, 23.1, 21.6, 24.5, 23.8, 22.0, 21.9, 22.4,
                       23.7, 24.1, 22.8, 21.5, 20.7, 22.6, 23.0, 24.3,
                       25.0, 22.2, 21.4, 23.2, 24.6, 25.1, 22.7, 21.3,
                       20.8, 23.5, 24.2, 25.3, 22.9, 21.2, 22.0, 23.6])
rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
mask=(rainfall==0)
print(mask)
res=temperatures[mask]
print(res) """

""" temperatures=np.array([22.1, 23.4, 21.8, 24.0, 25.2, 22.5, 21.7, 20.9,
                       22.3, 23.1, 21.6, 24.5, 23.8, 22.0, 21.9, 22.4,
                       23.7, 24.1, 22.8, 21.5, 20.7, 22.6, 23.0, 24.3,
                       25.0, 22.2, 21.4, 23.2, 24.6, 25.1, 22.7, 21.3,
                       20.8, 23.5, 24.2, 25.3, 22.9, 21.2, 22.0, 23.6])
rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
mask=(temperatures>=22) & (temperatures<=24)
res=rainfall[mask]
print(res) """

#8
""" temperatures=np.array([22.1, 23.4, 21.8, 24.0, 25.2, 22.5, 21.7, 20.9,
                       22.3, 23.1, 21.6, 24.5, 23.8, 22.0, 21.9, 22.4,
                       23.7, 24.1, 22.8, 21.5, 20.7, 22.6, 23.0, 24.3,
                       25.0, 22.2, 21.4, 23.2, 24.6, 25.1, 22.7, 21.3,
                       20.8, 23.5, 24.2, 25.3, 22.9, 21.2, 22.0, 23.6])
mean_temp=np.mean(temperatures)
std_temp=np.std(temperatures)
result=(temperatures-mean_temp)/std_temp
print(result) """

""" temperatures=np.array([22.1, 23.4, 21.8, 24.0, 25.2, 22.5, 21.7, 20.9,
                       22.3, 23.1, 21.6, 24.5, 23.8, 22.0, 21.9, 22.4,
                       23.7, 24.1, 22.8, 21.5, 20.7, 22.6, 23.0, 24.3,
                       25.0, 22.2, 21.4, 23.2, 24.6, 25.1, 22.7, 21.3,
                       20.8, 23.5, 24.2, 25.3, 22.9, 21.2, 22.0, 23.6])
rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
ratios=rainfall/temperatures
ratio_mean=np.mean(ratios)
ratio_std=np.std(ratios)

new_ratio=np.round(ratios,2)
new_mean=np.round(ratio_mean,2)
new_std=np.round(ratio_std,2)
print(new_ratio)
print("Mean:",new_mean)
print("Standard diveation",new_std) """

#9
""" rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
rainfall[[5,15]]=np.nan
print(rainfall) """

""" rainfall = np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 1.5, 2.1, 0.0, 0.0,
                     0.0, 0.0, 1.8, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                     0.0, 1.2, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.3, 0.0,
                     0.0, 0.0, 2.1, 0.0, 0.0, 1.4, 0.0, 0.0, 1.6, 0.0])
rainfall[[5,15]]=np.nan
new_mean=np.nanmean(rainfall)
print(new_mean) """



