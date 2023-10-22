import random
import time
#通过调用time.time获取运行开始时间和结束时间，相减获得的结果就是运行时间

def selection_sort(arr):
    for i in range(len(arr)):
        min_index = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_index]:
                
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

def generate_random_list(length):

    return [random.randint(0, 1000000) for _ in range(length)]

lengths = [100, 1000, 5000, 10000]
for length in lengths:
    arr = generate_random_list(length)
    
    start_time = time.time()
    sorted_arr = selection_sort(arr.copy())
    end_time = time.time()
    selection_sort_time = end_time - start_time


    print(f"长度为 {length} 的数列:")
    print(f"选择排序耗时：{selection_sort_time:.6f} 秒")
    print("--------------------------")