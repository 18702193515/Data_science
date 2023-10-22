import random
import time


def selection_sort(arr):
    for i in range(len(arr)):
        min_index = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    return quick_sort(less) + equal + quick_sort(greater)

def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr


def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def generate_random_list(length):

    return [random.randint(0, 1000000) for _ in range(length)]

lengths = [100, 1000, 5000, 10000]
for length in lengths:
    arr = generate_random_list(length)
    
    start_time = time.time()
    sorted_arr = selection_sort(arr.copy())
    end_time = time.time()
    selection_sort_time = end_time - start_time

    start_time = time.time()
    sorted_arr = merge_sort(arr.copy())
    end_time = time.time()
    merge_sort_time = end_time - start_time

    start_time = time.time()
    sorted_arr = quick_sort(arr.copy())
    end_time = time.time()
    quick_sort_time = end_time - start_time

    start_time = time.time()
    sorted_arr = shell_sort(arr.copy())
    end_time = time.time()
    shell_sort_time = end_time - start_time

    print(f"长度为 {length} 的数列:")
    print(f"选择排序耗时：{selection_sort_time:.6f} 秒")
    print(f"归并排序耗时：{merge_sort_time:.6f} 秒")
    print(f"快速排序耗时：{quick_sort_time:.6f} 秒")
    print(f"希尔排序耗时：{shell_sort_time:.6f} 秒")
    print("--------------------------")