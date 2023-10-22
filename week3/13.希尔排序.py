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

arr = [5, 2, 7, 9, 8, 4, 6, 1, 3]
print("原数组：",arr)

shell_sort(arr)
print(arr)

#希尔排序的时间复杂度是根据步长序列的选择而变化的，最坏情况下的时间复杂度是O(n^2)，平均情况下的时间复杂度介于O(n log n)和O(n^2)之间。
#空间复杂度为O(1)，因为希尔排序使用的是原地排序，不需要额外的空间来存储数据。