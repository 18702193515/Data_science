class Node():
    def __init__(self, value=None, next=None): #这样的特殊命名保证了每次都会初始化
        self.value = value
        self.next = next

class LinkedList():
    def __init__(self):
        self.head = None

    def insert(self, value):
        new_node = Node(value)
        new_node.next = self.head
        self.head = new_node

    def remove(self, value):
        cur = self.head
        pre = None
        while cur is not None:
            if cur.value == value:
                if not pre:
                    self.head = cur.next
                else:
                    pre.next = cur.next
                return True
            else:
                pre = cur
                cur = cur.next
        return False

    def find(self, value):
        cur = self.head
        while cur is not None:
            if cur.value == value:
                return True
            cur = cur.next
        return False

    def change(self, old_value, new_value):
        cur = self.head
        while cur is not None:
            if cur.value == old_value:
                cur.value = new_value
                return True
            cur = cur.next
        return False
    



my_list = LinkedList()

my_list.insert(10)
my_list.insert(20)
my_list.insert(30)

current = my_list.head
print("链表元素：")
while current is not None:
    print(current.value)
    current = current.next

print("寻找20、40：")
print(my_list.find(20))  
print(my_list.find(40))  

print("将20改为25,40改为50：")
print(my_list.change(20, 25))  
print(my_list.change(40, 50))  

print("将20、40删除：")
print(my_list.remove(20))  
print(my_list.remove(40)) 

print("链表元素：")
current = my_list.head
while current is not None:
    print(current.value)
    current = current.next
