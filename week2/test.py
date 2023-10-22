from collections import deque

class FarmerTransport:
    def __init__(self):
        self.s = '0000'
        self.t = '1111'
        self.pre = {}
        self.des = {}

        queue = deque()

        if self.is_failed(self.s):
            return

        sta = self.s
        queue.append(sta)
        self.pre[sta] = sta
        while queue:
            sta = queue.popleft()

            next_sta = self.get_next_sta(sta)

            for cur_sta in next_sta:
                if cur_sta not in self.pre:
                    queue.append(cur_sta)
                    self.pre[cur_sta] = sta
                    self.des[sta + cur_sta] = next_sta[cur_sta]
                    if cur_sta == self.t:
                        return

    def get_next_sta(self, sta):
        next_sta = {}
        chars = list(sta)
        backup = ''
        n = ''
        if chars[0] == '1':
            chars[0] = '0'
            n = ''.join(chars)
            if not self.is_failed(n):
                next_sta[n] = '农夫从东侧到西侧'
            backup = chars[1]
            if chars[1] == '1':
                chars[1] = '0'
                n = ''.join(chars)
                if not self.is_failed(n):
                    next_sta[n] = '农夫从东侧带狼到西侧'
            chars[1] = backup
            backup = chars[2]
            if chars[2] == '1':
                chars[2] = '0'
                n = ''.join(chars)
                if not self.is_failed(n):
                    next_sta[n] = '农夫从东侧带羊到西侧'
            chars[2] = backup
            backup = chars[3]
            if chars[3] == '1':
                chars[3] = '0'
                n = ''.join(chars)
                if not self.is_failed(n):
                    next_sta[n] = '农夫从东侧带菜到西侧'
            chars[3] = backup
        elif chars[0] == '0':
            chars[0] = '1'
            n = ''.join(chars)
            if not self.is_failed(n):
                next_sta[n] = '农夫从西侧到东侧'
            backup = chars[1]
            if chars[1] == '0':
                chars[1] = '1'
                n = ''.join(chars)
                if not self.is_failed(n):
                    next_sta[n] = '农夫从西侧带狼到东侧'
            chars[1] = backup
            backup = chars[2]
            if chars[2] == '0':
                chars[2] = '1'
                n = ''.join(chars)
                if not self.is_failed(n):
                    next_sta[n] = '农夫从西侧带羊到东侧'
            chars[2] = backup
            backup = chars[3]
            if chars[3] == '0':
                chars[3] = '1'
                n = ''.join(chars)
                if not self.is_failed(n):
                    next_sta[n] = '农夫从西侧带菜到东侧'
            chars[3] = backup
        return next_sta

    def process(self):
        res = []
        if self.t not in self.pre:
            return res
        cur = self.t
        while cur != self.s:
            res.append(cur)
            cur = self.pre[cur]
        res.append(self.s)
        res.reverse()

        ret = []
        p = res[0]
        for i in range(1, len(res)):
            ret.append("状态 : " + self.get_sta_des(p))
            ret.append("步骤 : " + str(i))
            s = res[i]
            ret.append("操作 : " + self.des[p + s])
            p = s
        ret.append("状态 : " + self.get_sta_des(p))
        return ret

    def get_sta_des(self, sta):
        s = list(sta)
        dc = ['东', '：', '空', '空', '空', '空']
        xc = ['西', '：', '空', '空', '空', '空']
        d = []
        x = []
        c = None
        for i in range(len(s)):
            if s[i] == '1':
                c = d
            else:
                c = x

            if i == 0:
                c.append('人')
            elif i== 1:
                c.append('狼')
            elif i == 2:
                c.append('羊')
            elif i == 3:
                c.append('菜')

        dc[2:6] = d
        xc[2:6] = x

        return ''.join(dc) + '   ' + ''.join(xc)

    def is_failed(self, sta):
        s = list(sta)
        if (s[1] == s[2] and s[0] != s[1]) or (s[2] == s[3] and s[0] != s[2]):
            return True
        return False


if __name__ == '__main__':
    ft = FarmerTransport()
    result = ft.process()
    for item in result:
        print(item)