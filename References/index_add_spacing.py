f = open('index.txt')
count = 1
q = []
for i in f.readlines():
    q.append(i)
    q.append('\n')
    count+=1

f.close()
k = open('output.txt','w+')
k.writelines(q)
k.close()
