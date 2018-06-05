import os
import random

root='/home/tdteach/data/yellowset/'

jfs = os.listdir(root)
rate = 0.01


import pickle
def read_from_bin(fname):
  print('read from '+fname)
  with open(fname,'rb') as f:
    data = pickle.load(f)
  return data

def write_to_bin(data,fname):
  st = 0
  sp = 10000
  k = 0
  while st < len(data):
    ed = min(st+sp, len(data))
    tn = '%s.%d'%(fname,k)
    with open(tn,'wb') as f:
      print('write to '+tn)
      bd = pickle.dumps(data[st:ed])
      f.write(bd)
    st = ed
    k = k+1


tr = []
vl = []

for fn in jfs:
  if 'bin' not in fn:
    continue
  data = read_from_bin(root+fn)
  if 'yellow' in fn:
    lb = 1
  else:
    lb = 0
  print('deal '+fn)
  for n,d in data.items():
    if random.random() < rate:
      vl.append((d,lb))
    else:
      tr.append((d,lb))

write_to_bin(tr,'train.bin')
write_to_bin(vl,'valid.bin')

'''
for i in range(10):
    data = read_from_bin(root+('white_imgs.bin.%d' % i))
    print(len(data))
    s += len(data)

print(s)
'''
