import os
import random

root='/home/tdteach/data/yellowset/'

jfs = os.listdir(root)
rate = 0.01
split_number = 10000


import pickle
def read_from_bin(fname):
  print('read from '+fname)
  with open(fname,'rb') as f:
    data = pickle.load(f)
  return data

def write_to_bin(data,fname,k=0):
  st = 0
  sp = split_number
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
tr_kk = 0
vl_kk = 0
for fn in jfs:
  if 'bin' not in fn:
    continue
  data = read_from_bin(root+fn)
  if 'yellow' in fn or 'porn' in fn:
    lb = 1
  else:
    lb = 0
  print('deal '+fn)
  for n,d in data.items():
    if random.random() < rate:
      vl.append((d,lb))
    else:
      tr.append((d,lb))
  if len(vl) >= split_number:
    write_to_bin(vl[0:split_number],'valid.bin',vl_kk)
    vl_kk = vl_kk+1
    vl = vl[split_number:]
  if len(tr) >= split_number:
    write_to_bin(tr[0:split_number],'train.bin',tr_kk)
    tr_kk = tr_kk+1
    tr = tr[split_number:]

if len(tr) > 0:
  write_to_bin(tr,'train.bin')
if len(vl) > 0:
  write_to_bin(vl,'valid.bin')

'''
for i in range(10):
    data = read_from_bin(root+('white_imgs.bin.%d' % i))
    print(len(data))
    s += len(data)

print(s)
'''
