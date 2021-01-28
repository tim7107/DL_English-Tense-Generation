import torch
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
hidden_size = 256 #Hyper Parameters
vocab_size = 28 #The number of vocabulary
teacher_forcing_ratio = 1.0
empty_input_ratio = 0.1
KLD_weight = 0.0
LR = 0.05

'''
path : dataset path
n_iters : total iters for training
mode : "train" or "eval"
'''

def read_training_data( n_iters , mode ):
    """
       open txt file
    """
    fp=open(r'/home/ubuntu/DL_LAB4/train.txt')
    input_data = []
    for line in fp:
      input_data.append(line.strip())
      
    """
       vocabulary = abcdefghijklmnopqrstuvwxyz
       type(vocabulary) = str
       len(vocabulary) = 26
       word2index = {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20,                      't': 21, 'u': 22, 'v': 23, 'w': 24, 'x': 25, 'y': 26, 'z': 27}
       index2word = {2: 'a', 3: 'b', 4: 'c', 5: 'd', 6: 'e', 7: 'f', 8: 'g', 9: 'h', 10: 'i', 11: 'j', 12: 'k', 13: 'l', 14: 'm', 15: 'n', 16: 'o', 17: 'p', 18: 'q', 19: 'r', 20: 's',                      21: 't', 22: 'u', 23: 'v', 24: 'w', 25: 'x', 26: 'y', 27: 'z'}
    """
    vocabulary = str()
    for voc in range(97 , 97+26):
        vocabulary += chr(voc)
    word2index =  {c : i+2 for i, c in enumerate(vocabulary)}
    index2word =  {i+2 : c  for i, c in enumerate(vocabulary)}
    
    """
       tense = [[1 0 0 0]
               [0 1 0 0]
               [0 0 1 0]
               [0 0 0 1]]
       iter : 0~1226
       idx : 0~3 
       i : len of seq 
       one hot vector for 4 tenses
       tense = ['sp' , 'tp' , 'pc' , 'pa']
       list_input = [SOS,.....,EOS,one hot vector]  EX.[0, 6, 25, 2, 4, 6, 19, 3, 2, 21, 6, 5, 1,    0, 0, 0, 1]
    """
    tense = np.eye(4).astype('int32')
    training_input = []
    for iter in range(len(input_data)):
      training_pair = input_data[iter]
      training_data  = training_pair.split(' ')
      training_data[-1] = training_data[-1].strip('\n')
      for idx , data , in enumerate(training_data):
          list_input = []
          for i in range(len(data)):
                #transpose seq to index
                list_input.append(word2index[data[i]])
          #insert SOS and EOS
          list_input.insert(0,0)
          list_input.append(1)
          #add tense one hot into list_input
          for j in range( len(tense[idx])):
                list_input.append(tense[idx][j]) 
          #tranfer type list to type np.array then to tensor
          input_array = np.array(list_input,dtype=float)
          input_tensor = torch.LongTensor(input_array).cuda().to(device)
          training_input.append(input_tensor)
    """
       make(add) the number of training_data to n_iters
    """
    current_length = len(training_input)
    times = (n_iters-current_length) // current_length +1     
    final_data = training_input.copy()
    for t in range(times):
        if t < times-1:
            length = len(training_input)
        else:
            length = (n_iters-current_length) % current_length
        for i in range(length):
            final_data.append(training_input[i])
    if mode == "train":
        random.shuffle(final_data)

    return final_data

def read_testing_data( n_iters , mode ):
    """
       open txt file
    """
    fp=open(r'/home/ubuntu/DL_LAB4/test.txt')
    input_data = []
    for line in fp:
      input_data.append(line.strip())
      
    """
       vocabulary = abcdefghijklmnopqrstuvwxyz
       type(vocabulary) = str
       len(vocabulary) = 26
       word2index = {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20,                      't': 21, 'u': 22, 'v': 23, 'w': 24, 'x': 25, 'y': 26, 'z': 27}
       index2word = {2: 'a', 3: 'b', 4: 'c', 5: 'd', 6: 'e', 7: 'f', 8: 'g', 9: 'h', 10: 'i', 11: 'j', 12: 'k', 13: 'l', 14: 'm', 15: 'n', 16: 'o', 17: 'p', 18: 'q', 19: 'r', 20: 's',                      21: 't', 22: 'u', 23: 'v', 24: 'w', 25: 'x', 26: 'y', 27: 'z'}
    """
    vocabulary = str()
    for voc in range(97 , 97+26):
        vocabulary += chr(voc)
    word2index =  {c : i+2 for i, c in enumerate(vocabulary)}
    index2word =  {i+2 : c  for i, c in enumerate(vocabulary)}
    
    """
       tense = [[1 0 0 0]
               [0 1 0 0]
               [0 0 1 0]
               [0 0 0 1]]
       iter : 0~1226
       idx : 0~3 
       i : len of seq 
       one hot vector for 4 tenses
       tense = ['sp' , 'tp' , 'pc' , 'pa']
       list_input = [SOS,.....,EOS,one hot vector]  EX.[0, 6, 25, 2, 4, 6, 19, 3, 2, 21, 6, 5, 1,    0, 0, 0, 1]
    """
    tense = np.eye(4).astype('int32')
    training_input = []
    for iter in range(len(input_data)):
      training_pair = input_data[iter]
      training_data  = training_pair.split(' ')
      training_data[-1] = training_data[-1].strip('\n')
      for idx , data , in enumerate(training_data):
          list_input = []
          for i in range(len(data)):
                #transpose seq to index
                list_input.append(word2index[data[i]])
          #insert SOS and EOS
          list_input.insert(0,0)
          list_input.append(1)
          #add tense one hot into list_input
          for j in range( len(tense[idx])):
                list_input.append(tense[idx][j]) 
          #tranfer type list to type np.array then to tensor
          input_array = np.array(list_input,dtype=float)
          input_tensor = torch.LongTensor(input_array).cuda().to(device)
          training_input.append(input_tensor)
    """
       make(add) the number of training_data to n_iters
    """
    current_length = len(training_input)
    times = (n_iters-current_length) // current_length +1     
    final_data = training_input.copy()
    for t in range(times):
        if t < times-1:
            length = len(training_input)
        else:
            length = (n_iters-current_length) % current_length
        for i in range(length):
            final_data.append(training_input[i])
    if mode == "train":
        random.shuffle(final_data)
    return final_data         
         
train_data=read_training_data(75000,"train")
if train_data:
  print('read training input data successfully')

#print(train_data)
#print(len(train_data))