"""========================================================================================
                                           Import
========================================================================================"""
import torch 
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import dataloader
import numpy as np
import torch.nn.functional as F
from lab5 import EncoderRNN, DecoderRNN

"""========================================================================================
                                    basic setting
========================================================================================"""
device = torch.device("cuda:0" )
SOS_token = 0
EOS_token = 1
UKN_token = 2
MAX_LENGTH = 25
################################
#Example inputs of compute_bleu
################################
#The target word
reference = 'variable'
#The word generated by your model
output = 'varable'

vocab_size = 28
hidden_size = 256
latent_size = 32
condition_size = 4

"""========================================================================================
                                   Definition of Function
========================================================================================"""
"""
   load_weight
"""
def load_weight(model , path):
    model.load_state_dict(torch.load(path+".pkl"))
    return model

def Gaussian_score(words):
    #words=['arrise', 'ents', 'enting', 'ent']
    #print(words)
    words_list = []
    score = 0
    yourpath = '/home/ubuntu/DL_LAB4/train.txt'
    with open(yourpath,'r') as fp:
        for line in fp:
            #print(line)
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        #print(words_list)
        for t in words:
            for i in words_list:
                #print(i)
                if t == i:
                    score += 1
    return score/len(words)
"""
   compute BLEU-4 score
"""

def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

"""
   evaluate
"""
def evaluate(encoder, decoder, input_tensor, target_tensor, target_condition):
    vocabulary = str()
    for voc in range(97 , 97+26):
        vocabulary += chr(voc)
    index2word =  {i+2 : c  for i, c in enumerate(vocabulary)}
    index2word[0] = 'SOS'
    index2word[1] = 'EOS'
    encoder.eval()
    decoder.eval()
    input_length = input_tensor.size(0)
    encoder_hidden = encoder.initHidden()
    encoder_cell   = encoder.initHidden()
    # tense condition
    #print('target_condition')
    #print(target_condition)
    target_condition    = target_condition.view(1,1,-1).float()
    mean , log_var , encoder_hidden , encoder_cell = encoder(input_tensor, encoder_hidden , encoder_cell)
    #z = encoder.reparameter( mean , log_var , "eval")
    z = encoder.reparameter( mean , log_var , "gaussian")
    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

    decoder_hidden = encoder_hidden
    decoder_cell   = torch.cat([z , target_condition] , dim = -1)
    decoder_cell   = decoder.latent_to_hidden(decoder_cell)
    decoder_chars = []
    for di in range(vocab_size):
        decoder_output, decoder_hidden , decoder_cell = decoder(
            decoder_input, decoder_hidden , decoder_cell)
        decoder_input = F.softmax(decoder_output,dim= -1).argmax(dim=-1).view(1,-1)
        if decoder_input.item() == EOS_token:
            decoder_chars.append(index2word[decoder_input.view(-1).cpu().numpy()[0]])
            break
        else :
            decoder_chars.append(index2word[decoder_input.view(-1).cpu().numpy()[0]])   

    decoder_word = ""
    return decoder_word.join(decoder_chars[:-1])
"""
   test_run_evaluate
"""
def test_run_evaluate(encoder, decoder,print_result ,path_encoder , path_decoder):
    '''
       test tense
       sp -> p
       sp -> pg
       sp -> tp
       sp -> tp
       p  -> tp
       sp -> pg
       p  -> sp
       pg -> sp
       pg -> p
       pg -> tp
    '''
    bleu_score = 0
    n_iters = 10
    testing_pairs = dataloader.read_testing_data( n_iters , "eval")
    if testing_pairs:
        print('Read testing data successfully')
    vocabulary = str()
    for voc in range(97 , 97+26):
        vocabulary += chr(voc)
    index2word =  {i+2 : c  for i, c in enumerate(vocabulary)}
    index2word[0] = 'SOS'
    index2word[1] = 'EOS'
    encoder.load_state_dict(torch.load(path_encoder+".pkl"))
    decoder.load_state_dict(torch.load(path_decoder+".pkl"))
    for iter in range(1, n_iters + 1):
        input_tensor  = testing_pairs[2*(iter - 1)]
        target_tensor = testing_pairs[2*(iter - 1)+1][1:-4]
        target_condition = testing_pairs[2*(iter - 1)+1][-4:]
        
        if iter-1==6 or iter-1==7:
            target_condition[0] = 1
            target_condition[1] = 0
        elif iter-1 == 1 or iter-1 == 5 :
            target_condition[1] = 0
            target_condition[2] = 1
        elif iter-1 ==0 or iter-1 == 8:
            target_condition[1] = 0
            target_condition[3] = 1
            
        output_word = evaluate(encoder, decoder, input_tensor , target_tensor, target_condition)
        target_chars = []
        input_chars  = []
        print("============================")
        for di in range(len(target_tensor)):
            if target_tensor[di].item() == EOS_token:
               target_chars.append(index2word[target_tensor[di].view(-1).cpu().numpy()[0]])
               break
            elif target_tensor[di].item() == SOS_token :
                pass
            else:
                target_chars.append(index2word[target_tensor[di].view(-1).cpu().numpy()[0]])
        for di in range(len(input_tensor)):
            if input_tensor[di].item() == EOS_token:
               break
            elif input_tensor[di].item() == SOS_token :
                pass
            else:
                input_chars.append(index2word[input_tensor[di].view(-1).cpu().numpy()[0]])
        input_word = ""
        input_word = input_word.join(input_chars[0:])
        target_word = ""
        target_word = target_word.join(target_chars[:-1])
        bleu_score += compute_bleu(output_word, target_word)
        print("Input: ",input_word,
              "\nTarget: ",target_word,
              "\nPred: ",output_word)
    bleu_score = bleu_score / n_iters
    if print_result:
        print("BLEU4-Score: %0.4f" % bleu_score)
    return bleu_score
    
"""
   run_evaluate
"""
def run_evaluate(encoder, decoder,print_result ):
    '''
       test tense
       sp -> p
       sp -> pg
       sp -> tp
       sp -> tp
       p  -> tp
       sp -> pg
       p  -> sp
       pg -> sp
       pg -> p
       pg -> tp
    '''
    bleu_score = 0
    n_iters = 10
    
    testing_pairs = dataloader.read_testing_data( n_iters , "eval")
    if testing_pairs:
        print('Read testing data successfully')
    vocabulary = str()
    for voc in range(97 , 97+26):
        vocabulary += chr(voc)
    index2word =  {i+2 : c  for i, c in enumerate(vocabulary)}
    index2word[0] = 'SOS'
    index2word[1] = 'EOS'
    for iter in range(1, n_iters + 1):
        input_tensor  = testing_pairs[2*(iter - 1)]
        target_tensor = testing_pairs[2*(iter - 1)+1][1:-4]
        target_condition = testing_pairs[2*(iter - 1)+1][-4:]
        
        if iter-1==6 or iter-1==7:
            target_condition[0] = 1
            target_condition[1] = 0
        elif iter-1 == 1 or iter-1 == 5 :
            target_condition[1] = 0
            target_condition[2] = 1
        elif iter-1 ==0 or iter-1 == 8:
            target_condition[1] = 0
            target_condition[3] = 1
        
        output_word = evaluate(encoder, decoder, input_tensor , target_tensor, target_condition)
        target_chars = []
        input_chars  = []
        print("============================")
        for di in range(len(target_tensor)):
            if target_tensor[di].item() == EOS_token:
               target_chars.append(index2word[target_tensor[di].view(-1).cpu().numpy()[0]])
               break
            elif target_tensor[di].item() == SOS_token :
                pass
            else:
                target_chars.append(index2word[target_tensor[di].view(-1).cpu().numpy()[0]])
        for di in range(len(input_tensor)):
            if input_tensor[di].item() == EOS_token:
               break
            elif input_tensor[di].item() == SOS_token :
                pass
            else:
                input_chars.append(index2word[input_tensor[di].view(-1).cpu().numpy()[0]])
        input_word = ""
        input_word = input_word.join(input_chars[0:])
        target_word = ""
        target_word = target_word.join(target_chars[:-1])
        bleu_score += compute_bleu(output_word, target_word)
        print("Input: ",input_word,
              "\nTarget: ",target_word,
              "\nPred: ",output_word)
    bleu_score = bleu_score / n_iters
    if print_result:
        print("BLEU4-Score: %0.4f" % bleu_score)

    return bleu_score

"""
   test_run_gaussian
"""
def test_run_gaussian(encoder, decoder,print_result ,path_encoder , path_decoder):
    '''
    test tense
    sp -> p
    sp -> pg
    sp -> tp
    sp -> tp
    p  -> tp
    sp -> pg
    p  -> sp
    pg -> sp
    pg -> p
    pg -> tp
    '''
    gaussian_score = 0
    n_iters = 10
    # testing_pairs = dataloader.read_data(, n_iters)
    # testing_pairs = dataloader.read_data('./test.txt', n_iters , "eval")
    vocabulary = str()
    for voc in range(97 , 97+26):
        vocabulary += chr(voc)
    index2word =  {i+2 : c  for i, c in enumerate(vocabulary)}
    index2word[0] = 'SOS'
    index2word[1] = 'EOS'
    # index2word[2] = 'UKN'
    encoder.load_state_dict(torch.load(path_encoder+".pkl"))
    decoder.load_state_dict(torch.load(path_decoder+".pkl"))
    for iter in range(1, n_iters + 1):
        output_word = gaussian_evaluate(encoder, decoder)
        # target_chars = []
        # input_chars  = []
        print("============================")
       
        gaussian_score += Gaussian_score(output_word)
        # simple present, third person, present progressive, past
        print("Simple present: ",output_word[0],
              "\nthird person: ",output_word[1],
              "\npresent progressive: ",output_word[2],
            "\npast: ",output_word[-1])
    gaussian_score = gaussian_score / n_iters
    if print_result:
        print("Gaussian-Score: %0.4f" % gaussian_score)

"""
   gaussian_evaluate
"""
def gaussian_evaluate(encoder, decoder):
    vocabulary = str()
    for voc in range(97 , 97+26):
        vocabulary += chr(voc)
    index2word =  {i+2 : c  for i, c in enumerate(vocabulary)}
    index2word[0] = 'SOS'
    index2word[1] = 'EOS'
    # index2word[2] = 'UKN'
    encoder.eval()
    decoder.eval()
    encoder_hidden = encoder.initHidden()
    encoder_cell   = encoder.initHidden()
    # tense condition
    tense = np.eye(4).astype('int32')

    # for di in range(input_length):
    #     encoder_output, encoder_hidden , encoder_cell= encoder(input_tensor[di], encoder_hidden , encoder_cell)
    # mean , log_var , encoder_hidden , encoder_cell = encoder(input_tensor, encoder_hidden , encoder_cell)
    mean=0
    std=0
    z = encoder.reparameter( 10 , 10000 , "gaussian")
    #print(z)
    z = np.array(z)
    for i in range(32):
      z[0][0][i] = np.exp(std) * z[0][0][i] + mean
    z = torch.tensor(z)
    z = z.cuda().to(device)
    #print('z')
    #print(z)
    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
    words = []
    for  l in range( tense.shape[0]):
        decoder_hidden = encoder_hidden
        tense =torch.Tensor(tense)
        condition    = tense[l].view(1 ,1 ,-1 ).float()
        condition    = condition.cuda().to(device)
        decoder_cell   = torch.cat([z , condition] , dim = -1)
        decoder_cell   = decoder.latent_to_hidden(decoder_cell)
        # decoder_cell   = decoder.initHidden()
        decoder_chars = []
        for di in range(vocab_size):
            decoder_output, decoder_hidden , decoder_cell = decoder(
                decoder_input, decoder_hidden , decoder_cell)
            # decoder_input = F.softmax(decoder_output,dim=1).argmax(dim=1).view(1,-1)
            decoder_input = F.softmax(decoder_output,dim= -1).argmax(dim=-1).view(1,-1)
            if decoder_input.item() == EOS_token:
                decoder_chars.append(index2word[decoder_input.view(-1).cpu().numpy()[0]])
                break
            else :
                decoder_chars.append(index2word[decoder_input.view(-1).cpu().numpy()[0]])   
    
        decoder_word = ""
        decoder_word=decoder_word.join(decoder_chars[:-1])
        words.append(decoder_word)
    return words
    
"""if __name__ == '__main__':
  path_encoder = './checkpoint/ori/encoder/85200'
  path_decoder = './checkpoint/ori/decoder/85200'
  encoder = EncoderRNN(vocab_size, hidden_size).to(device)
  decoder = DecoderRNN(hidden_size, vocab_size, latent_size, condition_size).to(device)
  score = test_run_gaussian(encoder, decoder,True,path_encoder,path_decoder)
"""