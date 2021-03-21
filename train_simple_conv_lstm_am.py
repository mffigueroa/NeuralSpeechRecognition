import argparse
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_warmup as lr_warmup

from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms, utils
from text_dataset import TextDataset
from speech_dataset import SpeechDataset
from torch_transforms import DeleteSequencePrefix, ToTensor, RemapUsingMinWordID, SpecAugment, PadSequencesCollater

from conv_lstm_am import ConvLSTM_AcousticModel
from resnet_lstm_am import ResNetLSTM_AcousticModel
from resnet_am import ResNet_AcousticModel
from vocabulary import VocabularySpecialWords
from tqdm import tqdm

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def log_model_output_sample(model_samples_file, model_targets, word_scores, vocabulary, lm_min_word_id, dataset_name):
    targets = model_targets[0,:].cpu().numpy() + lm_min_word_id
    target_sentence = vocabulary.tokens_to_sentence(targets)
    target_sentence = target_sentence.replace('[START]','[BLANK]')
    model_samples_file.write(f'\n\n{dataset_name} Target Sentence:\n\t{target_sentence}\n\n')
    model_outputs = torch.argmax(word_scores[:, 0, :], dim=1).cpu().numpy() + lm_min_word_id
    model_output_sentence = vocabulary.tokens_to_sentence(model_outputs)
    model_output_sentence = model_output_sentence.replace('[START]','[BLANK]')
    model_output_sentence = re.sub(r'(\[BLANK\] ?)+', '[BLANK] ', model_output_sentence)
    model_output_sentence = model_output_sentence.replace('  ', ' ')
    model_samples_file.write(f'\n\n{dataset_name} Model Output Sentence:\n\t{model_output_sentence}\n\n')
    model_samples_file.flush()

def str_begins_with(s, prefix):
    if len(s) < len(prefix):
        return False
    return s[:len(prefix)] == prefix
    
def get_model_weight_files(dir):
    weight_files = {}
    weight_file_prefix = 'model_epoch_'
    val_loss_suffix = '_val_'
    files = os.listdir(dir)
    for file in files:
        if not str_begins_with(file, weight_file_prefix):
            continue
        filename_parts = os.path.splitext(file)
        filename_without_ext = filename_parts[0]
        filename_ext = filename_parts[1]
        if not filename_ext.lower() == '.pth':
            continue
        model_suffix_w_loss = re.sub(r'model_epoch_\d+_', '', filename_without_ext)
        file_wo_suffix = filename_without_ext[:-len(model_suffix_w_loss)]
        val_loss_index = model_suffix_w_loss.index(val_loss_suffix)
        val_loss_str = model_suffix_w_loss[val_loss_index+len(val_loss_suffix):]
        model_suffix = model_suffix_w_loss[:val_loss_index]
        val_loss = float(val_loss_str)
        epoch_num_str = re.sub(r'model_epoch_(\d+)_', r'\1', file_wo_suffix)
        epoch_num = int(epoch_num_str) - 1
        file_fullpath = os.path.join(dir, file)
        if not model_suffix in weight_files:
            weight_files[model_suffix] = {}
        weight_files[model_suffix][epoch_num] = {}
        weight_files[model_suffix][epoch_num]['full_path'] = file_fullpath
        weight_files[model_suffix][epoch_num]['val_loss'] = val_loss
    return weight_files

if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    parser = argparse.ArgumentParser(description='Train a simple ConvLSTM acoustic model.')
    parser.add_argument('log_file', help='Path to output log file')
    parser.add_argument('train_dataset', help='Path to processed train dataset file')
    parser.add_argument('valid_dataset', help='Path to processed validation dataset file')
    parser.add_argument('--lm_train_dataset', help='Path to processed language model train dataset file')
    parser.add_argument('--lm_valid_dataset', help='Path to processed language model train dataset file')
    parser.add_argument('--vocab_unk_rate', help='UNKing rate to use for vocabulary, by default will use true UNK rate based on validation set OOV rate', default=-1.0)
    parser.add_argument('--character_level', help='Train the acoustic model at the character level', action='store_true')
    parser.add_argument('--phoneme_level', help='Train the acoustic model at the phoneme level', action='store_true')
    args = parser.parse_args()
    
    n_epochs = 1000
    train_samples_per_epoch = 80000
    valid_samples_per_epoch = 500
    batch_size = 24
    max_sequence_length = 50
    
    logfile_prefix = os.path.splitext(args.log_file)[0]
    logfile_dir = os.path.dirname(args.log_file)
    
    weight_files = get_model_weight_files(logfile_dir)
    
    lm_train_vocab = None
    if not args.character_level and not args.phoneme_level:
        lm_train_dataset = TextDataset(args.lm_train_dataset, max_sequence_length)
        lm_valid_dataset = TextDataset(args.lm_valid_dataset, max_sequence_length)
        
        if args.vocab_unk_rate == -1.0:
            lm_train_dataset.unk_vocabulary_with_true_oov_rate(lm_valid_dataset)
        elif args.vocab_unk_rate > 0:
            lm_train_dataset.unk_vocabulary_with_oov_rate(args.vocab_unk_rate)
        
        lm_train_vocab = lm_train_dataset.vocabulary
    
    train_dataset = SpeechDataset(args.train_dataset, vocabulary=lm_train_vocab, character_level=args.character_level, phoneme_level=args.phoneme_level)
    valid_dataset = SpeechDataset(args.valid_dataset, vocabulary=lm_train_vocab, character_level=args.character_level, phoneme_level=args.phoneme_level)
    max_dataset_input_length = max(train_dataset.get_max_input_length(), valid_dataset.get_max_input_length())
    max_dataset_target_length = max(train_dataset.get_max_transcription_length(), valid_dataset.get_max_transcription_length())
    
    max_word_id = train_dataset.vocabulary.get_max_word_id()
    
    # need to leave 0 for a BLANK symbol for CTC decoding, and thus STOP should map to 1 not 0
    lm_min_word_id = VocabularySpecialWords.STOP.value - 1 #train_dataset.vocabulary.get_min_valid_lm_output_word_id()
    
    vocabulary_size = train_dataset.vocabulary.get_vocab_size()
    valid_dataset.use_vocabulary_from_dataset(train_dataset)
    print(f'Vocabulary Size: {vocabulary_size}')
    
    dataset_transformer = transforms.Compose([
        DeleteSequencePrefix('target', 1), # target should not include START symbol at beginning
        RemapUsingMinWordID('target',lm_min_word_id),
        ToTensor()])#,
        #SpecAugment(sample_key='input', freq_mask_length=3, time_mask_length=70)])
    
    train_dataset.set_transform(dataset_transformer)
    valid_dataset.set_transform(dataset_transformer)
    
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=train_samples_per_epoch)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, collate_fn=PadSequencesCollater())
    valid_sampler = RandomSampler(valid_dataset, replacement=True, num_samples=valid_samples_per_epoch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0, collate_fn=PadSequencesCollater())
    
    spectogram_num_filters = 13
    
    bottleneck_size = 2**np.arange(3,10)
    learning_rates = 10.**np.arange(-5,-2)
    hyperparameters_tried = set()
    
    while True:
        random_bottleneck_size = np.random.choice(bottleneck_size)
        random_learning_rate = 1e-5#np.random.choice(learning_rates)
        hyperparameter_combo = (random_learning_rate,random_bottleneck_size)
        if hyperparameter_combo in hyperparameters_tried:
            continue
        hyperparameters_tried.add(hyperparameter_combo)
        
        model_suffix = f'bottleneck_{random_bottleneck_size}_lr_{random_learning_rate}_resnet50'
        model_save_name = f'model_{model_suffix}.pth'
        model_save_path = os.path.join(logfile_dir, model_save_name)
        if os.path.isfile(model_save_path):
            continue
        
        model = ResNet_AcousticModel(spectogram_num_filters, lm_min_word_id, max_word_id, bottleneck_size=random_bottleneck_size, resnet_backbone_name='resnet50')
        model.cuda()
        model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\n\nTotal Parameters: {model_total_params}\n\n' + str(model))
        
        loss_function = nn.CTCLoss(zero_infinity=True)
        optimizer = optim.Adam(model.parameters(), lr=random_learning_rate)
        print(f'\nLearning rate: {get_lr(optimizer)}')
        
        first_epoch = 0
        validation_losses = []
        best_validation_loss = np.inf
        
        if model_suffix in weight_files:
            largest_epoch_of_weights = max(weight_files[model_suffix].keys())
            first_epoch = largest_epoch_of_weights + 1
            weight_file_path = weight_files[model_suffix][largest_epoch_of_weights]['full_path']
            epoch_checkpoint = torch.load(open(weight_file_path, 'rb'))
            model.load_state_dict(epoch_checkpoint['model_state_dict'])
            optimizer.load_state_dict(epoch_checkpoint['optimizer_state_dict'])
            
            for epoch_num, weight_info in weight_files[model_suffix].items():
                validation_losses.append(weight_info['val_loss'])
            
            best_validation_loss = min(validation_losses)
        
        logfile_path = f'{logfile_prefix}_{model_suffix}.txt'
        
        if not os.path.isfile(logfile_path):
            log_file = open(logfile_path, 'w')            
            log_file.write(f'Model Total Parameters: {model_total_params}\n')
            log_file.write(f'Epoch #,Train Average Loss,Validation Average Loss\n')
            log_file.flush()
        else:
            log_file = open(logfile_path, 'a')
        
        model_samples_filepath = f'{logfile_prefix}_output_samples_{model_suffix}.txt'
        
        if not os.path.isfile(model_samples_filepath):
            model_samples_file = open(model_samples_filepath, 'w')
        else:
            model_samples_file = open(model_samples_filepath, 'a')
        
        model.train()
        
        steps_per_epoch = len(train_dataloader)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=5)
        
        warmup_last_step = -1
        if first_epoch > 0:
            warmup_last_step = first_epoch * steps_per_epoch
            
        warmup_scheduler = lr_warmup.UntunedLinearWarmup(optimizer, last_step=warmup_last_step)
        for epoch in range(first_epoch, n_epochs):
            train_number_of_words = 0
            batches_loop = tqdm(train_dataloader)
            total_train_loss = 0
            num_train_batches = 0
            
            num_batch_exceptions = 0
            for batch_sample in batches_loop:
                try:
                    model.zero_grad()
                    model_inputs = batch_sample['input'].cuda()
                    model_targets = batch_sample['target'].cuda()
                    model_target_lengths = batch_sample['target_lengths'].cuda()
                    #print(f'\nmodel_targets: {model_targets.size()}')
                    
                    word_scores = model(model_inputs)
                    ctc_loss_input_length_dim = word_scores.size()[0]
                    sample_batch_size = word_scores.size()[1]
                    ctc_loss_input_length = torch.LongTensor([ctc_loss_input_length_dim]).repeat(sample_batch_size).cuda()
                    
                    loss = loss_function(word_scores, model_targets, ctc_loss_input_length, model_target_lengths)
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step(epoch + num_train_batches / steps_per_epoch)
                    warmup_scheduler.dampen()
                    
                    with torch.no_grad():
                        if np.random.uniform(0,1) >= 0.97:
                            log_model_output_sample(model_samples_file, model_targets, word_scores, train_dataset.vocabulary, lm_min_word_id, 'TRAIN')
                        total_train_loss += loss
                    
                    batches_loop.set_description('Train Epoch {}/{}'.format(epoch + 1, n_epochs))
                    batches_loop.set_postfix(loss=loss.item())
                    num_train_batches += 1
                    num_batch_exceptions = 0
                    
                except Exception as e:
                    num_batch_exceptions += 1
                    if num_batch_exceptions > 5:
                        raise e
            
            valid_number_of_words = 0
            batches_loop = tqdm(valid_dataloader)
            total_valid_loss = 0
            with torch.no_grad():
                avg_train_loss = total_train_loss / num_train_batches
                avg_train_loss = avg_train_loss.cpu().numpy()
                num_valid_batches = 0
                for batch_sample in batches_loop:
                    model_inputs = batch_sample['input'].cuda()
                    model_targets = batch_sample['target'].cuda()
                    model_target_lengths = batch_sample['target_lengths'].cuda()
                    
                    word_scores = model(model_inputs)
                    ctc_loss_input_length_dim = word_scores.size()[0]
                    sample_batch_size = word_scores.size()[1]
                    ctc_loss_input_length = torch.LongTensor([ctc_loss_input_length_dim]).repeat(sample_batch_size).cuda()
                    
                    loss = loss_function(word_scores, model_targets, ctc_loss_input_length, model_target_lengths)
                    
                    total_valid_loss += loss
                    
                    if np.random.uniform(0,1) >= 0.5:
                        log_model_output_sample(model_samples_file, model_targets, word_scores, train_dataset.vocabulary, lm_min_word_id, 'VALID')
                    
                    batches_loop.set_description('Validation Epoch {}/{}'.format(epoch + 1, n_epochs))
                    batches_loop.set_postfix(loss=f'{loss.item():.2f}')
                    num_valid_batches += 1
                
                avg_valid_loss = total_valid_loss / num_valid_batches
                avg_valid_loss = avg_valid_loss.cpu().numpy()
                validation_losses.append(avg_valid_loss)
                
                if avg_valid_loss < best_validation_loss:
                    print(f'New Best Avg Validation Loss: {avg_valid_loss:.6}')
                    best_validation_loss = avg_valid_loss
                    model_epoch_save_name = f'model_epoch_{epoch+1}_{model_suffix}_val_{avg_valid_loss:.5}.pth'
                    model_epoch_save_path = os.path.join(logfile_dir, model_epoch_save_name)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'avg_train_loss': avg_train_loss,
                        'avg_valid_loss': avg_valid_loss
                        }, model_epoch_save_path)
                
                log_file.write(f'{epoch+1},{avg_train_loss},{avg_valid_loss}\n')
                log_file.flush()
        
        torch.save(model.state_dict(), model_save_path)
