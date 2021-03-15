import argparse
import os
import sys
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_warmup as lr_warmup

from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms, utils
from text_dataset import TextDataset
from speech_dataset import SpeechDataset
from torch_transforms import DeleteSequencePrefix, ToTensor, RemapUsingMinWordID, PadSequencesCollater

from conv_lstm_am import ConvLSTM_AcousticModel
from resnet_lstm_am import ResNetLSTM_AcousticModel
from resnet_am import ResNet_AcousticModel
from vocabulary import VocabularySpecialWords
from tqdm import tqdm

if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    parser = argparse.ArgumentParser(description='Train a simple ConvLSTM acoustic model.')
    parser.add_argument('log_file', help='Path to output log file')
    parser.add_argument('train_dataset', help='Path to processed train dataset file')
    parser.add_argument('valid_dataset', help='Path to processed validation dataset file')
    parser.add_argument('lm_train_dataset', help='Path to processed language model train dataset file')
    parser.add_argument('lm_valid_dataset', help='Path to processed language model train dataset file')
    parser.add_argument('--vocab_unk_rate', help='UNKing rate to use for vocabulary, by default will use true UNK rate based on validation set OOV rate', default=-1.0)
    args = parser.parse_args()
    
    n_epochs = 1000
    train_samples_per_epoch = 40000
    valid_samples_per_epoch = 100
    batch_size = 1
    max_sequence_length = 50
    
    lm_train_dataset = TextDataset(args.lm_train_dataset, max_sequence_length)
    lm_valid_dataset = TextDataset(args.lm_valid_dataset, max_sequence_length)
    
    if args.vocab_unk_rate == -1.0:
        lm_train_dataset.unk_vocabulary_with_true_oov_rate(lm_valid_dataset)
    elif args.vocab_unk_rate > 0:
        lm_train_dataset.unk_vocabulary_with_oov_rate(args.vocab_unk_rate)
    
    train_dataset = SpeechDataset(args.train_dataset, vocabulary=lm_train_dataset.vocabulary)
    valid_dataset = SpeechDataset(args.valid_dataset, vocabulary=lm_train_dataset.vocabulary)
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
        ToTensor()])
    
    train_dataset.set_transform(dataset_transformer)
    valid_dataset.set_transform(dataset_transformer)
    
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=train_samples_per_epoch)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=12, collate_fn=PadSequencesCollater())
    valid_sampler = RandomSampler(valid_dataset, replacement=True, num_samples=valid_samples_per_epoch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0, collate_fn=PadSequencesCollater())
    
    spectogram_num_filters = 40
    
    bottleneck_size = 2**np.arange(3,10)
    learning_rates = 10.**np.arange(-5,-2)
    hyperparameters_tried = set()
    
    logfile_prefix = os.path.splitext(args.log_file)[0]
    logfile_dir = os.path.dirname(args.log_file)
    
    while True:
        random_bottleneck_size = np.random.choice(bottleneck_size)
        random_learning_rate = np.random.choice(learning_rates)
        hyperparameter_combo = (random_learning_rate,random_bottleneck_size)
        if hyperparameter_combo in hyperparameters_tried:
            continue
        hyperparameters_tried.add(hyperparameter_combo)
        
        model_suffix = f'bottleneck_{random_bottleneck_size}_lr_{random_learning_rate}_resnext101'
        model_save_name = f'model_{model_suffix}.pth'
        model_save_path = os.path.join(logfile_dir, model_save_name)
        if os.path.isfile(model_save_path):
            continue
        
        model = ResNet_AcousticModel(spectogram_num_filters, lm_min_word_id, max_word_id, bottleneck_size=random_bottleneck_size, resnet_backbone_name='resnext101')
        model.cuda()
        model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\n\nTotal Parameters: {model_total_params}\n\n' + str(model))
        
        loss_function = nn.CTCLoss(zero_infinity=True)
        optimizer = optim.Adam(model.parameters(), lr=random_learning_rate)
        
        logfile_path = f'{logfile_prefix}_{model_suffix}.txt'
        log_file = open(logfile_path, 'w')
        model_samples_filepath = f'{logfile_prefix}_output_samples_{model_suffix}.txt'
        model_samples_file = open(model_samples_filepath, 'w')
        
        log_file.write(f'Model Total Parameters: {model_total_params}\n')
        log_file.write(f'Epoch #,Train Average Loss,Validation Average Loss\n')
        log_file.flush()
        model.train()
        
        validation_losses = []
        best_validation_loss = np.inf
        
        num_train_steps = len(train_dataloader) * n_epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_steps)
        warmup_scheduler = lr_warmup.UntunedLinearWarmup(optimizer)
        for epoch in range(n_epochs):
            train_number_of_words = 0
            batches_loop = tqdm(train_dataloader)
            total_train_loss = 0
            num_train_batches = 0
            
            for batch_sample in batches_loop:
                model.zero_grad()
                model_inputs = batch_sample['input'].cuda()
                model_targets = batch_sample['target'].cuda()
                model_target_lengths = batch_sample['target_lengths'].cuda()
                
                word_scores = model(model_inputs)
                ctc_loss_input_length_dim = word_scores.size()[0]
                ctc_loss_input_length = torch.LongTensor([ctc_loss_input_length_dim]).repeat(batch_size).cuda()
                
                loss = loss_function(word_scores, model_targets, ctc_loss_input_length, model_target_lengths)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                warmup_scheduler.dampen()
                
                with torch.no_grad():
                    if np.random.uniform(0,1) >= 0.97:
                        targets = model_targets[0,:].cpu().numpy() + lm_min_word_id
                        target_sentence = train_dataset.vocabulary.tokens_to_sentence(targets)
                        model_samples_file.write(f'\n\nTarget Sentence:\n\t{target_sentence}\n\n')
                        model_outputs = torch.argmax(word_scores[:, 0, :], dim=1).cpu().numpy() + lm_min_word_id
                        model_output_sentence = train_dataset.vocabulary.tokens_to_sentence(model_outputs)
                        model_output_sentence = model_output_sentence.replace('[START]','[BLANK]')
                        model_output_sentence = re.sub(r'(\[BLANK\] ?)+', '[BLANK] ', model_output_sentence)
                        model_output_sentence = model_output_sentence.replace('  ', ' ')
                        model_samples_file.write(f'\n\nModel Output Sentence:\n\t{model_output_sentence}\n\n')
                        model_samples_file.flush()
                    total_train_loss += loss
                
                batches_loop.set_description('Train Epoch {}/{}'.format(epoch + 1, n_epochs))
                batches_loop.set_postfix(loss=loss.item())
                num_train_batches += 1
            
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
                    ctc_loss_input_length = torch.LongTensor([ctc_loss_input_length_dim]).repeat(batch_size).cuda()
                    
                    loss = loss_function(word_scores, model_targets, ctc_loss_input_length, model_target_lengths)
                    
                    total_valid_loss += loss
                    
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
