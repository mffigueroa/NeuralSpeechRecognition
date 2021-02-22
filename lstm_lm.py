import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTM_LanguageModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, min_output_word_id, max_word_id, num_lstm_layers=None):
        super(LSTM_LanguageModel, self).__init__()
        if num_lstm_layers is None:
            num_lstm_layers = 1
        self.num_lstm_layers = num_lstm_layers
        self.hidden_dim = hidden_dim
        self.max_word_id = max_word_id
        self.min_output_word_id = min_output_word_id
        self.word_embeddings = nn.Embedding(self.max_word_id + 1, embedding_dim, padding_idx=0) # inputs range from [0, self.max_word_id]
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_lstm_layers)
        
        # The linear layer that maps from hidden state space to word space
        self.hidden2word = nn.Linear(hidden_dim, self.max_word_id - self.min_output_word_id + 1) # outputs range from [self.min_output_word_id, self.max_word_id]
    
    def forward(self, sentence, previous_state=None):
        embeds = self.word_embeddings(sentence)
        if previous_state is None:
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, last_lstm_state = self.lstm(embeds, previous_state)
        word_space = self.hidden2word(lstm_out.view(-1, self.hidden_dim))
        batch_size = lstm_out.size()[0]
        sequence_length = lstm_out.size()[1]
        word_scores = F.log_softmax(word_space.view(batch_size, sequence_length, -1), dim=2)
        if previous_state is None:
            return word_scores
        else:
            return word_scores, previous_state
    
    def get_log_word_distribution_for_next_input(self, input_token, lstm_state):
        input = np.array([input_token]).astype(np.int64)
        input = torch.from_numpy(input).cuda()
        input = torch.unsqueeze(input, 0)
        model_output, lstm_state = self(input, lstm_state)
        model_output = torch.squeeze(model_output)
        model_output = model_output.cpu().numpy()
        return model_output, lstm_state
    
    def sample_from_model(self, max_length, beam_width=None):
        from vocabulary import VocabularySpecialWords
        
        beam_width = beam_width or 10
        vocab_word_ids = np.arange(self.min_output_word_id, self.max_word_id+1, dtype=np.int64)
        input_token = VocabularySpecialWords.START.value
        beam_sentences = [[input_token]]
        beam_sentence_log_probabilities = [0]
        sentence_samples_per_iteration = 100
        with torch.no_grad():
            lstm_hidden_state = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim).cuda()
            lstm_cell_state = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim).cuda()
            lstm_state = lstm_hidden_state, lstm_cell_state
            beam_states = [lstm_state]
            
            for _ in range(max_length):
                sentence_log_probabilities = np.empty((0,))
                next_token_samples = np.empty((0,), dtype=np.int64)
                beam_index_for_sample = np.empty((0,), dtype=np.int64)
                for beam_index in range(len(beam_states)):
                    sentence_last_token = beam_sentences[beam_index][-1]
                    if sentence_last_token == VocabularySpecialWords.STOP.value:
                        continue
                    
                    lstm_state = beam_states[beam_index]
                    word_log_probabilities, lstm_state = self.get_log_word_distribution_for_next_input(sentence_last_token, lstm_state)
                    beam_sentence_log_probability = beam_sentence_log_probabilities[beam_index]
                    beam_sentence_log_probabilities = beam_sentence_log_probability + word_log_probabilities
                    beam_states[beam_index] = lstm_state
                    sentence_log_probabilities = np.hstack([sentence_log_probabilities, beam_sentence_log_probabilities])
                    next_token_samples = np.hstack([next_token_samples, vocab_word_ids])
                    beam_index_for_sample = np.hstack([beam_index_for_sample, np.ones_like(next_token_samples) * beam_index])
                
                sentence_indices_to_sample_from = np.argsort(-sentence_log_probabilities)[:sentence_samples_per_iteration]
                
                sentence_sampling_probability = sentence_log_probabilities[sentence_indices_to_sample_from]
                sentence_sampling_probability = np.exp(sentence_sampling_probability)
                sentence_sampling_probability /= np.sum(sentence_sampling_probability)
                
                num_sentences_to_sample_from = sentence_indices_to_sample_from.shape[0]
                sampled_sentence_subset_indices = np.random.choice(np.arange(num_sentences_to_sample_from), beam_width, p=sentence_sampling_probability, replace=False)
                probabilities_for_sampled_sentences = sentence_sampling_probability[sampled_sentence_subset_indices]
                sampled_sentence_indices = sentence_indices_to_sample_from[sampled_sentence_subset_indices]
                
                next_beam_states = []
                next_beam_sentences = []
                next_beam_sentence_log_probabilities = []
                for sentence_index in sampled_sentence_indices:
                    sentence_next_token = next_token_samples[sentence_index]
                    sentence_beam_index = beam_index_for_sample[sentence_index]
                    sentence = beam_sentences[sentence_beam_index]
                    sentence_lstm_next_state = beam_states[sentence_beam_index]
                    sentence = sentence + [sentence_next_token]
                    sentence_log_probability = sentence_log_probabilities[sentence_index]
                    next_beam_states.append(sentence_lstm_next_state)
                    next_beam_sentences.append(sentence)
                    next_beam_sentence_log_probabilities.append(sentence_log_probability)
                
                beam_states = next_beam_states
                beam_sentences = next_beam_sentences
                beam_sentence_log_probabilities = next_beam_sentence_log_probabilities
                
                max_probability_sampled_sentence_index = np.argmax(beam_sentence_log_probabilities)
                last_token_for_max_probability_sentence = beam_sentences[max_probability_sampled_sentence_index][-1]
                if last_token_for_max_probability_sentence == VocabularySpecialWords.STOP.value:
                    if beam_sentences[max_probability_sampled_sentence_index][-1] != VocabularySpecialWords.STOP.value:
                        import pdb; pdb.set_trace()
                    return beam_sentences[max_probability_sampled_sentence_index]
        
        max_log_probability_sentence = beam_sentences[np.argmax(beam_sentence_log_probabilities)]
        return max_log_probability_sentence