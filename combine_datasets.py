import os
import joblib

rootdir = r'C:\Data\NLP\LibriSpeech'
filenames = [ 'train-clean-100_mfcc.pickle', 'train-clean-360_mfcc_13.pickle', 'train-other-500_mfcc_13.pickle' ]

all_spectrograms = []
all_transcriptions = []
for file in filenames:
    filepath = os.path.join(rootdir, file)
    data = joblib.load(open(filepath, 'rb'))
    all_spectrograms += data['audio_spectrograms']
    all_transcriptions += data['transcription_tokens']
    
final_data = { 'audio_spectrograms' : all_spectrograms, 'transcription_tokens' : all_transcriptions }

with open(os.path.join(rootdir, 'train_mfcc_13.pickle'), 'wb') as final_file:
    joblib.dump(final_data, final_file)

