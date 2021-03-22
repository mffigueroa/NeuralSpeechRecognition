import argparse
import re
import os
from text_tokenizer import tokenize_line

parser = argparse.ArgumentParser(description='Process text file from Gutenberg Dataset to produce a set of corpus sequences.')
parser.add_argument('gutenberg_dir', help='Path to Gutenberg dataset folder containing .txt files')
parser.add_argument('output_path', help='Path to output corpus sequences text file')
parser.add_argument('--max_sequence_length', help='Maximum tokens per sequence', default=50, type=int)
parser.add_argument('--sequence_stride', help='Stride to next token sequence', default=5, type=int)
args = parser.parse_args()

txt_files = os.listdir(args.gutenberg_dir)
txt_files = { file : os.path.join(args.gutenberg_dir, file) for file in txt_files }

with open(args.output_path, 'w', encoding='utf-8') as output_file:
    for file, path in txt_files.items():
        with open(path, 'r', encoding='utf-8') as file_obj:
            current_token_sequence = []
            while True:
                if len(current_token_sequence) < args.max_sequence_length:
                    try:
                        line = file_obj.readline()
                    except UnicodeDecodeError:
                        continue
                    
                    if len(line) < 1 or line is None:
                        break
                    
                    tokens = tokenize_line(line)
                    if len(tokens) < 1:
                        continue
                
                    current_token_sequence.extend(tokens)
                
                if len(current_token_sequence) >= args.max_sequence_length:
                    output_sequence = current_token_sequence[:args.max_sequence_length]
                    assert len(output_sequence) == args.max_sequence_length
                    for token in output_sequence:
                        assert len(token) > 0 and token.isalnum()
                    
                    current_token_sequence = current_token_sequence[args.sequence_stride:]
                    output_csv_row = ','.join(output_sequence)
                    if output_csv_row.count(',') != args.max_sequence_length - 1:
                        import pdb; pdb.set_trace()
                    output_file.write(output_csv_row + '\n')

