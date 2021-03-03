import os

path = r'C:\Data\NLP\Gutenberg\train'
search_string = 'northanger'
search_string = search_string.lower()

for filename in os.listdir(path):
    filepath = os.path.join(path, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        line_number = 1
        while True:
            try:
                line = file.readline()
            except UnicodeDecodeError:
                continue
            if len(line) < 1 or line is None:
                break
            line = line.lower()
            if search_string in line:
                print(f'{filename} - Line {line_number}')
            line_number += 1

