import re

def tokenize_line(line):
    line = line.strip()
    if len(line) < 1:
        return []
    line = re.sub(r'[^a-zA-Z0-9]+', ' ', line)
    line = re.sub(r'\s+', ' ', line)
    line = line.lower()
    
    if line[0] == ' ':
        line = line[1:]
    if len(line) < 1:
        return []
    
    if line[-1] == ' ':
        line = line[:-1]
    if len(line) < 1:
        return []
    
    for i in range(len(line)):
        c = line[i]
        assert c == ' ' or c.isalnum()
        if c == ' ':
            assert i > 0
            if i < len(line) - 1:
                assert line[i-1].isalnum() and line[i+1].isalnum()
    
    tokens = line.split(' ')
    return tokens