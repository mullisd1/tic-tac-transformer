

def boards_to_format(key):
    key = key[1:-1]
    return [int(c)+1 for c in key.split(' ') if c in ['-1', '0', '1']]