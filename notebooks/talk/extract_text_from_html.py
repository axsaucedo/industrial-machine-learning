import glob
import re
import sys
from typing import Dict, Mapping, Optional, Tuple

from bs4 import BeautifulSoup
import numpy as np


START_TOKEN = '๏'
_PAGE_NUM_PATTERN = re.compile(
    r'(\[\s*(pg|page)\.?\s*(\d+)\s*?\]|\s(pg|page)\.?\s*(\d+)\s)',
    re.IGNORECASE)


def load_chars_file(
        filename: str) -> Tuple[int, Dict[str, int], Dict[int, str]]:
    with open(chars_file) as f:
        chars = f.read().split('\n')
    vocab_size = len(chars)
    ix_to_char = {ix:char for ix, char in enumerate(chars)}
    char_to_ix = {char: ix for ix, char in enumerate(chars)}
    return vocab_size, char_to_ix, ix_to_char


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, 'lxml')
    # kill all script and style elements
    for script in soup(['script', 'style']):
        script.extract()    # rip it out
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split('  '))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return _clean_text(text)


def _clean_text(text: str) -> str:
    lines = [_clean_line(line) for line in text.split('\n')]
    cleaned = '\n'.join(line for line in lines if line is not None)
    return START_TOKEN + cleaned


def _clean_line(line: str) -> Optional[str]:
    if _PAGE_NUM_PATTERN.match(line):
        return None
    else:
        return line


def generate_text(model,
                  vocab_size: int,
                  ix_to_char: Mapping[int, str],
                  char_to_ix: Mapping[str, int],
                  length: int) -> str:
    ix = char_to_ix[START_TOKEN]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, vocab_size))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end='')
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)


if __name__ == '__main__':
    pattern = sys.argv[1]
    for filename in glob.glob(pattern):
        html = open(filename).read()
        print(filename)
        print(extract_text(html))
        print('-' * 60)
