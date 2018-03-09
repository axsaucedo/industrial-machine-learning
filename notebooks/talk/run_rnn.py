import sys
from typing import Mapping

from keras.models import load_model

from extract_text_from_html import generate_text, load_chars_file, START_TOKEN


def _main():
    # Parse command line args
    model_file = sys.argv[1]
    chars_file = sys.argv[2]
    chars_to_generate = int(sys.argv[2])

    # Load model and char<->int mappings from files
    model = load_model(model_file)
    vocab_size, ix_to_char, char_to_ix = load_chars_file(chars_file)

    # Use network to generate characters using chosen start char
    print(generate_text(
        model,
        vocab_size,
        ix_to_char,
        char_to_ix
        chars_to_generate))


if __name__ == '__main__':
    _main()
