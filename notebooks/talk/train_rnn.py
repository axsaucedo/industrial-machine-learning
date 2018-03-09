from functools import partial
import sys
from typing import Generator, List, Tuple

from keras.layers import Activation, Dense, LSTM, TimeDistributed
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot

from extract_text_from_html import extract_text, generate_text, load_chars_file


_HIDDEN_DIMENSIONS = 10
_NUM_HIDDEN_LAYERS = 2
_GENERATED_TEXT_LENGTH = 512


def _main():
    dataset_dir = sys.argv[1]
    chars_file = sys.argv[2]
    output_model_file = sys.argv[3]

    # Map all possible characters to numbers, which play nicer with NNs.
    vocab_size, ix_to_char, char_to_ix = load_chars_file(chars_file)

    # Build deep network
    model = Sequential()
    model.add(LSTM(
        _HIDDEN_DIMENSIONS,
        input_shape=(None, vocab_size),
        return_sequences=True))
    for i in range(_NUM_HIDDEN_LAYERS - 1):
        model.add(LSTM(_HIDDEN_DIMENSIONS, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # Construct training dataset generator
    sample_generator = partial(_book_contents_generator, sys.argv[1])

    # Train the model
    try:
        nb_epoch = 0
        while True:
            print('\n\n')
            model.fit_generator(
                sample_generator, steps_per_epoch=64, epochs=1)
            nb_epoch += 1
            generate_text(model, _GENERATED_TEXT_LENGTH)
            if nb_epoch % 10 == 0:
                model.save_weights(
                    f'checkpoint_{_HIDDEN_DIMENSIONS}_epoch_{nb_epoch}.hdf5')
    except KeyboardInterrupt:
        model.save_weights(
            f'checkpoint_{_HIDDEN_DIMENSIONS}_epoch_{nb_epoch}.hdf5')

    # Save learned parameters to a file
    model.save(output_model_file)


def _book_contents_generator(
        glob_pattern: str) -> Generator[Tuple[List[str], List[str]],
                                        None,
                                        None]:
    files = glob.glob(glob_pattern)
    index = 0
    while True:
        with open(files[index]) as f:
            contents = f.read()
        yield extract_text(contents)

        index += 1
        # Wrap around to first file if all files have bene processed. Keras
        # expects sample generators to generate samples forever.
        if index == len(files):
            index = 0


if __name__ == '__main__':
    _main()
