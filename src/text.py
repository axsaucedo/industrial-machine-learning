import glob
import logging
import os
from typing import Generator, List, Tuple

import numpy as np

_logger = logging.getLogger(__name__)


# Size of the alphabet that we work with. This contains alphanumeric characters
# and some additional special ones.
ALPHABET_SIZE = 98


# 3-tuple containing training text sequence, test text sequence and a file
# index for the training text sequence.
SequenceData = Tuple[List[int], List[int], List[dict]]


def read_data_files(glob_pattern: str, validation: bool=True) -> SequenceData:
    '''Read data files found with the specified `glob_pattern`.

    If `validation` is set to `True`, then set aside last file as validation
    data.
    '''
    encoded_text = []
    file_index = []
    files = glob.glob(glob_pattern, recursive=True)
    for fname in files:
        with open(fname, 'r') as f:
            start = len(encoded_text)
            encoded_text.extend(encode_text(f.read()))
            end = len(encoded_text)
            file_index.append({
                'start': start,
                'end': end,
                'name': os.path.basename(fname)
            })

    if len(file_index) == 0:
        _logger.info(
            f'No training data found in files matching {glob_pattern}')
        return [], [], []

    total_len = len(encoded_text)

    # For validation, use roughly 90K of text, but no more than 10% of the
    # entire text. Also no more than 1 data_file in 5, meaning we provide no
    # validation file if we have 5 files or fewer.
    validation_len = 0
    num_files_in_10percent_of_chars = 0
    for data_file in reversed(file_index):
        validation_len += data_file['end'] - data_file['start']
        num_files_in_10percent_of_chars += 1
        if validation_len > total_len // 10:
            break

    validation_len = 0
    num_files_in_first_90kb = 0
    for data_file in reversed(file_index):
        validation_len += data_file['end'] - data_file['start']
        num_files_in_first_90kb += 1
        if validation_len > 90 * 1024:
            break

    # 20% of the data_files is how many data_files ?
    num_files_in_20percent_of_files = len(file_index) // 5

    # pick the smallest
    num_files_in_training = min(
        num_files_in_10percent_of_chars,
        num_files_in_first_90kb,
        num_files_in_20percent_of_files)

    if num_files_in_training == 0 or not validation:
        training_chars_cutoff = len(encoded_text)
    else:
        training_chars_cutoff = file_index[-num_files_in_training]['start']
    training_text = encoded_text[:training_chars_cutoff]
    validation_text = encoded_text[training_chars_cutoff:]

    return training_text, validation_text, file_index


def encode_text(text: str) -> List[int]:
    """Encode given `text` as a list of integers suitable for NNs."""
    return [_convert_from_alphabet(ord(ch)) for ch in text]


def decode_to_text(chars: List[int], avoid_tab_and_lf: bool=False) -> str:
    """Decode given `chars` integer codes to an ASCII string."""
    return ''.join(
        [_convert_to_alphabet(ch, avoid_tab_and_lf) for ch in chars])


# A training batch consists of (X, Y, epoch), which are the input feature
# matrix, expected output matrix and epoch number respectively.
Batch = Tuple[np.matrix, np.matrix, int]


def rnn_minibatch_generator(data: List[int],
                            batch_size: int,
                            sequence_length: int,
                            num_epochs: int) -> Generator[Batch, None, None]:
    """Construct a generator that yield training batches.

    Divides `data` into batches of sequences so that all the sequences in
    one batch continue in the next batch. This is a generator that will keep
    returning batches until the input data has been seen `num_epochs` times.

    Sequences are continued even between epochs, apart from one, the one
    corresponding to the end of `data`. The remainder at the end of `data` that
    does not fit in an full batch is ignored.

    `batch_size` is the number of sequences to put in a each yielded `Batch`.

    `sequence_length` is the length of each training sequence (in chars). This
    must match the unroll size of the RNN you're feeding these batches into.
    """
    data = np.array(data)
    data_len = data.shape[0]
    # Using (data_len-1) because we must provide for the sequence shifted by
    # one too.
    num_batches = (data_len - 1) // (batch_size * sequence_length)
    if num_batches < 1:
        raise RuntimeError(
            'Not enough data, even for a single batch. Try using a smaller '
            'batch_size.')
    rounded_data_len = num_batches * batch_size * sequence_length
    xdata = np.reshape(
        data[0:rounded_data_len],
        [batch_size, num_batches * sequence_length])
    ydata = np.reshape(
        data[1:rounded_data_len + 1],
        [batch_size, num_batches * sequence_length])

    for epoch in range(num_epochs):
        for batch in range(num_batches):
            x = xdata[:, batch * sequence_length:(batch + 1) * sequence_length]
            y = ydata[:, batch * sequence_length:(batch + 1) * sequence_length]
            # If we've reached the end of the training data, we wrap around to
            # the start of the training data. We don't stop here because we
            # might have more epochs to run.
            x = np.roll(x, -epoch, axis=0)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch


class ProgressBar:
    """Text mode progress bar.

    Usage:
            p = Progress(30)
            p.step()
            p.step()
            p.step(start=True) # to restart form 0%

    The progress bar displays a new header at each restart.
    """
    def __init__(self,
                 steps_to_finish: int,
                 size_of_bar_in_chars: int=100,
                 header_message: str=""):
        self._steps_to_finish = steps_to_finish
        self._size_of_bar_in_chars = size_of_bar_in_chars
        self._header_message = header_message

        # Use `()` to get the iterator from the generator
        self.p = self.__start_progress(self._steps_to_finish)()
        self._header_printed = False

    def step(self, reset: bool=False):
        if reset:
            self.__init__(
                self._steps_to_finish,
                self._size_of_bar_in_chars,
                self._header_message)
        if not self._header_printed:
            self.__print_header()
        next(self.p)

    def __print_header(self):
        print()
        format_string = (
            "0%{: ^" + str(self._size_of_bar_in_chars - 6) + "}100%")
        print(format_string.format(self._header_message))
        self._header_printed = True

    def __start_progress(self,
                         steps_to_finish: int) -> Generator[int, None, None]:
        def print_progress():
            # Bresenham's algorithm. Yields the number of dots printed.
            # This will always print 100 dots in max invocations.
            dx = self._steps_to_finish
            dy = self._size_of_bar_in_chars
            d = dy - dx
            for x in range(self._steps_to_finish):
                k = 0
                while d >= 0:
                    print('=', end="", flush=True)
                    k += 1
                    d -= dx
                d += dy
                yield k

        return print_progress


def print_learning_learned_comparison(X: np.array,
                                      Y: np.array,
                                      losses: : np.array,
                                      file_index: List[dict],
                                      batch_loss: float,
                                      batch_accuracy: float,
                                      epoch_size: int,
                                      index: int,
                                      epoch: int):
    """Display utility for printing learning statistics.

    Displays the input text and the predicted text for a number of batches.
    This is useful for humans to see at a glance how well the network is doing.
    """
    print()
    # epoch_size in number of batches
    batch_size = X.shape[0]  # batch_size in number of sequences
    sequence_len = X.shape[1]  # sequence_len in number of characters
    start_index_in_epoch = index % (epoch_size * batch_size * sequence_len)
    for k in range(batch_size):
        index_in_epoch = index % (epoch_size * batch_size * sequence_len)
        decx = decode_to_text(X[k], avoid_tab_and_lf=True)
        decy = decode_to_text(Y[k], avoid_tab_and_lf=True)
        fname = _find_file(index_in_epoch, file_index)
        # min 10 and max 40 chars
        formatted_fname = '{: <10.40}'.format(fname)
        epoch_string = '{:4d}'.format(index) + ' (epoch {}) '.format(epoch)
        loss_string = 'loss: {:.5f}'.format(losses[k])
        print_string = epoch_string + formatted_fname + ' │ {} │ {} │ {}'
        print(print_string.format(decx, decy, loss_string))
        index += sequence_len
    # box formatting characters:
    # │ \u2502
    # ─ \u2500
    # └ \u2514
    # ┘ \u2518
    # ┴ \u2534
    # ┌ \u250C
    # ┐ \u2510
    format_string = f'└{{:─^{len(epoch_string)}}}'
    format_string += f'{{:─^{len(formatted_fname)}}}'
    format_string += f'┴{{:─^{len(decx) + 2}}}'
    format_string += f'┴{{:─^{len(decy) + 2}}}'
    format_string += f'┴{{:─^{len(loss_string)}}}┘'
    footer = format_string.format(
        'INDEX', 'BOOK NAME', 'TRAINING SEQUENCE',
        'PREDICTED SEQUENCE', 'LOSS')
    print(footer)
    # print statistics
    batch_index = start_index_in_epoch // (batch_size * sequence_len)
    batch_string = f'batch {batch_index}/{epoch_size} in epoch {epoch},'
    stats = 'f{: <28} batch loss: {:.5f}, batch accuracy: {:.5f}'.format(
        batch_string, batch_loss, batch_accuracy)
    print()
    print(f'TRAINING STATS: {stats}')


def _find_file(char_index: int, file_index: List[dict]) -> str:
    return next(
        f['name'] for f in file_index
        if (f['start'] <= char_index < f['end']))


def _convert_from_alphabet(ch: str) -> int:
    """Encode character `ch` to an integer.

    Specification of the supported alphabet (subset of ASCII-7):
        * 10 line feed LF
        * 32-64 numbers and punctuation
        * 65-90 upper-case letters
        * 91-97 more punctuation
        * 97-122 lower-case letters
        *  123-126 more punctuation
    """
    if ch == 9:
        return 1
    if ch == 10:
        return 127 - 30  # LF
    elif 32 <= ch <= 126:
        return ch - 30
    else:
        return 0  # unknown


def _convert_to_alphabet(ch: int, avoid_tab_and_lf: bool=False) -> chr:
    """Decode an encoded character `ch` to a string chartacter.

    What each input integer will be converted to:
        * 0 = unknown (ascii char 0)
        * 1 = tab
        * 2 = space
        * 2 to 96 = 36 to 126 ASCII codes
        * 97 = LF (linefeed)
    """
    return chr(_convert_to_alphabet_impl(ch, avoid_tab_and_lf))


def _convert_to_alphabet_impl(ch: int, avoid_tab_and_lf: bool=False) -> int:
    if ch == 1:
        return 32 if avoid_tab_and_lf else 9  # space instead of TAB
    if ch == 127 - 30:
        return 92 if avoid_tab_and_lf else 10  # \ instead of LF
    if 32 <= ch + 30 <= 126:
        return ch + 30
    else:
        return 0  # unknown
