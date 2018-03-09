import argparse

from model import RNNTextModel
from text import ProgressBar, read_data_files


def _main():
    args = _parse_args()

    data = read_data_files(args.files, validation=True)

    sequence_length = 30
    batch_size = 200
    display_every_n_batches = 50
    progress = ProgressBar(
        display_every_n_batches,
        size_of_bar_in_chars=111 + 2,
        header_message=f'Training on next {display_every_n_batches} batches')
    def on_step_complete(step: int):
        # After each step, update the display progress bar to reflect a step
        # has been taken. After a certain number of steps, we reset the bar.
        max_steps = display_every_n_batches * batch_size * sequence_length
        should_reset = (step % max_steps == 0)
        progress.step(reset=should_reset)

    model = RNNTextModel(
        sequence_length=sequence_length,
        batch_size=batch_size,
        gru_internal_size=512,
        num_hidden_layers=3,
        stats_log_dir='logs')
    model.run_training_loop(
        data,
        num_epochs=10000,
        learning_rate=0.001,
        display_every_n_batches=50,
        checkpoint_dir='checkpoints',
        on_step_complete=on_step_complete,
        should_stop=lambda step: False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Helper program for training RNNs on textual data')
    parser.add_argument(
        '-f', '--files', type=str,
        help='Glob pattern of files to train on',
        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    _main()
