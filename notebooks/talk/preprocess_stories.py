from extract_text_from_html import extract_text
import glob
import os
import sys


# Last processed story (in alphabetical order):
# /Users/donaldwhyte/Downloads/gutenberg/A/The Plants of Michigan
# ...
# /Users/donaldwhyte/Downloads/gutenberg/A/The San Rosario Ranch
# ...
# /Users/donaldwhyte/Downloads/gutenberg/A/The Spirit Lake
# ...
# /Users/donaldwhyte/Downloads/gutenberg/A/The Statute of Anne

def _main():
    pattern = sys.argv[1]
    output_dir = sys.argv[2]

    os.mkdir(output_dir)

    all_unique_chars = set()
    num_failed = 0
    files = glob.glob(pattern)
    for filename in files:
        print(filename)
        with open(filename, 'rb') as f:
            b = f.read()
            try:
                contents = b.decode('utf-8')
                text = extract_text(contents)
            except UnicodeDecodeError:
                print(f'\tFAILED')
                num_failed += 1
                continue

            with open(os.path.join(output_dir, os.path.basename(filename)), 'wt') as f:
                f.write(text)

    print(f'{len(files) - num_failed}/{len(files)} succeeded')


if __name__ == '__main__':
    _main()
