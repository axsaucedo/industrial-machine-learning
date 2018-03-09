import glob
import sys


def _main():
    pattern = sys.argv[1]
    output_file = sys.argv[2]

    all_unique_chars = set()
    num_processed = 0
    for filename in glob.glob(pattern):
        print(filename)
        with open(filename) as f:
            unique_chars = set(f.read())
        all_unique_chars = all_unique_chars.union(unique_chars)

        num_processed += 1
        if num_processed % 1000 == 0:
            print(f'Current char count: {len(all_unique_chars)}')

    print(f'Final char count: {len(all_unique_chars)}')
    with open(output_file, 'w') as f:
        f.write('\n'.join(sorted(all_unique_chars)))




if __name__ == '__main__':
    _main()
