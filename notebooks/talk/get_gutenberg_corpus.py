import requests
from typing import List
import uuid


def _main():
    output_dir == str(uuid.uuid4())
    print(f'Output directory: {output_dir}')

    book_ids = _get_all_books_by_language('en')
    print(f'Downloading contents of {len(book_ids)} books')
    return

    num_downloaded = 0
    for book in book_ids:
        contents = _get_book()
        with open(os.path.join(output_dir, str(book_id)), 'w') as f:
            f.write(contents)

        num_downloaded += 1
        if num_downloaded % 100 == 0:
            print(f'Downloaded {num_downloaded}/{len(book_ids)} books')


def _get_book(id: int) -> str:
    response = requests.get(f'https://gutenbergapi.org/texts/{id}/body')
    _check_for_error(response)
    return response.text


def _get_all_books_by_language(language: str) -> List[str]:
    response = requests.get(
        f'https://gutenbergapi.org/search/{TODO}')
    _check_for_error(response)
    return [text['text_id'] for text in response.json()['texts']]


def _check_for_error(response):
    if response.status_code != 200:
        raise RuntimeError(f'Bad response: {response}')


if __name__ == '__main__':
    _main()
