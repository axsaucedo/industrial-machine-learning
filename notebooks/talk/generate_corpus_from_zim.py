import requests
from typing import List
import sys
from zimpy import ZimFile



def _main():
    filename = sys.argv[1]
    output_file = sys.argv[2]

    zim_file = ZimFile(filename)
    try:
        for article in zim_file.articles():
            index = article['index']
            raw_content = zim_file.get_article_by_index(index)[0]
    finally:
        zim_file.close()


if __name__ == '__main__':
    _main()
