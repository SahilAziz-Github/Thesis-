#!/usr/bin/env python3
# Author: Mohammed Sazid Al Rashid

import requests
import csv
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning


# Suppress the InsecureRequestWarning since we use verify=False parameter.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

URLS_FILE = "urls.csv"
HTML_SAVE_DIRECTORY = "crawled_pages"


"""
{
    "tag": "tag name",
    "extra": {
        # Extra attribute identifiers such as 'input[type='button']
    },
    "exclude": [
        # List of attributes to ignore while parsing
    ]
}
"""
TAG_LIST = [
    { "name": "button",   "attrs": {},                 "exclude": None     },
    { "name": "input",    "attrs": {"type": "button"}, "exclude": None     },
    { "name": "input",    "attrs": {"type": "submit"}, "exclude": None     },
    { "name": "input",    "attrs": {"type": "text"},   "exclude": None     },
    { "name": "a",        "attrs": {},                 "exclude": ["href"] },
    { "name": "img",      "attrs": {},                 "exclude": ["src"]  },
    { "name": "textarea", "attrs": {},                 "exclude": None     },
]


def crawl_webpage(rank, url):
    print(f"Crawling #{rank}: {url}...")

    schemas = ["http://", "https://", "http://www.", "https://www."]
    r = None
    for schema in schemas:
        try:
            r = requests.get(schema + url, verify=False, timeout=60)

            html_content = add_custom_attribute(r.content)

            filename = Path(HTML_SAVE_DIRECTORY) / Path(f"{rank}_{url}.html")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(str(html_content))

            break
        except:
            # Skip if anything fails for now.
            continue


def add_custom_attribute(html_content):
    """
    Adds custom attribute:
    data-aiattr="0"
    data-aiattr="1"
    ...
    to target elements.
    """
    try:
        tree = BeautifulSoup(html_content, "lxml")
    except:
        tree = BeautifulSoup(html_content, "html.parser")

    body = tree.body

    element_idx = 0
    for target in TAG_LIST:
        try:
            elements = body.find_all(
                name=target["name"],
                attrs=target["attrs"],
                recursive=True,
                limit=5,            # 5 elements per tag type
            )

            # Add an index to all the elements
            for elm in elements:
                elm["data-aiattr"] = element_idx
                element_idx += 1
        except:
            # Skip for failing elements
            pass

    return body


def main():
    if len(sys.argv) == 3:
        start = int(sys.argv[1])
        batch = int(sys.argv[2])
    else:
        print(
"Please run:\n> python crawler.py start batch\n\n\
Where start and batch = any number. Example:\n\n\
> python crawler.py 0 500")
        return

    try:
        with open("urls.csv") as csvfile:
            url_reader = csv.reader(csvfile)

            try:
                # Skip the first `start` lines
                for _ in range(start):
                    next(url_reader)
            except:
                print("Failed to seek first `start` lines.")
                return

            for i, row in enumerate(url_reader):
                try:
                    if i >= batch:
                        break
                    rank, url = row
                    crawl_webpage(rank, url)
                    i += 1
                except:
                    pass
    except:
        print("Failed to open urls.csv file")


if __name__ == "__main__":
    main()
