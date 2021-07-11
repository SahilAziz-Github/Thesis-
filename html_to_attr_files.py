from pathlib import Path
import glob
import csv
from itertools import chain, combinations
from bs4 import BeautifulSoup
from joblib import Parallel, delayed

def powerset(iterable, max_len=2):
    """
    Gives the powerset of the given iterable
    """
    # https://stackoverflow.com/a/1482316/1941132
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return [i for i in chain.from_iterable(combinations(s, r) for r in range(len(s)+1)) if len(i) <= max_len and len(i) > 0]


def read_files():
    CRAWLED_PAGES_GLOB = Path("./crawled_pages/*.html")
    for filepath in glob.glob(str(CRAWLED_PAGES_GLOB)):
        process_file(filepath)

    # Parallel(n_jobs=4) \
    #     (delayed(process_file)(filepath) \
    #         for filepath in glob.glob(str(CRAWLED_PAGES_GLOB))
    #     )


def process_file(filepath):
    try:
        f = open(filepath, "r", encoding="utf-8")
        html_content = f.read()

        tree = BeautifulSoup(html_content, "lxml")
        body = tree.body

        read_to_csv(body)
    except:
        pass


def read_to_csv(body):
    # Remove unnecessary tags/elements, this should make the parsing faster.
    remove_tags = (
        "script",
        "iframe",
        "style",
    )

    for tag_name in remove_tags:
        for s in body.select(tag_name):
            s.decompose()


    # Collect elements
    elements = []
    for i in range(30):
        el = body.find(attrs={"data-aiattr": i})
        if el:
            elements.append(el)


    # Skip, if we didn't find any elements
    if len(elements) == 0:
        return


    def convert_attrs_to_str(attrs):
        attr_str = ""
        for k in attrs:
            attr_str = f'{attr_str}{k}="{attrs[k]}" '
        return attr_str.strip()


    def find_best_attr_combination(el, attrs):
        attr_powerset = powerset(attrs)
        result = []
        for i in attr_powerset:
            attr_dict = {}
            for k in i:
                attr_dict.update(k)

            elements_found = body.find_all(attrs=attr_dict, recursive=True)
            # print(len(elements_found), attr_dict)

            result.append((len(elements_found) == 1, convert_attrs_to_str(attr_dict),))

        return result


    # Collect element attributes except the ones from the exclusion list.
    exclusion_list = (
        "data-aiattr",
        "href",
        "src",
    )

    # Attributes converted to strings
    attrs = []

    for el in elements:
        cur_attr = []
        for key in el.attrs:
            if key in exclusion_list:
                continue

            key = key
            val = el.attrs[key]

            if isinstance(val, list):
                val = " ".join(val)

            val = val.replace("\n", " ")

            if len(val) > 0:
                cur_attr.append({key: val})

        best_attrs = find_best_attr_combination(el, cur_attr)
        attrs += best_attrs
        # print(best_attrs)

    # print(len(attrs), attrs)
    # print(len(attrs))

    ones   = 0
    zeroes = 0

    valid_dir   = Path.cwd() / "train" / "valid"
    invalid_dir = Path.cwd() / "train" / "invalid"

    try:
        valid_dir.mkdir(parents=True, exist_ok=True)
        invalid_dir.mkdir(parents=True, exist_ok=True)
    except:
        print("Data directory already exists.")


    try:
        next_valid_index = int(max(
            valid_dir.glob("*.txt"),
            key=lambda x: int(x.stem)
        ).stem) + 1
    except:
        next_valid_index = 0

    try:
        next_invalid_index = int(max(
            invalid_dir.glob("*.txt"),
            key=lambda x: int(x.stem)
        ).stem) + 1
    except:
        next_invalid_index = 0

    for i in attrs:
        label = i[0]
        data  = i[1]

        if label == True:
            filename = valid_dir / f"{next_valid_index}.txt"
            next_valid_index += 1
            ones += 1
        else:
            filename = invalid_dir / f"{next_invalid_index}.txt"
            next_invalid_index += 1
            zeroes += 1

        with open(filename, "w") as f:
            f.write(data)

    print(ones, zeroes)


if __name__ == "__main__":
    read_files()
