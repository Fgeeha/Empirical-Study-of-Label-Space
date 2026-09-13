import re


def normalize_class_name(name: str) -> str:
    name = name.lower()

    # ___, _, -, / → space
    name = re.sub(r'[\/_\-]+', ' ', name)

    # remove non-alphanumeric
    name = re.sub(r'[^a-z0-9 ]+', '', name)

    # collapse spaces
    name = re.sub(r'\s+', ' ', name)

    return name.strip()
