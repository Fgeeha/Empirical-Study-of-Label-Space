import re


def normalize_class_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[\/_\-]+', ' ', name)
    name = re.sub(r'[^a-z0-9 ]+', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()
