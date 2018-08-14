#!/usr/bin/python

import os

def parse_filename(filename, key):
    parts = os.path.split(filename)[0].split("/")[-1].split(",")
    for part in parts:
        k,v = part.split("=")
        if k == key:
            return v

    raise KeyError("legend key not found in filename")
