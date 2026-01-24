#!/bin/bash

. env/bin/activate && \
    coverage run -m pytest && \
    coverage html