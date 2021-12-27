#!/usr/bin/env bash

set -e

if [ "$(id -u)" -ne "0" ]; then
    echo "This script must be executed with root privileges"
    exit 1
fi

apt-get install python-pyaudio libatlas-base-dev
