#!/bin/env bash

if [[ $# -eq 0 ]]; then
  echo "=== running your code ===";
  python3 src/main.py;
elif [[ $1 == "1" ]] ; then
  echo "=== running demo one ===";
  python3 src/demo_one.py;
elif [[ $1 == "2" ]] ; then
  echo "=== running demo two ===";
  python3 src/demo_two.py;
elif [[ $1 == "3" ]] ; then
  echo "=== running demo three ===";
  python3 src/demo_three.py;
else
  echo "please use a valid input";
fi

