#!/bin/bash

# Prompt the user for input
read -p "Please provide input image folder: " user_input

granny -i cli --analysis blush --input $user_input
