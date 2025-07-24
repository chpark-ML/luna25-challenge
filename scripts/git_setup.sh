#!/bin/bash

# unset git credential store
# git config --global --unset credential.helper
# git config --global --unset user.name
# git config --global --unset user.email

git config credential.helper store

# git user.name, user.email
read -p "Git user.name: " username
read -p "Git user.email: " email

# Set the global Git configuration for user name and email
git config user.name "$username"
git config user.email "$email"

# Check all settings applied in the current directory
git config --list --show-origin

