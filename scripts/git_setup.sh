#!/bin/bash

# activate git credential store
git config --global --unset credential.helper
git config --global credential.helper store

# git user.name, user.email
read -p "Git user.name: " username
read -p "Git user.email: " email

# Set the global Git configuration for user name and email
git config --global user.name "$username"
git config --global user.email "$email"
