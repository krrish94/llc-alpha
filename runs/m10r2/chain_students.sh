#!/bin/zsh
# Waits for the round-2 bases of one kind to finish (final.pt of all three seeds), then launches that kind's students.
cd "$(dirname "$0")/../.."  # the repository root
kind=$1
until [ -f runs/m10r2/${kind}_ff-s0/final.pt ] && [ -f runs/m10r2/${kind}_ff-s1/final.pt ] && [ -f runs/m10r2/${kind}_ff-s2/final.pt ]; do sleep 60; done
sleep 20
./runs/m10r2/launch_students.sh $kind
