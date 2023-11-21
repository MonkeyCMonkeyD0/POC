#!/bin/bash

dataset_id="1Nn7P7PBXfYpOFL3E3A8b7yhF9kqlV6n4"

google_drive_link="https://drive.google.com/uc?export=download&id=$dataset_id&confirm=t"

curl -L $google_drive_link > dataset.zip

unzip dataset.zip -d ./
