#!/bin/bash

dataset_id="1CIhofqCNqMOlFyPqg5gAAMO3UoxNe8qy"

google_drive_link="https://drive.google.com/uc?export=download&id=$dataset_id&confirm=t"

curl -L $google_drive_link > dataset.zip

unzip dataset.zip -d ./
