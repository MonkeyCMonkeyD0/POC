#!/bin/bash

dataset_id="1Ts4xRnxOCklxOkuiOeKEUIzRHv4whL09"

google_drive_link="https://drive.google.com/uc?export=download&id=$dataset_id"

curl -LO $google_drive_link
echo $google_drive_link