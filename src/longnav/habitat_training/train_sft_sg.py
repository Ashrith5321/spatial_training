import os
import sys
import argparse
from pathlib import path

import torch
import torch.nn as nn

#  Automodel loads the VLM
#  Autoprocessor loads the input ouput machinary
from transformers import AutoModelForImageTextToText, AutoProcessor


