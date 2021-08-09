import torchvision
import torch
from dataloader import SewageDataset

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

def save_checkpoint(state, filename):
    logging.info("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    logging.info("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
