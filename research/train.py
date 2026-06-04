#!/usr/bin/env python3
import argparse
import json
import os
from time import time

import torch

from steganogan import SteganoGAN
from steganogan.critics import BasicCritic
from steganogan.decoders import DenseDecoder
from steganogan.encoders import BasicEncoder, DenseEncoder, ResidualEncoder
from steganogan.loader import DataLoader


def main():
    torch.manual_seed(42)
    # Convertiamo subito in stringa per evitare errori in os.path.join
    timestamp = str(int(time())) 

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--encoder', default="basic", type=str) # Questo parametro verrà ora ignorato dal caricamento
    parser.add_argument('--data_depth', default=1, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--dataset', default="div2k", type=str)
    parser.add_argument('--output', default=False, type=str)
    args = parser.parse_args()

    train = DataLoader(os.path.join("data", args.dataset, "train"), shuffle=True)
    validation = DataLoader(os.path.join("data", args.dataset, "val"), shuffle=False)

    # Sostituiamo l'inizializzazione da zero con il caricamento del modello
    print("Caricamento del modello pre-addestrato per il fine-tuning...")
    steganogan = SteganoGAN.load("dense", cuda=True)
    
    # Reimpostiamo le directory di log per il nuovo addestramento
    steganogan.log_dir = os.path.join('models', timestamp)
    os.makedirs(steganogan.log_dir, exist_ok=True)
    steganogan.samples_path = os.path.join(steganogan.log_dir, 'samples')
    os.makedirs(steganogan.samples_path, exist_ok=True)
    
    # Forza la reinizializzazione degli ottimizzatori per prendere il nuovo learning rate ridotto
    steganogan.critic_optimizer = None
    
    # Salvataggio della configurazione
    with open(os.path.join("models", timestamp, "config.json"), "wt") as fout:
        fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))

    # Avvio del Fine-Tuning
    steganogan.fit(train, validation, epochs=args.epochs)
    
    # Salvataggio dei pesi finali
    steganogan.save(os.path.join("models", timestamp, "weights.steg"))
    if args.output:
        steganogan.save(args.output)

if __name__ == '__main__':
    main()