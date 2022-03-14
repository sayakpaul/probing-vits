# USAGE
# python train.py --classifier class --position learn
# python train.py --classifier class --position sincos
# python train.py --classifier gap --position learn
# python train.py --classifier gap --position sincos

# import the necessary packages
import tensorflow as tf
import argparse
from vit import (
    get_config,
    ViTClassifier,
)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classifier", required=True,
    help="token or gap for the vit representation")
ap.add_argument("-p", "--position", required=True,
    help="learn or sincos for positional embedding")
args = vars(ap.parse_args())

# build the models
vit_b16_config = get_config()
with vit_b16_config.unlocked():
    vit_b16_config.pos_emb_mode = args["position"]
    vit_b16_config.classifier = args["classifier"]

print(
    f"classifier: {vit_b16_config.classifier}\npos_emb_mode: {vit_b16_config.pos_emb_mode}"
)
vit_classifier = ViTClassifier(vit_b16_config, name="vit_cls_token_pos_sincos")
random_logits = vit_classifier(tf.ones((10, 224, 224, 3)))
print(random_logits.shape)