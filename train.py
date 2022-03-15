# USAGE
# python train.py --classifier token --position learn
# python train.py --classifier token --position sincos
# python train.py --classifier gap --position learn
# python train.py --classifier gap --position sincos

# import the necessary packages
import tensorflow as tf
import argparse
from vit import (
    get_config,
    ViTClassifier,
)

import pprint
pp = pprint.PrettyPrinter(indent=4)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classifier", required=True,
    help="token or gap for the vit representation")
ap.add_argument("-p", "--position", required=True,
    help="learn or sincos for positional embedding")
args = vars(ap.parse_args())

# manipulating the configuration
vit_b16_config = get_config()
with vit_b16_config.unlocked():
    vit_b16_config.pos_emb_mode = args["position"]
    vit_b16_config.classifier = args["classifier"]

print("CONFIG 🚀")
pp.pprint(vit_b16_config)

vit_classifier = ViTClassifier(vit_b16_config, name="vit_cls_token_pos_sincos")

# With training True
random_logits = vit_classifier(
    inputs=tf.ones((10, 224, 224, 3)),
    training=True
)
print(random_logits.shape)

# With training False
random_logits, random_attention_scores = vit_classifier(
    inputs=tf.ones((10, 224, 224, 3)),
    training=False
)
for name, random_attention_score in random_attention_scores.items():
    print(name, random_attention_score.shape)