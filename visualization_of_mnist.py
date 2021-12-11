import tensorflow as tf
import tensorflow_datasets as tfds


ds, ds_info = tfds.load("mnist", split="train", with_info=True)
fig = tfds.visualization.show_examples(ds, ds_info)