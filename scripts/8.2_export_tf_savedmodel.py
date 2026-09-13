import argparse
from pathlib import Path

import tensorflow as tf
import torch
from torchvision import models

MODELS = Path('models')
EXPORT = Path('export')
EXPORT.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    num_classes = None  # inferred from checkpoint

    # ---- Load PyTorch model ----
    model = models.resnet50()
    ckpt = torch.load(
        MODELS / f'resnet50_pv_{args.strategy}_best.pt',
        map_location='cpu',
    )

    num_classes = ckpt['fc.weight'].shape[0]
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt)
    model.eval()

    # ---- Wrap for TF ----
    class TFWrapper(tf.Module):
        def __init__(self, torch_model):
            super().__init__()
            self.model = torch_model

        @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
        def __call__(self, x):
            x = tf.transpose(x, [0, 3, 1, 2])  # NHWC → NCHW
            x = torch.from_numpy(x.numpy())
            with torch.no_grad():
                y = self.model(x).numpy()
            return tf.convert_to_tensor(y)

    tf_model = TFWrapper(model)

    tf.saved_model.save(
        tf_model,
        EXPORT / 'resnet50_tf_fp32',
    )

    print('[OK] TensorFlow SavedModel exported')


if __name__ == '__main__':
    main()
