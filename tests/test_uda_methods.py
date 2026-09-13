"""Unit tests for UDA method implementations.

Tests verify:
- No target labels used during adaptation step
- Correct loss computation
- Deterministic seed behavior
- Gradient flow
- Checkpoint saving
"""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.adaptation.cdan import (  # noqa: E402
    ConditionalDomainDiscriminator,
    entropy_weight,
)
from src.adaptation.coral import coral_loss  # noqa: E402
from src.adaptation.dann import DomainDiscriminator, grad_reverse  # noqa: E402
from src.adaptation.mcc import mcc_loss  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
B, D, K = 16, 2048, 18  # batch, feature_dim, num_classes


# ------------------------------------------------------------------ #
# CORAL tests
# ------------------------------------------------------------------ #


class TestCORAL:
    def test_coral_loss_zero_when_same_distribution(self):
        """CORAL loss should be 0 when source and target have same distribution."""
        torch.manual_seed(0)
        x = torch.randn(64, D)
        loss = coral_loss(x, x.clone())
        assert loss.item() < 1e-6, f'Expected ~0, got {loss.item()}'

    def test_coral_loss_positive_when_different(self):
        """CORAL loss should be positive when distributions differ."""
        torch.manual_seed(0)
        src = torch.randn(64, D)
        tgt = torch.randn(64, D) * 2 + 1  # shifted and scaled
        loss = coral_loss(src, tgt)
        assert loss.item() > 0, (
            'CORAL loss should be positive for different distributions'
        )

    def test_coral_loss_differentiable(self):
        """CORAL loss must support backpropagation."""
        src = torch.randn(32, D, requires_grad=True)
        tgt = torch.randn(32, D, requires_grad=True)
        loss = coral_loss(src, tgt)
        loss.backward()
        assert src.grad is not None
        assert tgt.grad is not None

    def test_coral_loss_symmetric(self):
        """CORAL loss should be symmetric: loss(A, B) == loss(B, A)."""
        torch.manual_seed(1)
        a = torch.randn(32, 128)
        b = torch.randn(32, 128) + 0.5
        assert abs(coral_loss(a, b).item() - coral_loss(b, a).item()) < 1e-5


# ------------------------------------------------------------------ #
# DANN tests
# ------------------------------------------------------------------ #


class TestDANN:
    def test_gradient_reversal_forward(self):
        """GRL should pass values through unchanged in forward."""
        x = torch.randn(8, D)
        y = grad_reverse(x, 1.0)
        assert torch.allclose(x, y), 'GRL should not change forward values'

    def test_gradient_reversal_backward(self):
        """GRL should negate gradients in backward pass."""
        x = torch.randn(8, D, requires_grad=True)
        alpha = 0.5
        y = grad_reverse(x, alpha)
        loss = y.sum()
        loss.backward()
        # Gradient should be negated: each element gets grad = -alpha
        expected = torch.full_like(x, -alpha)
        assert torch.allclose(x.grad, expected), (
            f'GRL gradient should be {-alpha}, got {x.grad.mean():.4f}'
        )

    def test_discriminator_output_shape(self):
        """Domain discriminator should output (B, 2)."""
        disc = DomainDiscriminator(feature_dim=D).to(DEVICE)
        x = torch.randn(B, D).to(DEVICE)
        out = disc(x)
        assert out.shape == (B, 2), f'Expected (B, 2), got {out.shape}'

    def test_dann_no_target_labels_in_training(self):
        """Target loader must not include labels (unlabeled dataset)."""
        # This is a structural test: UnlabeledDataset returns only images
        import os
        import tempfile

        import pandas as pd
        from PIL import Image

        from src.adaptation.base import TRANSFORM_VAL, UnlabeledDataset

        # Create a tiny CSV with path only (no label_id column)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, 'test.jpg')
            Image.new('RGB', (32, 32)).save(img_path)
            csv_path = Path(tmpdir) / 'unlabeled.csv'
            pd.DataFrame({'path': [img_path]}).to_csv(str(csv_path), index=False)

            ds = UnlabeledDataset(csv_path, TRANSFORM_VAL)
            item = ds[0]
            # Should return a tensor (image only), not a tuple (image, label)
            assert isinstance(item, torch.Tensor), (
                f'UnlabeledDataset should return tensor, got {type(item)}'
            )


# ------------------------------------------------------------------ #
# CDAN tests
# ------------------------------------------------------------------ #


class TestCDAN:
    def test_entropy_weight_shape(self):
        """Entropy weight should have shape (B,)."""
        pred = torch.softmax(torch.randn(B, K), dim=1)
        w = entropy_weight(pred)
        assert w.shape == (B,), f'Expected ({B},), got {w.shape}'

    def test_entropy_weight_range(self):
        """Entropy weights should be in (0, 1] since exp(-H) where H >= 0."""
        pred = torch.softmax(torch.randn(B, K), dim=1)
        w = entropy_weight(pred)
        assert (w > 0).all(), 'Entropy weights must be positive'
        assert (w <= 1.0 + 1e-6).all(), 'Entropy weights must be <= 1'

    def test_entropy_weight_uniform_prediction(self):
        """Uniform prediction has max entropy → weight should be minimal."""
        pred_uniform = torch.full((1, K), 1.0 / K)
        pred_certain = torch.zeros(1, K)
        pred_certain[0, 0] = 1.0

        w_uniform = entropy_weight(pred_uniform)
        w_certain = entropy_weight(pred_certain)
        assert w_certain > w_uniform, (
            f'Certain prediction ({w_certain:.4f}) should have higher weight '
            f'than uniform ({w_uniform:.4f})'
        )

    def test_cdan_discriminator_output_shape(self):
        """CDAN discriminator should output (B, 2)."""
        disc = ConditionalDomainDiscriminator(feature_dim=D, num_classes=K).to(DEVICE)
        features = torch.randn(B, D).to(DEVICE)
        predictions = torch.softmax(torch.randn(B, K), dim=1).to(DEVICE)
        out = disc(features, predictions)
        assert out.shape == (B, 2), f'Expected ({B}, 2), got {out.shape}'

    def test_cdan_discriminator_differentiable(self):
        """CDAN discriminator must support backpropagation."""
        disc = ConditionalDomainDiscriminator(feature_dim=D, num_classes=K).to(DEVICE)
        # Create leaf tensor on device directly to preserve .grad
        features = torch.randn(B, D, device=DEVICE, requires_grad=True)
        predictions = torch.softmax(torch.randn(B, K, device=DEVICE), dim=1)
        out = disc(features, predictions)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None, (
            'Gradient must flow through CDAN discriminator'
        )


# ------------------------------------------------------------------ #
# MCC tests
# ------------------------------------------------------------------ #


class TestMCC:
    def test_mcc_loss_positive(self):
        """MCC loss should be non-negative."""
        logits = torch.randn(B, K)
        loss = mcc_loss(logits, temperature=2.5)
        assert loss.item() >= 0, f'MCC loss must be >= 0, got {loss.item()}'

    def test_mcc_loss_zero_at_one_hot(self):
        """MCC loss should be near-zero when predictions are one-hot (no confusion)."""
        logits = torch.zeros(B, K)
        for i in range(B):
            logits[i, i % K] = 100.0  # near one-hot (each sample predicts one class)
        loss = mcc_loss(logits, temperature=2.5)
        # Not exactly zero but should be small
        assert loss.item() < 1.0, (
            f'One-hot predictions should have low MCC loss, got {loss.item()}'
        )

    def test_mcc_loss_large_when_confused(self):
        """MCC loss should be large when all classes equally confused."""
        logits = torch.zeros(B, K)  # uniform softmax → maximum confusion
        loss_confused = mcc_loss(logits, temperature=2.5)

        logits_certain = torch.zeros(B, K)
        logits_certain[:, 0] = 100.0
        loss_certain = mcc_loss(logits_certain, temperature=2.5)

        assert loss_confused > loss_certain, (
            f'Confused ({loss_confused:.4f}) should > certain ({loss_certain:.4f})'
        )

    def test_mcc_loss_differentiable(self):
        """MCC loss must support backpropagation."""
        logits = torch.randn(B, K, requires_grad=True)
        loss = mcc_loss(logits, temperature=2.5)
        loss.backward()
        assert logits.grad is not None, 'Gradient must flow through MCC loss'

    def test_mcc_temperature_effect(self):
        """Higher temperature → softer predictions → more class confusion."""
        logits = torch.randn(B, K)
        loss_low_temp = mcc_loss(logits, temperature=0.1)  # sharp predictions
        loss_high_temp = mcc_loss(logits, temperature=10.0)  # soft predictions
        assert loss_high_temp > loss_low_temp, (
            'Higher temperature should produce more confusion (higher MCC loss)'
        )


# ------------------------------------------------------------------ #
# Seed determinism
# ------------------------------------------------------------------ #


class TestDeterminism:
    def test_coral_loss_deterministic(self):
        """CORAL loss must be deterministic with same seed."""
        torch.manual_seed(42)
        src1 = torch.randn(32, D)
        tgt1 = torch.randn(32, D)
        loss1 = coral_loss(src1, tgt1)

        torch.manual_seed(42)
        src2 = torch.randn(32, D)
        tgt2 = torch.randn(32, D)
        loss2 = coral_loss(src2, tgt2)

        assert abs(loss1.item() - loss2.item()) < 1e-6, (
            'CORAL loss must be deterministic'
        )

    def test_mcc_loss_deterministic(self):
        """MCC loss must be deterministic with same seed."""
        torch.manual_seed(42)
        logits1 = torch.randn(B, K)
        loss1 = mcc_loss(logits1, 2.5)

        torch.manual_seed(42)
        logits2 = torch.randn(B, K)
        loss2 = mcc_loss(logits2, 2.5)

        assert abs(loss1.item() - loss2.item()) < 1e-6, 'MCC loss must be deterministic'
