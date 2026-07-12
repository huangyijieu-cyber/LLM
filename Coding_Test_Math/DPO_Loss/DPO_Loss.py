from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Coding_Test_Math.Examples.loss.DPOLoss import dpo_loss

__all__ = ["dpo_loss"]
