"""Entry point for python -m guardian (installed via pip)."""
from guardian.cli import main as _main
import sys
sys.exit(_main())
