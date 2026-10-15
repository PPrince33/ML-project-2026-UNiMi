"""Use matplotlib's file-only backend in tests: no windows, works headless."""
import matplotlib

matplotlib.use("Agg")
