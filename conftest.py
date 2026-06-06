"""
conftest.py — makes the project root importable during tests.

Its mere presence at the project root tells pytest to add this directory to
sys.path, so tests in tests/ can `import plume`, `import processing`, etc.
without any extra setup. No fixtures needed yet.
"""
