"""Compatibility entry point for the bounded-memory LANL example builder."""

from .streaming_examples import FEATURES, generate, main

build_examples = generate

if __name__ == "__main__":
    main()
