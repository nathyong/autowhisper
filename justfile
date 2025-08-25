_:
    just -l

setup:
    uv pip install 'onnx-asr[cpu,hub]'

run:
    python main.py --server
