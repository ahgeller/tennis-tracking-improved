"""The TensorRT engines and the ball detector backends: the session the engines share,
locating engine files, decoding their outputs, the ball, player and court detectors, and
``build_engines`` (``python -m tennis_tracker.engines.build_engines``), which exports the YOLO
checkpoints to ONNX and builds TensorRT engines from them."""
