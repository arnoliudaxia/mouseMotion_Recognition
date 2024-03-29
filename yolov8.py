from ultralytics import YOLO


model = YOLO('runs/classify/train/weights/best.pt')


results = model(r"Z:\Motion_Recognition\Head Twitch\largeSlice\output_000.mp4", stream=True, save=True)


for step, result in enumerate(results):
    # print(f"Step {step} Result: {result.probs.top1}")
    pass
