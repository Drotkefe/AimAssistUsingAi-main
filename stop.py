import dxcam
import time
start_time, fps = time.perf_counter(), 0
cam = dxcam.create()
start = time.perf_counter()
while fps < 1000:
    frame = cam.grab()
    if frame is not None:  # New frame
        fps += 1
end_time = time.perf_counter() - start_time
print(fps/end_time)
