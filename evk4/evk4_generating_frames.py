import cv2
import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm, PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, Window


def frame_generation(mode="periodic"):
    if mode == "periodic":
        with Window("Periodic frame generator", width, height, Window.RenderMode.BGR) as window:
            # Do something whenever a frame is ready
            def periodic_cb(ts, frame):
                cv2.putText(frame, "Timestamp: " + str(ts), (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
                window.show(frame)

            # Instantiate the frame generator
            periodic_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=5000, fps=100)
            periodic_gen.set_output_callback(periodic_cb)

            for evs in mv_iterator:
                if evs.size == 0:
                    continue
                else:
                    max_t = evs['t'][-1]
                    print(f"timestamps: = {max_t}")
                    EventLoop.poll_and_dispatch()  # Dispatch system events to the window
                    periodic_gen.process_events(evs)
                    # Feed events to the frame generator
                    if window.should_close():
                        break
    elif mode == "demand":         
        with Window("OnDemand frame generator", width, height, Window.RenderMode.BGR) as window:
            # Instantiate the frame generator
            on_demand_gen = OnDemandFrameGenerationAlgorithm(width, height, accumulation_time_us=4000)

            frame_period_us = int(1e6/100)  # 50 FPS
            next_processing_ts = frame_period_us
            frame = np.zeros((height, width, 3), np.uint8)
            for evs in mv_iterator:
                if evs.size == 0:
                    continue
                else:
                    EventLoop.poll_and_dispatch()  # Dispatch system events to the window

                    on_demand_gen.process_events(evs)  # Feed events to the frame generator

                    ts = evs["t"][-1] # Trigger new frame generations as long as the last event is high enough
                    while(ts > next_processing_ts):
                        on_demand_gen.generate(next_processing_ts, frame)
                        cv2.putText(frame, "Timestamp: " + str(next_processing_ts),
                                    (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
                        window.show(frame)
                        next_processing_ts += frame_period_us

                    if window.should_close():
                        break
                
                
if __name__ == "__main__":
    # Events iterator on Camera
    mv_iterator = EventsIterator(input_path=r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\evk4\sample_prepare_Nov_03_clean\MR\run_32\recording_251103_180549_980341.raw", delta_t=1e3)
    height, width = mv_iterator.get_size()  # Camera Geometry
    frame_generation()