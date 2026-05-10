# LeLamp Demo Recording Script

Target length: 60-120 seconds.

1. Open `http://localhost:8000` and show the UI overlay.
2. Click `Start Camera`; if local ML dependencies are unavailable, use `Mock Engaged` and `Mock Away`.
3. Look at the lamp for 8-10 seconds. Narrate that engagement starts and the lamp brightens/tracks.
4. Look away or click `Mock Away`. Wait through head wiggle, light pulse, and chirp.
5. Put a cup/book/laptop in view. If YOLO is unavailable, click `Mock Cup Memory`. Show the memory toast.
6. Ask via text or push-to-talk: `Where is my cup?`
7. Show the grounded answer with location and latency.
8. End on the README metrics table or run `python eval/report.py`.

What to emphasize: this is not a generic chatbot; the answer is grounded in visual memory stored by the perception pipeline.

