## Python implementation for CV arm height

### Notes
- pip install requirements.txt
- **note, python 3.9-3.12 compatible with google's [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/setup_python)**
- Might need to bypass process-level policy
- Runs in windows powershell to access camera<br>

### Setup
1. Install compatible python (3.12 works)
2. In powershell, setup environment
```
python -m venv .venv
py -3.12 -m venv .venv
pip install numpy opencv-python mediapipe
```
3. Activate environment
```
.venv\Scripts\Activate
```
4. Might need to change the execution policy
```
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```
5. Run `vision.py` script & hope everything is working

### Updates
Currently, mediapipe version is working. Able to calculate arm angles with reasonable accuracy

### TODO:
- [x] time startup
- [x] optimize startup
- [ ] try other CV implementation
