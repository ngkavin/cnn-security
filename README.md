# CNN Security

Program for operating a rudimentary security system
on the Raspberry Pi 4. Requires a PIR sensor and camera (legacy stack).
Also requires a TorchScript model at MODEL_PATH and a PushBullet access token
under TOKEN_PATH.

Runs until interrupted. If motion is detected, the camera
passes what it sees into the nerual network. If an authorized user
is not detected within a specified time (default 20 seconds), then
an alert is sent to the owner via PushBullet.
