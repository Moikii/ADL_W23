# SHEL: Dataset generation for Single German playing cards and object detection using YOLO

## How does the application work?
We recoginze a card as played if it is detected in 3 frames in a row, to reduce false detections. When
the number of detected cards is the same as the number of players, the winner can be determined. The cards are added
to a list containing the used cards to avoid detecting cards again during the game.


## Why did we not deploy with Streamlit directly?

Using Streamlit to deploy the app did not work with our implementation, because we use an IP-Camera, that is only connected to the local network. Rewriting the code to make it work online would have taken up too much time at this point. Additionally, we did not have a external webcam at hand to capture the videos from a nice angle, which is the reason we went with the phone camera as IP-Webcam in the first place. The video lags a bit, but does not influence the predicitons of the model itself, although it looks not that nice for the user.