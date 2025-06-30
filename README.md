StarSense is a tool used to tell the satellite how it’s oriented in 3D space using background reference, and with no GPS or compass in orbit. This project is meant to reflect a Star-Tracker, But I am curious whether we can push this futher especailly with the recent advancements in AI.

What the Project Does, Step-by-Step:
  User uploads an image of the night sky (could be taken by a phone, telescope, or simulated).
    
    Program:
    Detects stars in the image.
    Matches those stars to known stars from a star catalog (like Hipparcos).
    Figures out which direction the camera was pointing when the photo was taken.
    
    Output:
    A 3D rotation (quaternion or matrix) that says: “This camera was pointing toward these celestial coordinates.”
      >show a globe or sky sphere with an arrow indicating the direction.
      >label known stars or constellations in the image.
