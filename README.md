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

How its currently done:
(1) Star identification utilizing modified triangle algorithms: This algorithm uses geometric shapes and angles to identify stars in the sky. It works by comparing the angles between three stars in the sky to a database of known star patterns to determine the identity of the stars. This algorithm is computationally efficient and is suitable for use in onboard navigation systems of spacecrafts [1]. (2) Star identification utilizing star patterns: This algorithm uses a database of star patterns to identify stars in the sky. It works by comparing the star pattern in the sky to a database of known star patterns to determine the identity of the stars




Future V2 - Implement a real time tracker