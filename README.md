# lasergo

This project is a working proof of concept for a computer vision based robot that allows playing go agains a computer on a real board by using a laser to point to where the computer wishes to move on the board. The goal of this project was to get a working project up and running quickly so there is quite a bit of room for improvements and plenty of rough edges in the code. I highly encourage contrinutions and collaboration as it is my hope that the community can take this proof of concept and turn it into something that can be enjoyed by all.

![lasergo in action](https://i.imgur.com/9hwDuZm.jpg)

## setup

- install dependencies from requirements.txt
- attatch servos and laser to one another
![servo configuration](https://i.imgur.com/wqhLF1N.jpg)
- connect servos and laser to power supply. In my case I wired the laser to a 5v DC power supply I had lying around and connected the servos to the Arduino's 5v and 3v out pins.
- connect servo to Arduino. In my case I used pins 8 and 9 for the servo data.
![arduino configuration](https://i.imgur.com/kMMM1Ip.jpg)
- connect webcam to computer
- set default laser position in the code
- **if you are running this on linux or something other than windows you will need to replace the gnugo executable with the correct binary for your operating system**

## usage

- position an empty go board in frame of the webcam
- to start lasergo run `python3 .\boardProcess.py` this will open the program and take a picture with your webcam
- click on the 4 corners of the go board starting with the top left of the image and going counter clockwise. the left mouse button will add a point and the right mouse button will remove a point
![marking the corners](https://i.imgur.com/EUSc4g6.png)
- once the 4 corners have been selected the image will turn into a top down perspective of the board. After this happens click on `initialize board`.This will bring up squares representing the automatically detected intersections you can manually correct them by using left mouseclick to add an intersection or right mouseclick to remove an intersection. To end the corrections press `ESC` and close the window with the detected intersections.
![detecting intersections](https://i.imgur.com/NrL6hmJ.png)
- you can visualize the intersections on the original image by clicking on the `visualize grid` button
![visualize grid](https://i.imgur.com/OR15COY.png)
- You're now all set to play a game. Simply play a move and then press the `capture frame` button this will detect the position of the boards and show the computer's response close those windows iwth `ESC` and a video feed of the laser moving towards te target will appear. Once you have placed the computer's stone close the window with `ESC`
![board state](https://i.imgur.com/ZBiRnsU.png)
![laser tracking](https://i.imgur.com/PYbulJn.png)
