#define FASTLED_INTERNAL
#include <FastLED.h>
#define LED_PIN     5
#define NUM_LEDS    38
#define BRIGHTNESS  100
#define LED_TYPE    WS2811
#define COLOR_ORDER GRB
CRGB leds[NUM_LEDS];
String inByte;

#define UPDATES_PER_SECOND 100

void setup() {
  Serial.begin(9600);
  //delay( 3000 ); // power-up safety delay
  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS).setCorrection( TypicalLEDStrip );
  FastLED.setBrightness(  BRIGHTNESS );
  // turn off all leds first
}

void loop() {

    while (Serial.available() == 0) {
      // Wait for User to Input Data
    }
    inByte = Serial.readString(); // read data until newline
    String command = inByte.substring(0, 1);
    String str_data = inByte.substring(1);
    int firstCommaIndex = str_data.indexOf(",");
    int secondCommaIndex = str_data.indexOf(",", firstCommaIndex + 1);
    String cmd = str_data.substring(0, firstCommaIndex);
    String param1 = str_data.substring(firstCommaIndex + 1, secondCommaIndex);
    String param2 = str_data.substring(secondCommaIndex + 1);

    // turn off all leds first
    for (int i = 0; i < NUM_LEDS; i++) {
      leds[i] = CRGB::Black;
    }
    FastLED.show();
    // there is an on instruction, light up the respective leds
    if (command == "O") {
      int x_coord = param1.toInt();
      int y_coord = param2.toInt();
      leds[x_coord] = CRGB::Red;
      leds[y_coord] = CRGB::Red;
    }
    FastLED.show();
    delay(100);
    Serial.println("done");
  
}
