import googlemaps
import json
import re
import requests
import base64


def text_to_speech(string, output_file_name):
    API_KEY = "AIzaSyBUl3fxlhMXDshN3SV6MbhRGuhTBpNZ_3c"
    URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={API_KEY}"
    # Define the request payload
    data = {
        "input": {"text": string},
        "voice": {"languageCode": "en-US", "ssmlGender": "NEUTRAL"},
        "audioConfig": {"audioEncoding": "MP3"}
    }

    # Make the API request
    response = requests.post(URL, json=data)

    # Process the response
    if response.status_code == 200:
        audio_content = response.json()["audioContent"]  # Base64 string
        audio_bytes = base64.b64decode(audio_content)  # Decode to binary

        # Save the audio as an MP3 file
        with open(output_file_name, "wb") as audio_file:
            audio_file.write(audio_bytes)
    else:
        print("Error:", response.text)
        raise Exception(response.text)


def getRouteToDest(lat, lon, destination, out_file_name):
    # Initialize the client with your API key
    gmaps = googlemaps.Client(key='AIzaSyDS98MViODka2tWeuzjLm8f9vT7ehgLWwA')

    # Define origin and destination
    origin = (lat, lon)

    # Request directions
    directions_result = gmaps.directions(
        origin,
        destination,
        mode='transit',
        departure_time='now',
        transit_mode = 'bus'
    )

    filename = 'directions_result.json'

    # Write the directions_result to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(directions_result, json_file, indent=4)

    if directions_result:
        steps = directions_result[0]['legs'][0]['steps']
        step = steps[0]
        first_step_str = re.sub(r'<.*?>', '', step["html_instructions"])
        text_to_speech(first_step_str, out_file_name)
        return out_file_name
    else:
        return None

    #print(f'Directions have been saved to {filename}')
    '''
    ctr = 0
    if directions_result:
        steps = directions_result[0]['legs'][0]['steps']
        for step in steps:
            text_instruction = re.sub(r'<.*?>', '', step["html_instructions"])  # Remove HTML tags
            print(text_instruction)
            text_to_speech(text_instruction, str(ctr) + ".mp3")
            ctr += 1
            
            for direction in step['steps']:
                print(direction['html_instructions'])
            
    else:
        print("No directions found.")
    '''