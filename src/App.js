import React, { useEffect, useState } from "react";
import './styles.css';

export default function App() {
  const [location, setLocation] = useState(null);
  const [destination, setDestination] = useState("");
  const [error, setError] = useState(null);
  const [audioSrc, setAudioSrc] = useState(null); // State to store the MP3 file URL
  const [isFetching, setIsFetching] = useState(false); // Track if fetching is active

  // Get location of the user on page load
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude });
        },
        (err) => setError(err.message)
      );
    } else {
      setError("Geolocation not supported");
    }
  }, []);

  // Update destination input
  const handleDestinationChange = (e) => {
    setDestination(e.target.value);
  };

  // Submit the form to fetch MP3 file from backend
  const handleSubmit = async () => {
    setIsFetching(true); // Start fetching after submit
    if (location && destination) {
      try {
        const response = await fetch("http://localhost:5000/get-mp3", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            location,
            destination,
          }),
        });

        if (response.ok) {
          const blob = await response.blob(); // Convert the response to a Blob
          const audioUrl = URL.createObjectURL(blob); // Create a URL for the Blob
          setAudioSrc(audioUrl); // Set the audio source to the Blob URL
        } else {
          throw new Error("Error fetching MP3 file.");
        }
      } catch (err) {
        console.error("Error connecting to server:", err);
        setError("Error connecting to server");
      }
    } else {
      setError("Please make sure location and destination are provided.");
    }
    setIsFetching(false); // Stop fetching after the operation
  };

  return (
    <div className="container">
      <h1 className="heading">Web Project</h1>
      {location && (
        <p className="location">
          Location: {location.lat}, {location.lon}
        </p>
      )}
      <input
        type="text"
        placeholder="Enter destination"
        value={destination}
        onChange={handleDestinationChange}
        className="input"
      />
      <button onClick={handleSubmit} className="button">
        Submit
      </button>
      {error && <p className="error">Error: {error}</p>}

      {/* Display webcam feed from the Python backend */}
      <div className="video-container">
        <img
          src="http://localhost:5000/video_feed"
          alt="Webcam Stream"
          className="webcam-stream"
        />
      </div>

      {/* Display the audio player if audio is available */}
      {audioSrc && (
        <div className="audio-container">
          <audio controls>
            <source src={audioSrc} type="audio/mp3" />
            Your browser does not support the audio element.
          </audio>
        </div>
      )}
    </div>
  );
}
