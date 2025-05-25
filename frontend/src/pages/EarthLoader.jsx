import React from 'react';
import './EarthLoader.css';

export default function EarthLoader() {
  return (
    <div className="loader-overlay">
      <img
        src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Earth_Western_Hemisphere_transparent_background.png/1024px-Earth_Western_Hemisphere_transparent_background.png"
        alt="Rotating Earth"
        className="rotating-earth"
      />
    </div>
  );
}
