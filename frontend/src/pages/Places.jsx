import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import './PlacePage.css';
import EarthLoader from './EarthLoader';

export default function PlacePage() {
  const navigate = useNavigate();
  const { id } = useParams();

  const [place, setPlace] = useState(null);
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 3000);

    fetchPlaceDetails();
    return () => clearTimeout(timer);
  }, [id]);

  const fetchPlaceDetails = async () => {
    try {
      const res = await fetch(`http://127.0.0.1:8000/place/${id}`);
      if (!res.ok) {
        navigate('/recommend');
        return;
      }
      const data = await res.json();
      setPlace(data);
      fetchImage(data.name);
    } catch (err) {
      console.error('Error fetching place details:', err);
      navigate('/recommend');
    }
  };

  const fetchImage = async (placeName) => {
    const query = placeName.toLowerCase().replace(/\s+/g, '-');
    const accessKey = 'eAUXHbda4vzHJ_ODWtIuKLc2GXwwTDCeFXe0qln7x-c';
    try {
      const res = await fetch(`https://api.unsplash.com/search/photos?page=1&query=${query}`, {
        headers: { Authorization: `Client-ID ${accessKey}` }
      });
      const data = await res.json();
      if (data.results && data.results.length > 0) {
        setImage(data.results[0].urls.regular);
      }
    } catch (err) {
      console.error('Error loading image:', err);
    }
  };

  if (loading) {
    return <EarthLoader />;
  }

  if (!place) {
    return <p>Place not found.</p>;
  }

  return (
    <div className="place-page-wrapper">
      <div className="place-card">
        <button onClick={() => navigate(-1)} className="back-button">← Back</button>
        <div className="place-content">
          {image && (
            <div className="place-image">
              <img src={image} alt={place.name} />
            </div>
          )}
          <div className="place-text">
            <h2>{place.name}</h2>
            <p className="location">{place.city}, {place.state}</p>
            <p><strong>Rating:</strong> {place.rating}</p>
            <p><strong>Best Time To Visit:</strong> {place.Best_Time_to_visit}</p>
            <p><strong>Entrance Fees:</strong> {place.Entrance_Fee}</p>
            <p><strong>Airport with 50km Radius:</strong> {place.Airport_with_50km_Radius}</p>
            <p><strong>DSLR Allowed:</strong> {place.dslr_allowed}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
