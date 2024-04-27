import React, { useState } from 'react';
import axios from 'axios'; // Import Axios
import './Main.css'; // Importing CSS file

function Main() {
  const [textInput, setTextInput] = useState('');
  const [imageInput, setImageInput] = useState(null); // Changed initial state to null
  const [outputText, setOutputText] = useState('');

  const handleTextChange = (e) => {
    setTextInput(e.target.value);
  };

  const handleFileChange = (e) => {
    setImageInput(e.target.files[0]); // Store the file object itself
  };

  const processInput = async () => {
    const formData = new FormData();
    formData.append('text', textInput);
    formData.append('image', imageInput); // Now 'imageInput' contains the file data
    
    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 600000, // Adjust the timeout value as needed (in milliseconds)
      });
  
      if (response.status !== 200) {
        throw new Error('Failed to upload image');
      }
  
      const responseData = response.data;
      console.log(responseData['text-generated']);
      setOutputText(responseData['text-generated']);
    } catch (error) {
      console.error('Error uploading image:', error);
      setOutputText(error.message);
    }
  };

  return (
    <div className="container">
      <h1>MATH APP</h1>
      <div className="input-section">
        <label>Text Input:</label>
        <input type="text" name="query" value={textInput} onChange={handleTextChange} />
      </div>
      <div className="input-section">
        <label>Image Input:</label>
        <input type="file" accept="image/*" onChange={handleFileChange} />
      </div>
      <button onClick={processInput}>Process</button>
      <div className="output-section">
        <h2>Output:</h2>
        <textarea rows="4" cols="50" value={outputText} readOnly />
        {imageInput && <img src={URL.createObjectURL(imageInput)} alt="Input" className="input-image" />}
      </div>
    </div>
  );
}

export default Main;
