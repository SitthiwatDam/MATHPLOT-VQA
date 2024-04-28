// UI imports
import * as React from 'react';
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import TextField from '@mui/material/TextField';

import Link from '@mui/material/Link';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import Typography from '@mui/material/Typography';
import { createTheme, ThemeProvider } from '@mui/material/styles';
// Everythings about AI imports
import { useState  } from 'react';
import axios from 'axios'; // Import Axios
import './Main.css'; // Importing CSS file

const defaultTheme = createTheme();

export default function Main() {
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
    <ThemeProvider theme={defaultTheme}>
      <Grid container component="main" sx={{ height: '100vh' }}>
        <CssBaseline />
        <Grid
          item
          xs={false}
          sm={4}
          md={7}
          sx={{
            backgroundImage: 'url(https://source.unsplash.com/random/?chart)',
            backgroundRepeat: 'no-repeat',
            backgroundColor: (t) =>
              t.palette.mode === 'light' ? t.palette.grey[50] : t.palette.grey[900],
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        >
        {imageInput && (
            <img
            src={URL.createObjectURL(imageInput)}
            alt="Input"
            className="input-image"
            style={{ width: '80%', height: '80%', objectFit: 'contain', marginRight:'10%', marginLeft:'10%'}}
            />
        )}
    
        </Grid>
        <Grid item xs={12} sm={8} md={5} component={Paper} elevation={6} square>
          <Box
            sx={{
              my: 8,
              mx: 4,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <Avatar sx={{ m: 1, bgcolor: 'secondary.main' ,width: 50, height:50}} >
              <AutoGraphIcon sx={{width : 30, height : 30 }}/>
            </Avatar>
            <Typography component="h1" variant="h4" marginTop='1%'>
              MATHPLOT VQA
            </Typography>
            <Box component="form" noValidate  sx={{ mt: 1 }}>
              <TextField
                margin="normal"
                required
                fullWidth
                label="User Query"
                name="query"
                autoComplete="query"
                autoFocus
                value={textInput} 
                onChange={handleTextChange}>
                </TextField>
              <input margin="normal" required type="file" accept="image/*" onChange={handleFileChange} />
              <Button
                fullWidth
                onClick={processInput}
                variant="contained"
                sx={{ mt: 3, mb: 2 }}
              >
                Process
              </Button>
              <div className="output-section">
                <Typography component="h1" variant="h4">
                    OUTPUT
                </Typography>
                <textarea rows="4" cols="50" value={outputText} readOnly />
              </div>
              <Typography variant="body2" color="text.secondary" align="center" marginTop='2%'>
                    {'Visit us on '}
                    <Link color="inherit" href="https://github.com/SitthiwatDam/MATHPLOT-VQA">
                    MATHPLOT-VQA github repository
                    </Link>{'  '}
                    {new Date().getFullYear()}
                    {'.'}
              </Typography>
            </Box>
          </Box>
        </Grid>
      </Grid>
    </ThemeProvider>
  );
}