import React, { useState } from 'react';
import { Box, Typography, TextField, Button, styled } from '@mui/material';

const StyledBox = styled(Box)(({ theme }) => ({
  backgroundColor: '#FDEDE0',
  padding: theme.spacing(4),
  borderRadius: '20px',
  border: '3px solid #F37576',
  boxShadow: '0 10px 30px rgba(0, 0, 0, 0.1)',
  width: '100%',
  maxWidth: '500px',
  [theme.breakpoints.down('sm')]: {
    maxWidth: '90%',
    padding: theme.spacing(3),
  },
}));

const StyledButton = styled(Button)({
  backgroundColor: '#610023',
  color: 'white',
  '&:hover': {
    backgroundColor: '#450019',
  },
});

const IceCreamWishForm = ({ onSubmit, onClose }) => {
  const [name, setName] = useState('');
  const [flavors, setFlavors] = useState('');
  const [description, setDescription] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({ name, flavors, description });
  };

  return (
    <Box
      sx={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: '100%',
        display: 'flex',
        justifyContent: 'center',
      }}
    >
      <StyledBox>
        <Typography variant="h5" component="h2" gutterBottom sx={{ color: '#610023', fontWeight: 'bold' }}>
          Add Your Ice Cream Wish
        </Typography>
        <Typography variant="body2" gutterBottom sx={{ color: '#610023', mb: 2 }}>
          Share what you would love to see in our Gelato Joy!
        </Typography>
        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Name of Your Ice Cream"
            value={name}
            onChange={(e) => setName(e.target.value)}
            margin="normal"
            required
            sx={{ 
              '& label': { color: '#610023' }, 
              '& .MuiOutlinedInput-root': { 
                '& fieldset': { borderColor: '#610023' } 
              } 
            }}
          />
          <TextField
            fullWidth
            label="Flavors (comma-separated)"
            value={flavors}
            onChange={(e) => setFlavors(e.target.value)}
            margin="normal"
            required
            sx={{ 
              '& label': { color: '#610023' }, 
              '& .MuiOutlinedInput-root': { 
                '& fieldset': { borderColor: '#610023' } 
              } 
            }}
          />
          <TextField
            fullWidth
            label="Description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            margin="normal"
            required
            multiline
            rows={4}
            sx={{ 
              '& label': { color: '#610023' }, 
              '& .MuiOutlinedInput-root': { 
                '& fieldset': { borderColor: '#610023' } 
              } 
            }}
          />
          <Box 
            sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              flexDirection: { xs: 'column', sm: 'row' }, 
              gap: { xs: 2, sm: 1 }, 
              mt: 2,
            }}
          >
            <Button 
              onClick={onClose} 
              sx={{ 
                color: '#610023',
                width: { xs: '100%', sm: 'auto' }, 
              }}
            >
              Cancel
            </Button>
            <StyledButton 
              type="submit" 
              variant="contained" 
              sx={{ 
                width: { xs: '100%', sm: 'auto' },
                ml: { sm: 2 }, 
              }}
            >
              Submit Your Wish
            </StyledButton>
          </Box>
        </form>
      </StyledBox>
    </Box>
  );
};

export default IceCreamWishForm;
