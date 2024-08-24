// components/SuccessModal.js
import React from 'react';
import { Box, Typography, Button, styled } from '@mui/material';

const ModalContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  backgroundColor: '#FDEDE0',
  padding: theme.spacing(4),
  borderRadius: '20px',
  border: '3px solid #F37576',
  boxShadow: '0 10px 30px rgba(0, 0, 0, 0.1)',
  width: '90%',
  maxWidth: '400px',
  textAlign: 'center',
}));

const StyledButton = styled(Button)({
  backgroundColor: '#610023',
  color: 'white',
  '&:hover': {
    backgroundColor: '#450019',
  },
});

const SuccessModal = ({ message, onClose }) => (
  <ModalContainer>
    <Typography variant="h6" gutterBottom sx={{ color: '#610023' }}>
      {message}
    </Typography>
    <StyledButton onClick={onClose}>Close</StyledButton>
  </ModalContainer>
);

export default SuccessModal;
