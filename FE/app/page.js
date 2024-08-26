'use client'

import { useState } from 'react'
import axios from 'axios'
import IceCreamWishForm from './components/IceCreamWishForm';
import SuccessModal from './components/SuccessModal';
import { 
  Container, 
  Typography, 
  TextField, 
  Button, 
  List, 
  ListItem, 
  ListItemText, 
  Paper,
  Box,
  styled,
  Modal
} from '@mui/material'

const BackgroundContainer = styled(Box)({
  backgroundImage: 'url("/gelato02.svg")',
  backgroundSize: 'cover',
  backgroundAttachment: 'fixed',
  minHeight: '100vh',
  width: '100%',
  padding: '2rem 0',
});

const ContentContainer = styled(Paper)(({ theme }) => ({
  backgroundColor: '#FDEDE0',
  padding: theme.spacing(4),
  borderRadius: '20px',
  border: '5px solid #F37576',
  boxShadow: '0 10px 30px rgba(0, 0, 0, 0.1)',
  position: 'relative',
  overflow: 'visible',
}));

const MeltingIceCream = styled('div')({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  height: '300px',
  backgroundImage: 'url("/gelato01.svg")',
  backgroundSize: 'cover',
  backgroundPosition: 'center top',
  backgroundRepeat: 'no-repeat',
  zIndex: 1,
});

const ContentWrapper = styled(Box)({
  position: 'relative',
  zIndex: 2,
  paddingTop: '160px',
});

const HeaderSection = styled(Box)({
  backgroundColor: 'transparent',
  paddingTop: '0px',
});

const StyledButton = styled(Button)({
  backgroundColor: '#F37576',
  '&:hover': {
    backgroundColor: '#E56565',
  },
});

export default function Home() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [aiResponse, setAiResponse] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [openModal, setOpenModal] = useState(false)
  const [showSuccessModal, setShowSuccessModal] = useState(false) 
  const [successMessage, setSuccessMessage] = useState('') 

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    try {
      const response = await axios.post(`${process.env.NEXT_PUBLIC_BACKEND_URL}/query`, { query })
  
      if (response.data) {
        console.log('Response data:', response.data);  
        setResults(response.data.results || [])
        setAiResponse(response.data.ai_response || 'No response from AI')
      } else {
        console.error('Unexpected response structure:', response.data)
        setResults([])
        setAiResponse('Unexpected response structure')
      }
    } catch (error) {
      console.error('Error querying the backend:', error)
      console.error('Error response:', error.response?.data)  
      setResults([])
      setAiResponse('Error: Unable to fetch AI response')
    } finally {
      setIsLoading(false)
      setQuery('') 
    }
  }

  const handleAddWish = async (formData) => {
    try {
      const response = await axios.post(`${process.env.NEXT_PUBLIC_BACKEND_URL}/add_ice_cream_wish`, formData)
      console.log('Response from server:', response.data);
      if (response.data && response.data.message) {
        setSuccessMessage(response.data.message)
        setShowSuccessModal(true) 
        setOpenModal(false)
      } else {
        console.error('Unexpected response structure:', response.data)
        setSuccessMessage('Unexpected response from server. Please try again.')
        setShowSuccessModal(true) 
      }
    } catch (error) {
      console.error('Error adding ice cream wish:', error.response ? error.response.data : error.message)
      setSuccessMessage('Failed to add ice cream wish. Please try again.')
      setShowSuccessModal(true) 
    }
  }

  
  return (
    <BackgroundContainer>
      <Container maxWidth="md">
        <ContentContainer elevation={3}>
          <MeltingIceCream />
          <ContentWrapper>
            <HeaderSection>
              <Typography variant="h3" component="h1" gutterBottom align="center" color="primary" sx={{ fontWeight: 'bold', mb: 3 }}>
                Gelato Joy
              </Typography>
              <Typography variant="h5" align="center" color="primary" sx={{ mb: 3 }}>
                Joy in every bite, bliss in every scoop!
              </Typography>
              <Typography variant="h6" align="center" color="secondary" sx={{ mb: 4 }}>
                Welcome to our magical world of ice cream! 🍦 Ask me anything about our delicious flavors!
              </Typography>
            </HeaderSection>
            <form onSubmit={handleSubmit}>
              <TextField
                fullWidth
                variant="outlined"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What ice cream flavor are you curious about?"
                multiline
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                sx={{ mb: 2 }}
              />
              <StyledButton 
                type="submit" 
                variant="contained" 
                disabled={isLoading}
                fullWidth
              >
                {isLoading ? "Thinking..." : "Ask me!"}
              </StyledButton>
            </form>
            <StyledButton 
              onClick={() => setOpenModal(true)} 
              variant="contained" 
              fullWidth 
              sx={{ mt: 2 }}
            >
              Add Your Wish
            </StyledButton>
            {aiResponse && (
              <Box mt={3}>
                <Typography variant="h5" gutterBottom color="primary">
                  Gelato Joy says:
                </Typography>
                <Typography>{aiResponse}</Typography>
              </Box>
            )}
            {results.length > 0 && (
              <Box mt={3}>
                <Typography variant="h5" gutterBottom color="primary">
                  Ice Cream Insights:
                </Typography>
                <List>
                  {results.map((result, index) => (
                    <ListItem key={index} divider={index !== results.length - 1}>
                      <ListItemText 
                        primary={`${result.name} (${result.stars} ⭐)`}
                        secondary={
                          <>
                            <Typography component="span" variant="body2" color="text.primary">
                              Flavors: {result.flavors.join(', ')}
                            </Typography>
                            <br />
                            {result.review}
                          </>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </ContentWrapper>
        </ContentContainer>
      </Container>
      <Modal
        open={openModal}
        onClose={() => setOpenModal(false)} 
        aria-labelledby="ice-cream-wish-modal"
        aria-describedby="modal-to-add-ice-cream-wish"
      >
        <IceCreamWishForm 
          onSubmit={handleAddWish}
          onClose={() => setOpenModal(false)}
        />
      </Modal>
      <Modal
        open={showSuccessModal}
        onClose={() => setShowSuccessModal(false)}
        aria-labelledby="success-modal"
        aria-describedby="modal-success-message"
      >
        <SuccessModal 
          message={successMessage}
          onClose={() => setShowSuccessModal(false)}
        />
      </Modal>
    </BackgroundContainer>
  )
}
