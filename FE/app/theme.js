'use client'

import { createTheme } from '@mui/material/styles'

const theme = createTheme({
  palette: {
    primary: {
      main: '#610023',
    },
    secondary: {
      main: '#C27F79',
    },
    background: {
      default: '#C27F79',
      paper: '#FFD8B1',
    },
    border: '#C27F79',
  },
  typography: {
    fontFamily: 'inherit',
    h1: {
      fontWeight: 700,
    },
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          borderColor: '#C27F79', 
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
            '& fieldset': {
              borderColor: '#C27F79', 
            },
            '&:hover fieldset': {
              borderColor: '#C27F79', 
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          border: '1px solid #C27F79', // Add border to Paper components
        },
      },
    },
    MuiListItem: {
      styleOverrides: {
        root: {
          borderBottom: '1px solid #C27F79', // Add border to list items
          '&:last-child': {
            borderBottom: 'none', // Remove border from last list item
          },
        },
      },
    },
  },
})

export default theme