import React from 'react';
import ReactDOM from 'react-dom/client';
import { MantineProvider,createTheme } from '@mantine/core';
import '@mantine/core/styles.css';
import './style.css';
import App from './App';
const theme=createTheme({fontFamily:'Inter, Segoe UI, sans-serif',primaryColor:'blue',defaultRadius:'md',colors:{dark:['#d9e1ee','#a8b5c9','#8491a6','#46536a','#303c50','#222d40','#182233','#121b29','#0e1622','#0a111b']}});
ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><MantineProvider theme={theme} defaultColorScheme="dark"><App/></MantineProvider></React.StrictMode>);
