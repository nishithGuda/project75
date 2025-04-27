import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { BankDataProvider } from './context/BankDataContext';
import Navbar from './components/Navbar';
import Accounts from './pages/Accounts';
import Transactions from './pages/Transactions';
import Transfer from './pages/Transfer';
import NavigationAssistant from './components/NavigationAssistant';

function App() {
  return (
    <BankDataProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Navbar />
          <main className="container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<Accounts />} />
              <Route path="/transactions" element={<Transactions />} />
              <Route path="/transfer" element={<Transfer />} />
            </Routes>
          </main>
          <NavigationAssistant />
        </div>
      </Router>
    </BankDataProvider>
  );
}

export default App;