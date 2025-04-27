// src/context/BankDataContext.jsx
import React, { createContext, useState, useContext } from 'react';
import initialData from '../data/fakeData';

// Create the context
const BankDataContext = createContext();

// Custom hook to use the bank data context
export const useBankData = () => useContext(BankDataContext);

// Provider component
export const BankDataProvider = ({ children }) => {
  const [bankData, setBankData] = useState(initialData);

  // Function to process a transfer
  const processTransfer = (transferData) => {
    const { fromAccount, toAccount, amount, transferDate, memo } = transferData;
    const amountNum = parseFloat(amount);
    
    // Create a unique transaction ID (in real app would come from backend)
    const newTransactionId = Math.max(...bankData.transactions.map(t => t.id)) + 1;
    
    // Create two new transactions (debit from source, credit to destination)
    const debitTransaction = {
      id: newTransactionId,
      date: transferDate,
      description: memo || `Transfer to ${getAccountName(toAccount)}`,
      amount: -amountNum,
      category: 'Transfer',
      accountId: parseInt(fromAccount)
    };
    
    const creditTransaction = {
      id: newTransactionId + 1,
      date: transferDate,
      description: memo || `Transfer from ${getAccountName(fromAccount)}`,
      amount: amountNum,
      category: 'Transfer',
      accountId: parseInt(toAccount)
    };
    
    // Update account balances
    const updatedAccounts = bankData.accounts.map(account => {
      if (account.id === parseInt(fromAccount)) {
        return { ...account, balance: account.balance - amountNum };
      } else if (account.id === parseInt(toAccount)) {
        return { ...account, balance: account.balance + amountNum };
      }
      return account;
    });
    
    // Update transactions
    const updatedTransactions = [
      ...bankData.transactions,
      debitTransaction,
      creditTransaction
    ];
    
    // Update the context state
    setBankData({
      ...bankData,
      accounts: updatedAccounts,
      transactions: updatedTransactions
    });
    
    return {
      success: true,
      fromAccount: getAccountDetails(fromAccount),
      toAccount: getAccountDetails(toAccount),
      amount: amountNum,
      date: transferDate,
      memo: memo
    };
  };
  
  // Helper function to get account name by ID
  const getAccountName = (accountId) => {
    const account = bankData.accounts.find(acc => acc.id === parseInt(accountId));
    return account ? account.type : 'Unknown Account';
  };
  
  // Helper function to get account details by ID
  const getAccountDetails = (accountId) => {
    return bankData.accounts.find(acc => acc.id === parseInt(accountId)) || {};
  };

  return (
    <BankDataContext.Provider 
      value={{ 
        accounts: bankData.accounts, 
        transactions: bankData.transactions,
        processTransfer,
        getAccountName,
        getAccountDetails
      }}
    >
      {children}
    </BankDataContext.Provider>
  );
};