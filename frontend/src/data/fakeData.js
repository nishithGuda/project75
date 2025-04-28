// src/data/fakeData.js
const data = {
  accounts: [
    {
      id: 1,
      type: "Checking",
      number: "****4321",
      balance: 2456.78,
      currency: "$",
    },
    {
      id: 2,
      type: "Savings",
      number: "****7890",
      balance: 12345.67,
      currency: "$",
    },
    {
      id: 3,
      type: "Credit Card",
      number: "****5678",
      balance: 742.50,
      currency: "$",
    },
    {
      id: 4,
      type: "Checking",
      number: "****2345",
      balance: 3678.92,
      currency: "$",
    },
    {
      id: 5,
      type: "Savings",
      number: "****6789",
      balance: 24897.54,
      currency: "$",
    }
  ],
  transactions: [
    { id: 1, date: '2025-04-22', description: 'Grocery Store', amount: -85.42, category: 'Food', accountId: 1 },
    { id: 2, date: '2025-04-21', description: 'Salary Deposit', amount: 2400.00, category: 'Income', accountId: 1 },
    { id: 3, date: '2025-04-20', description: 'Electric Bill', amount: -75.40, category: 'Utilities', accountId: 1 },
    { id: 4, date: '2025-04-18', description: 'Coffee Shop', amount: -4.50, category: 'Food', accountId: 1 },
    { id: 5, date: '2025-04-15', description: 'Transfer to Savings', amount: -500.00, category: 'Transfer', accountId: 1 },
    { id: 6, date: '2025-04-15', description: 'Transfer from Checking', amount: 500.00, category: 'Transfer', accountId: 2 },
    { id: 7, date: '2025-04-10', description: 'Interest', amount: 12.67, category: 'Income', accountId: 2 },
    { id: 8, date: '2025-04-17', description: 'Restaurant', amount: -62.95, category: 'Food', accountId: 3 },
    { id: 9, date: '2025-04-14', description: 'Online Shopping', amount: -49.99, category: 'Shopping', accountId: 3 },
    { id: 10, date: '2025-04-11', description: 'Gas Station', amount: -35.00, category: 'Transportation', accountId: 3 },
  ]
};

export default data;